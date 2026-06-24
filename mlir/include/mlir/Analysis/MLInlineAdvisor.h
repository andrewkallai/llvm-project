//===- MLInlineAdvisor.h - ML-based InlineAdvisor for MLIR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLInlineAdvisor for MLIR, which uses a machine
// learning model to drive inlining decisions.  Features are extracted from
// MLIR's Region/Operation/CallGraph infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_MLINLINEADVISOR_H
#define MLIR_ANALYSIS_MLINLINEADVISOR_H

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace llvm {
class MLModelRunner;
} // namespace llvm

namespace mlir {
class Operation;
class Region;

class MLIRInlineAdvisor;

/// Gather simple structural statistics for every operation inside a Region.
struct RegionProperties {
  int64_t blockCount = 0;
  int64_t regionCount = 0;
  int64_t operandCount = 0;
  int64_t resultCount = 0;
  int64_t entryBlockArgCount = 0;
  bool isIsolatedFromAbove = false;

  static RegionProperties compute(Region *region);
};

/// An advice object that records feature values at decision time and updates
/// internal advisor state after (un)successful inlining.
class MLIRInlineAdvice {
public:
  MLIRInlineAdvice(MLIRInlineAdvisor *advisor, Operation *callOp,
                   Operation *callerOp, Region *calleeRegion,
                   bool recommendation,
                   const std::vector<int64_t> &featureValues);
  ~MLIRInlineAdvice() = default;

  /// Must be called when inlining succeeded.
  void recordInlining(bool calleeDeleted);

  /// Must be called when inlining was attempted but failed.
  void recordUnsuccessfulInlining();

  /// Must be called when inlining was never attempted.
  void recordUnattemptedInlining();

  bool isInliningRecommended() const { return recommendation; }
  Operation *getCallOp() const { return callOp; }
  Operation *getCallerOp() const { return callerOp; }
  Operation *getCalleeOp() const;
  Region *getCalleeRegion() const { return calleeRegion; }

private:
  MLIRInlineAdvisor *advisor;
  Operation *callOp;
  Operation *callerOp;
  Region *calleeRegion;
  bool recommendation;
  /// Snapshot of the caller's region properties before inlining.
  RegionProperties preInlineCallerProps;
  std::vector<int64_t> featureValues;
};

/// The ML-based inlining advisor for MLIR.  This mirrors the LLVM
/// MLInlineAdvisor architecture but extracts features from MLIR's Region /
/// Operation / CallGraph infrastructure instead of LLVM IR.
class MLIRInlineAdvisor {
public:
  MLIRInlineAdvisor(Operation *op, CallGraph &cg,
                    std::function<std::unique_ptr<llvm::MLModelRunner>(
                        const std::vector<llvm::TensorSpec> &)>
                        runnerFactory);

  ~MLIRInlineAdvisor() = default;

  /// Evaluate the model for a call site.
  std::unique_ptr<MLIRInlineAdvice>
  getAdvice(CallOpInterface callOp, Operation *callerOp, Region *calleeRegion);

  /// Notification that inlining succeeded (called from the advice object).
  void onSuccessfulInlining(MLIRInlineAdvice &advice, bool calleeDeleted);

  /// Accessors.
  const std::vector<llvm::TensorSpec> &getFeatureMap() const {
    return featureMap;
  }
  RegionProperties getCachedProps(Region *region);

  bool isForcedToStop() const { return forceStop; }

private:
  /// The ML model runner.
  std::unique_ptr<llvm::MLModelRunner> runner;

  /// The feature map (descriptors).
  std::vector<llvm::TensorSpec> featureMap;

  /// Pointers back to the inliner context.
  Operation *op;
  CallGraph &cg;

  /// Module-level graph statistics.
  int64_t graphNodeCount = 0;
  int64_t graphEdgeCount = 0;

  /// Whether the advisor has decided to stop recommending inlining.
  bool forceStop = false;

  /// Initial total operation count across all callables.
  int64_t initialTotalOps = 0;
  /// Current total operation count.
  int64_t currentTotalOps = 0;

  /// Evaluate the model with the current feature buffer.
  bool evaluateModel();

  /// Recompute module-level call-graph statistics.
  void recomputeGraphStats();

  /// Per-callable call-site height (distance from leaf).
  llvm::DenseMap<Region *, unsigned> regionLevels;

  /// A cache of per-region structural properties.
  llvm::DenseMap<Region *, RegionProperties> propsCache;
};

/// Factory function: creates an MLIRInlineAdvisor that delegates evaluation
/// to the provided `runnerFactory`.
///
/// \param op            The top-level operation containing the callables.
/// \param cg            The call graph for the operation.
/// \param runnerFactory Callback that receives the feature descriptors and
///                      returns an MLModelRunner (or nullptr if unavailable).
std::unique_ptr<MLIRInlineAdvisor>
createMLIRInlineAdvisor(Operation *op, CallGraph &cg,
                        std::function<std::unique_ptr<llvm::MLModelRunner>(
                            const std::vector<llvm::TensorSpec> &)>
                            runnerFactory);

/// Convenience factory for release mode.  Returns nullptr when no compiled
/// model is available.
std::unique_ptr<MLIRInlineAdvisor> getReleaseModeMLIRAdvisor(
    Operation *op, CallGraph &cg,
    std::function<bool(CallOpInterface, Operation *)> getDefaultAdvice);

} // namespace mlir

#endif // MLIR_ANALYSIS_MLINLINEADVISOR_H
