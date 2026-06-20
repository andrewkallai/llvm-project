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
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Analysis/MLModelRunner.h"
#include <functional>
#include <memory>
#include <vector>

namespace llvm {
class MLModelRunner;
} // namespace llvm

namespace mlir {
class Operation;
class Region;

class MLIRInlineAdvice;
class MLIRInlineAdvisor;

/// Factory function: creates an MLIRInlineAdvisor that delegates evaluation
/// to the provided `runnerFactory`.
///
/// \param op            The top-level operation containing the callables.
/// \param cg            The call graph for the operation.
/// \param runnerFactory Callback that receives the feature descriptors and
///                      returns an MLModelRunner (or nullptr if unavailable).
std::unique_ptr<MLIRInlineAdvisor>
createMLIRInlineAdvisor(
    Operation *op, CallGraph &cg,
    std::function<std::unique_ptr<llvm::MLModelRunner>(
        const std::vector<llvm::TensorSpec> &)> runnerFactory);

/// Convenience factory for release mode.  Returns nullptr when no compiled
/// model is available.
std::unique_ptr<MLIRInlineAdvisor> getReleaseModeMLIRAdvisor(
    Operation *op, CallGraph &cg,
    std::function<bool(CallOpInterface, Operation *)> getDefaultAdvice);

} // namespace mlir

#endif // MLIR_ANALYSIS_MLINLINEADVISOR_H
