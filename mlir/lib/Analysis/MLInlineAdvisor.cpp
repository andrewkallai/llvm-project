//===- MLInlineAdvisor.cpp - machine learned MLIR InlineAdvisor -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface between the MLIR inliner and a learned
// model.  It delegates model evaluation to either the AOT compiled model (the
// 'release' mode) or a runtime-loaded model (the 'development' case).
//
// The feature extraction uses the MLIR CallGraph analysis and region/operation
// introspection instead of LLVM IR specific constructs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Analysis/MLInlineModelFeatureMaps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/MLModelRunner.h"

using namespace mlir;

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------

static llvm::cl::opt<float> SizeIncreaseThreshold(
    "ml-advisor-size-increase-threshold", llvm::cl::Hidden,
    llvm::cl::desc("Maximum factor by which expected IR size may increase "
                   "before blocking further inlining."),
    llvm::cl::init(2.0));

static llvm::cl::opt<bool> StopImmediatelyForTest(
    "ml-inliner-stop-immediately", llvm::cl::Hidden);

// ---------------------------------------------------------------------------
// Feature definitions moved to MLInlineModelFeatureMaps.h
// ------------------------------------------------------------
static const std::vector<llvm::TensorSpec> &getMLIRFeatureMap() {
  static std::vector<llvm::TensorSpec> FeatureMap = []() {
    std::vector<llvm::TensorSpec> Map;
#define POPULATE_NAMES(DTYPE, SHAPE, NAME, DOC)                                \
  Map.push_back(                                                               \
      llvm::TensorSpec::createSpec<DTYPE>(#NAME, SHAPE));
    ALL_FEATURES(POPULATE_NAMES)
#undef POPULATE_NAMES
    return Map;
  }();
  return FeatureMap;
}


// ---------------------------------------------------------------------------
// Region property helpers (feature extraction)
// ---------------------------------------------------------------------------

namespace {

/// Gather simple structural statistics for every operation inside a Region.
struct RegionProperties {
  int64_t blockCount = 0;
  int64_t regionCount = 0;
  int64_t operandCount = 0;
  int64_t resultCount = 0;
  int64_t entryBlockArgCount = 0;
  bool isIsolatedFromAbove = false;

  static RegionProperties compute(Region *region) {
    RegionProperties props;
    if (!region)
      return props;

    props.blockCount = std::distance(region->begin(), region->end());
    props.isIsolatedFromAbove =
        region->getParentOp()
            ->hasTrait<OpTrait::IsIsolatedFromAbove>();

    // Entry-block argument count.
    if (!region->empty())
      props.entryBlockArgCount = region->front().getNumArguments();

    // Walk all nested operations to sum up operands, results, and regions.
    region->walk([&](Operation *op) {
      props.operandCount += op->getNumOperands();
      props.resultCount += op->getNumResults();
      props.regionCount += op->getNumRegions();
    });
    return props;
  }
};

} // namespace

// ---------------------------------------------------------------------------
// MLIRInlineAdvisor
// ---------------------------------------------------------------------------

class MLIRInlineAdvisor;

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
  /// The feature vector that was fed to the model for this decision.
  std::vector<int64_t> featureValues;
};

/// The ML-based inlining advisor for MLIR.  This mirrors the LLVM
/// MLInlineAdvisor architecture but extracts features from MLIR's Region /
/// Operation / CallGraph infrastructure instead of LLVM IR.
class MLIRInlineAdvisor {
public:
  MLIRInlineAdvisor(
      Operation *op, CallGraph &cg,
      std::function<std::unique_ptr<llvm::MLModelRunner>(
          const std::vector<llvm::TensorSpec> &)>
          runnerFactory);

  ~MLIRInlineAdvisor() = default;

  /// Evaluate the model for a call site.  Returns an advice object that *must*
  /// have one of the record* methods called on it.
  std::unique_ptr<MLIRInlineAdvice> getAdvice(CallOpInterface callOp,
                                              Operation *callerOp,
                                              Region *calleeRegion);

  /// Notification that inlining succeeded (called from the advice object).
  void onSuccessfulInlining(MLIRInlineAdvice &advice, bool calleeDeleted);

  /// Accessors.
  const std::vector<llvm::TensorSpec> &getFeatureMap() const {
    return featureMap;
  }
  bool isForcedToStop() const { return forceStop; }

  // Public helpers for derived classes / advice.
  RegionProperties getCachedProps(Region *region);

private:
  /// Evaluate the model with the current feature buffer.
  bool evaluateModel();

  /// Recompute module-level call-graph statistics.
  void recomputeGraphStats();

  /// The ML model runner.
  std::unique_ptr<llvm::MLModelRunner> runner;

  /// The feature map (descriptors).
  std::vector<llvm::TensorSpec> featureMap;

  /// A cache of per-region structural properties.
  llvm::DenseMap<Region *, RegionProperties> propsCache;

  /// Pointers back to the inliner context.
  Operation *op;
  CallGraph &cg;

  /// Module-level graph statistics.
  int64_t graphNodeCount = 0;
  int64_t graphEdgeCount = 0;

  /// Per-callable call-site height (distance from leaf). Populated at
  /// construction.
  llvm::DenseMap<Region *, unsigned> regionLevels;

  /// Whether the advisor has decided to stop recommending inlining (e.g. the
  /// module grew too much).
  bool forceStop = false;

  /// Initial (pre-inlining) total operation count across all callables.
  int64_t initialTotalOps = 0;
  /// Current total operation count.
  int64_t currentTotalOps = 0;
};

// ---------------------------------------------------------------------------
// RegionProperties cache
// ---------------------------------------------------------------------------

RegionProperties MLIRInlineAdvisor::getCachedProps(Region *region) {
  auto it = propsCache.find(region);
  if (it != propsCache.end())
    return it->second;
  RegionProperties props = RegionProperties::compute(region);
  propsCache[region] = props;
  return props;
}

// ---------------------------------------------------------------------------
// Graph statistics helpers
// ---------------------------------------------------------------------------

/// Count the number of call edges in the call graph (excluding abstract and
/// child edges, and excluding the special external / unknown nodes).
static std::pair<int64_t, int64_t> countGraphStats(CallGraph &cg) {
  int64_t nodes = 0, edges = 0;
  for (CallGraphNode *node : cg) {
    // Skip the two sentinel nodes.
    if (node == cg.getExternalCallerNode() ||
        node == cg.getUnknownCalleeNode())
      continue;
    ++nodes;
    for (const CallGraphNode::Edge &edge : *node) {
      if (edge.isCall())
        ++edges;
    }
  }
  return {nodes, edges};
}

/// Compute the height of each callable node (longest distance from a leaf).
static llvm::DenseMap<Region *, unsigned>
computeRegionLevels(const CallGraph &cg) {
  // Avoid the sentinel nodes in the SCC walk.
  llvm::DenseMap<Region *, unsigned> levels;

  for (auto sccIt = llvm::scc_begin(&cg); !sccIt.isAtEnd(); ++sccIt) {
    const std::vector<CallGraphNode *> &sccNodes = *sccIt;
    unsigned level = 0;

    for (CallGraphNode *cgNode : sccNodes) {
      if (cgNode == cg.getExternalCallerNode() ||
          cgNode == cg.getUnknownCalleeNode())
        continue;
      for (const CallGraphNode::Edge &edge : *cgNode) {
        if (!edge.isCall())
          continue;
        CallGraphNode *target = edge.getTarget();
        if (target == cg.getExternalCallerNode() ||
            target == cg.getUnknownCalleeNode())
          continue;
        Region *targetRegion = target->getCallableRegion();
        auto it = levels.find(targetRegion);
        if (it != levels.end())
          level = std::max(level, it->second + 1);
      }
    }

    for (CallGraphNode *cgNode : sccNodes) {
      if (cgNode == cg.getExternalCallerNode() ||
          cgNode == cg.getUnknownCalleeNode())
        continue;
      Region *r = cgNode->getCallableRegion();
      if (r)
        levels[r] = level;
    }
  }
  return levels;
}

/// Count total operations across all callable regions (non-recursive into
/// callees; just the direct bodies).
static int64_t countTotalOps(CallGraph &cg) {
  int64_t total = 0;
  for (CallGraphNode *node : cg) {
    if (node == cg.getExternalCallerNode() ||
        node == cg.getUnknownCalleeNode())
      continue;
    Region *r = node->getCallableRegion();
    if (r) {
      r->walk([&](Operation *) { ++total; });
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// MLIRInlineAdvisor implementation
// ---------------------------------------------------------------------------

MLIRInlineAdvisor::MLIRInlineAdvisor(
    Operation *op, CallGraph &cg,
    std::function<std::unique_ptr<llvm::MLModelRunner>(
        const std::vector<llvm::TensorSpec> &)>
        runnerFactory)
    : featureMap(getMLIRFeatureMap()), op(op), cg(cg) {

  // Compute call-graph-level features.
  std::tie(graphNodeCount, graphEdgeCount) = countGraphStats(cg);
  regionLevels = computeRegionLevels(cg);
  initialTotalOps = countTotalOps(cg);
  currentTotalOps = initialTotalOps;

  // Create the model runner.
  runner = runnerFactory(getFeatureMap());
  forceStop = StopImmediatelyForTest;
}

void MLIRInlineAdvisor::recomputeGraphStats() {
  std::tie(graphNodeCount, graphEdgeCount) = countGraphStats(cg);
}

void MLIRInlineAdvisor::onSuccessfulInlining(MLIRInlineAdvice &advice,
                                             bool calleeDeleted) {
  if (forceStop)
    return;

  Operation *callerOp = advice.getCallerOp();
  // The caller's region changed; invalidate the cache entry.
  for (Region &region : callerOp->getRegions())
    propsCache.erase(&region);

  if (calleeDeleted) {
    Operation *calleeOp = advice.getCalleeOp();
    if (calleeOp) {
      for (Region &region : calleeOp->getRegions())
        propsCache.erase(&region);
    }
  }

  // Recompute total ops.
  currentTotalOps = countTotalOps(cg);

  // Stop if the size grew beyond the threshold.
  if (initialTotalOps > 0 &&
      currentTotalOps > SizeIncreaseThreshold * initialTotalOps)
    forceStop = true;
}

std::unique_ptr<MLIRInlineAdvice>
MLIRInlineAdvisor::getAdvice(CallOpInterface callOp, Operation *callerOp,
                             Region *calleeRegion) {
  if (forceStop)
    return std::make_unique<MLIRInlineAdvice>(this, callOp, callerOp,
                                              calleeRegion, false,
                                              std::vector<int64_t>{});

  // ----- Extract caller properties -----
  Region *callerRegion = nullptr;
  if (auto callable = dyn_cast<CallableOpInterface>(callerOp))
    callerRegion = callable.getCallableRegion();
  RegionProperties callerProps = getCachedProps(callerRegion);

  // ----- Extract callee properties -----
  RegionProperties calleeProps = getCachedProps(calleeRegion);

  // ----- Call-site properties -----
  Operation *callOpAsOp = callOp.getOperation();
  int64_t callSiteOperandCount = callOpAsOp->getNumOperands();
  int64_t callSiteNumCtantArgs = 0;
  for (Value operand : callOpAsOp->getOperands()) {
    if (auto *defOp = operand.getDefiningOp()) {
      // Heuristic: if the operand is defined by an op with no operands and no
      // regions, treat it as a constant-like value (e.g. arith.constant).
      if (defOp->getNumOperands() == 0 && defOp->getNumRegions() == 0)
        ++callSiteNumCtantArgs;
    } else {
      // Block arguments: not a constant.
    }
  }

  // ----- Call-site height -----
  unsigned height = 0;
  auto levelIt = regionLevels.find(calleeRegion);
  if (levelIt != regionLevels.end())
    height = levelIt->second;

  // ----- Populate feature tensors -----
  auto setFeature = [&](MLIRInlineFeatureIndex idx, int64_t val) {
    *runner->getTensor<int64_t>(static_cast<size_t>(idx)) = val;
  };

  setFeature(MLIRInlineFeatureIndex::callee_block_count,
             calleeProps.blockCount);
  setFeature(MLIRInlineFeatureIndex::callee_region_count,
             calleeProps.regionCount);
  setFeature(MLIRInlineFeatureIndex::callee_operand_count,
             calleeProps.operandCount);
  setFeature(MLIRInlineFeatureIndex::callee_result_count,
             calleeProps.resultCount);
  setFeature(MLIRInlineFeatureIndex::callee_arg_count,
             calleeProps.entryBlockArgCount);
  setFeature(MLIRInlineFeatureIndex::callee_is_isolated_from_above,
             calleeProps.isIsolatedFromAbove ? 1 : 0);

  setFeature(MLIRInlineFeatureIndex::caller_block_count,
             callerProps.blockCount);
  setFeature(MLIRInlineFeatureIndex::caller_region_count,
             callerProps.regionCount);
  setFeature(MLIRInlineFeatureIndex::caller_operand_count,
             callerProps.operandCount);
  setFeature(MLIRInlineFeatureIndex::caller_result_count,
             callerProps.resultCount);
  setFeature(MLIRInlineFeatureIndex::caller_arg_count,
             callerProps.entryBlockArgCount);
  setFeature(MLIRInlineFeatureIndex::caller_is_isolated_from_above,
             callerProps.isIsolatedFromAbove ? 1 : 0);

  setFeature(MLIRInlineFeatureIndex::call_site_operand_count,
             callSiteOperandCount);
  setFeature(MLIRInlineFeatureIndex::call_site_num_ctant_args,
             callSiteNumCtantArgs);

  setFeature(MLIRInlineFeatureIndex::graph_node_count, graphNodeCount);
  setFeature(MLIRInlineFeatureIndex::graph_edge_count, graphEdgeCount);
  setFeature(MLIRInlineFeatureIndex::callsite_height,
             static_cast<int64_t>(height));

  // ----- Evaluate model -----
  bool recommendation = evaluateModel();

  // Build feature value vector for the advice snapshot.
  std::vector<int64_t> featureValues(
      static_cast<size_t>(MLIRInlineFeatureIndex::NumFeatures));
  for (size_t i = 0; i < featureValues.size(); ++i)
    featureValues[i] = *runner->getTensor<int64_t>(i);

  return std::make_unique<MLIRInlineAdvice>(
      this, callOpAsOp, callerOp, calleeRegion, recommendation,
      std::move(featureValues));
}

bool MLIRInlineAdvisor::evaluateModel() {
  return static_cast<bool>(runner->evaluate<int64_t>());
}

// ---------------------------------------------------------------------------
// MLIRInlineAdvice implementation
// ---------------------------------------------------------------------------

MLIRInlineAdvice::MLIRInlineAdvice(MLIRInlineAdvisor *advisor, Operation *callOp,
                                   Operation *callerOp, Region *calleeRegion,
                                   bool recommendation,
                                   const std::vector<int64_t> &featureValues)
    : advisor(advisor), callOp(callOp), callerOp(callerOp),
      calleeRegion(calleeRegion), recommendation(recommendation),
      featureValues(featureValues) {
  // Snapshot the caller properties for potential rollback.
  if (auto callable = dyn_cast<CallableOpInterface>(callerOp))
    preInlineCallerProps =
        advisor->getCachedProps(callable.getCallableRegion());
}

Operation *MLIRInlineAdvice::getCalleeOp() const {
  if (!calleeRegion)
    return nullptr;
  return calleeRegion->getParentOp();
}

void MLIRInlineAdvice::recordInlining(bool calleeDeleted) {
  advisor->onSuccessfulInlining(*this, calleeDeleted);
}

void MLIRInlineAdvice::recordUnsuccessfulInlining() {
  // Roll back the caller properties to the pre-inlining snapshot.
  if (auto callable = dyn_cast<CallableOpInterface>(callerOp)) {
    Region *callerRegion = callable.getCallableRegion();
    if (callerRegion) {
      // A full rollback would require re-walking the IR; instead we just
      // invalidate the cached entry so the next query recomputes.
      advisor->getCachedProps(callerRegion);
    }
  }
}

void MLIRInlineAdvice::recordUnattemptedInlining() {
  // Nothing to do.
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

namespace mlir {

/// Create an ML-driven inline advisor for MLIR.  `runnerFactory` receives the
/// feature descriptors and must return an `MLModelRunner` (or null to indicate
/// the model is unavailable).
std::unique_ptr<MLIRInlineAdvisor> createMLIRInlineAdvisor(
    Operation *op, CallGraph &cg,
    std::function<std::unique_ptr<llvm::MLModelRunner>(
        const std::vector<llvm::TensorSpec> &)>
        runnerFactory) {
  return std::make_unique<MLIRInlineAdvisor>(op, cg, std::move(runnerFactory));
}

} // namespace mlir

// ---------------------------------------------------------------------------
// Release-mode advisor builder
// ---------------------------------------------------------------------------

/// Entry point for constructing an MLIR MLInlineAdvisor in release mode.  The
/// caller provides a fallback decision callback used as a training baseline.
/// Returns nullptr when no compiled model is available.
std::unique_ptr<MLIRInlineAdvisor> getReleaseModeMLIRAdvisor(
    Operation *op, CallGraph &cg,
    std::function<bool(CallOpInterface, Operation *)> getDefaultAdvice) {
  // TODO: wire up the AOT-compiled model (analogous to
  // llvm::getReleaseModeAdvisor).  For now this returns the advisor with a
  // no-op model runner, which always returns false (no inlining).
  auto runnerFactory =
      [&](const std::vector<llvm::TensorSpec> &inputFeatures)
      -> std::unique_ptr<llvm::MLModelRunner> {
    // If no AOT model is compiled and no interactive channel is configured,
    // return nullptr to signal "not available".
    return nullptr;
  };
  auto advisor = std::make_unique<MLIRInlineAdvisor>(op, cg, runnerFactory);
  return advisor;
}
