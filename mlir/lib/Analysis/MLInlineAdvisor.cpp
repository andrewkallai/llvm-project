//===- MLInlineAdvisor.cpp - machine learned MLIR InlineAdvisor -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "mlir/Analysis/MLInlineAdvisor.h"
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
#include "mlir/Analysis/MLInlineModelFeatureMaps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/Support/CommandLine.h"
#include <string_view>

using namespace mlir;

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------

// Provide accessor functions to avoid global constructors/destructors.
static float getSizeIncreaseThreshold() {
  static llvm::cl::opt<float> SizeIncreaseThreshold(
      "mlir-ml-advisor-size-increase-threshold", llvm::cl::Hidden,
      llvm::cl::desc("Maximum factor by which expected IR size may increase "
                     "before blocking further inlining."),
      llvm::cl::init(2.0));
  return SizeIncreaseThreshold;
}

static bool getStopImmediatelyForTest() {
  static llvm::cl::opt<bool>
      StopImmediatelyForTest("mlir-ml-inliner-stop-immediately",
                             llvm::cl::Hidden);
  return StopImmediatelyForTest;
}
// ---------------------------------------------------------------------------
// Feature definitions moved to MLInlineModelFeatureMaps.h
// ------------------------------------------------------------
const std::vector<llvm::TensorSpec> &MLIRInlineAdvisor::getMLIRFeatureMap() {
  static std::vector<llvm::TensorSpec> FeatureMap = []() {
    std::vector<llvm::TensorSpec> Map;
#define POPULATE_NAMES(DTYPE, SHAPE, NAME, DOC)                                \
  Map.push_back(llvm::TensorSpec::createSpec<DTYPE>("action_" #NAME, SHAPE));
    ALL_FEATURES(POPULATE_NAMES)
#undef POPULATE_NAMES
    return Map;
  }();
  return FeatureMap;
}

// Return feature map with RL extras (discount, step_type, reward) for model loading.
const std::vector<llvm::TensorSpec> &MLIRInlineAdvisor::getMLIRInputFeatureMap() {
  static std::vector<llvm::TensorSpec> InputFeatureMap = []() {
    std::vector<llvm::TensorSpec> Map;
#define POPULATE_NAMES(DTYPE, SHAPE, NAME, DOC)                                \
  Map.push_back(llvm::TensorSpec::createSpec<DTYPE>("action_" #NAME, SHAPE));
    ALL_OBSERVATION_FEATURES(POPULATE_NAMES)
#undef POPULATE_NAMES
    // RL time step tensors required by TF-Agents TFLite models.
    Map.push_back(llvm::TensorSpec::createSpec<float>("action_discount", {1}));
    Map.push_back(llvm::TensorSpec::createSpec<int32_t>("action_step_type", {1}));
    Map.push_back(llvm::TensorSpec::createSpec<float>("action_reward", {1}));
    return Map;
  }();
  return InputFeatureMap;
}


// ---------------------------------------------------------------------------
// Region property helpers (feature extraction)
// ---------------------------------------------------------------------------

/// Gather simple structural statistics for every operation inside a Region.
// RegionProperties is defined in MLInlineAdvisor.h

RegionProperties RegionProperties::compute(Region *region) {
  RegionProperties props;
  if (!region)
    return props;

  props.blockCount = std::distance(region->begin(), region->end());
  props.isIsolatedFromAbove =
      region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>();

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

// ---------------------------------------------------------------------------
// Graph statistics helpers
// ---------------------------------------------------------------------------

/// Count the number of call edges in the call graph (excluding abstract and
/// child edges, and excluding the special external / unknown nodes).
static std::pair<int64_t, int64_t> countGraphStats(CallGraph &cg) {
  int64_t nodes = 0, edges = 0;
  for (CallGraphNode *node : cg) {
    // Skip the two sentinel nodes.
    if (node == cg.getExternalCallerNode() || node == cg.getUnknownCalleeNode())
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
    if (node == cg.getExternalCallerNode() || node == cg.getUnknownCalleeNode())
      continue;
    Region *r = node->getCallableRegion();
    if (r) {
      r->walk([&](Operation *) { ++total; });
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// MLIRInlineAdvisor
// ---------------------------------------------------------------------------

// Members below are in the .cpp because they use internal types.
// (e.g., propsCache uses RegionProperties which is defined in this file.)

// MLIRInlineAdvisor implementation
// ---------------------------------------------------------------------------

MLIRInlineAdvisor::MLIRInlineAdvisor(
    Operation *op, CallGraph &cg,
    std::function<std::unique_ptr<llvm::MLModelRunner>(
        const std::vector<llvm::TensorSpec> &)>
        runnerFactory,
    llvm::Logger *logger)
    : featureMap(getMLIRFeatureMap()), logger(logger), cg(cg) {

  // Compute call-graph-level features.
  std::tie(graphNodeCount, graphEdgeCount) = countGraphStats(cg);
  regionLevels = computeRegionLevels(cg);
  initialTotalOps = countTotalOps(cg);
  currentTotalOps = initialTotalOps;

  // Create the model runner.
  runner = runnerFactory(getMLIRInputFeatureMap());
  forceStop = getStopImmediatelyForTest();
}

RegionProperties MLIRInlineAdvisor::getCachedProps(Region *region) {
  auto it = propsCache.find(region);
  if (it != propsCache.end())
    return it->second;
  RegionProperties props = RegionProperties::compute(region);
  propsCache[region] = props;
  return props;
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
      currentTotalOps > getSizeIncreaseThreshold() * initialTotalOps)
    forceStop = true;
}

std::unique_ptr<MLIRInlineAdvice>
MLIRInlineAdvisor::getAdvice(CallOpInterface callOp, Operation *callerOp,
                             Region *calleeRegion) {
  if (forceStop)
    return std::make_unique<MLIRInlineAdvice>(
        this, callOp, callerOp, calleeRegion, false, std::vector<int64_t>{});

  // If no model runner is available, compute features inline and log them
  // for training data collection (default trace mode), then always inline.
  if (!runner) {
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
        if (defOp->getNumOperands() == 0 && defOp->getNumRegions() == 0)
          ++callSiteNumCtantArgs;
      }
    }

    // ----- Call-site height -----
    unsigned height = 0;
    auto levelIt = regionLevels.find(calleeRegion);
    if (levelIt != regionLevels.end())
      height = levelIt->second;

    // ----- Build feature values without a model runner -----
    std::vector<int64_t> featureValues(
        getFeatureMap().size());

    auto idx = [&](MLIRInlineFeatureIndex e) -> size_t {
      return static_cast<size_t>(e);
    };

    featureValues[idx(MLIRInlineFeatureIndex::callee_block_count)] =
        calleeProps.blockCount;
    featureValues[idx(MLIRInlineFeatureIndex::callee_region_count)] =
        calleeProps.regionCount;
    featureValues[idx(MLIRInlineFeatureIndex::callee_operand_count)] =
        calleeProps.operandCount;
    featureValues[idx(MLIRInlineFeatureIndex::callee_result_count)] =
        calleeProps.resultCount;
    featureValues[idx(MLIRInlineFeatureIndex::callee_arg_count)] =
        calleeProps.entryBlockArgCount;
    featureValues[idx(MLIRInlineFeatureIndex::callee_is_isolated_from_above)] =
        calleeProps.isIsolatedFromAbove ? 1 : 0;
    featureValues[idx(MLIRInlineFeatureIndex::caller_block_count)] =
        callerProps.blockCount;
    featureValues[idx(MLIRInlineFeatureIndex::caller_region_count)] =
        callerProps.regionCount;
    featureValues[idx(MLIRInlineFeatureIndex::caller_operand_count)] =
        callerProps.operandCount;
    featureValues[idx(MLIRInlineFeatureIndex::caller_result_count)] =
        callerProps.resultCount;
    featureValues[idx(MLIRInlineFeatureIndex::caller_arg_count)] =
        callerProps.entryBlockArgCount;
    featureValues[idx(MLIRInlineFeatureIndex::caller_is_isolated_from_above)] =
        callerProps.isIsolatedFromAbove ? 1 : 0;
    featureValues[idx(MLIRInlineFeatureIndex::call_site_operand_count)] =
        callSiteOperandCount;
    featureValues[idx(MLIRInlineFeatureIndex::call_site_num_ctant_args)] =
        callSiteNumCtantArgs;
    featureValues[idx(MLIRInlineFeatureIndex::graph_node_count)] =
        graphNodeCount;
    featureValues[idx(MLIRInlineFeatureIndex::graph_edge_count)] =
        graphEdgeCount;
    featureValues[idx(MLIRInlineFeatureIndex::callsite_height)] =
        static_cast<int64_t>(height);
    featureValues[idx(MLIRInlineFeatureIndex::graph_callee_region_level)] =
        static_cast<int64_t>(height);
    featureValues[idx(MLIRInlineFeatureIndex::graph_initial_total_ops)] =
        initialTotalOps;
    featureValues[idx(MLIRInlineFeatureIndex::graph_current_total_ops_ratio)] =
        initialTotalOps > 0 ? (currentTotalOps * 100 / initialTotalOps) : 0;
    featureValues[idx(MLIRInlineFeatureIndex::inlining_decision)] = 1;
    // Use a simple heuristic to produce varied training data. In CIR, callee
    // block counts are nearly always <= 2, so the operand count (which varies
    // widely) is used instead: inline small callees, decline larger ones.
    bool heuristicInline = calleeProps.operandCount <= 10;
    featureValues[idx(MLIRInlineFeatureIndex::inlining_decision)] = heuristicInline ? 1 : 0;

    // Log the observation with computed features.
    if (logger) {
      logger->startObservation();
      for (size_t i = 0; i < featureValues.size(); ++i)
        logger->logTensorValue(
            i, reinterpret_cast<const char *>(&featureValues[i]));
      logger->endObservation();
      // Log heuristic decision as reward so training log has outcome.
      int64_t heuristicReward = heuristicInline ? 100 : 0;
      logger->logReward(heuristicReward);
      logger->flush();
    }

    return std::make_unique<MLIRInlineAdvice>(
        this, callOp, callerOp, calleeRegion, heuristicInline, std::move(featureValues));
  }

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
  setFeature(MLIRInlineFeatureIndex::graph_callee_region_level,
             static_cast<int64_t>(height));
  setFeature(MLIRInlineFeatureIndex::graph_initial_total_ops,
             initialTotalOps);
  setFeature(MLIRInlineFeatureIndex::graph_current_total_ops_ratio,
             initialTotalOps > 0 ? (currentTotalOps * 100 / initialTotalOps) : 0);
  setFeature(MLIRInlineFeatureIndex::callsite_height,
             static_cast<int64_t>(height));

  // ----- Set RL time-step extras required by TF-Agents policy models -----
  // The observation features occupy indices [0, NumObsFeatures). The model also
  // expects discount / step_type / reward; initialize them to a first,
  // non-terminal decision step so evaluation is deterministic.
  const size_t NumObsFeatures =
      static_cast<size_t>(MLIRInlineFeatureIndex::inlining_decision);
  *runner->getTensor<float>(NumObsFeatures) = 1.0f;     // action_discount
  *runner->getTensor<int32_t>(NumObsFeatures + 1) = 0;  // action_step_type (FIRST)
  *runner->getTensor<float>(NumObsFeatures + 2) = 0.0f; // action_reward

  // ----- Evaluate model -----
  bool recommendation = evaluateModel();

  // Build feature value vector for the advice snapshot.
  // The runner has 20 observation features (indices 0-19) plus RL extras.
  // inlining_decision (index 20) is the model output, so set it manually.
  std::vector<int64_t> featureValues(
      getFeatureMap().size());
  size_t obsFeatureCount = static_cast<size_t>(MLIRInlineFeatureIndex::inlining_decision);
  for (size_t i = 0; i < obsFeatureCount; ++i)
    featureValues[i] = *runner->getTensor<int64_t>(i);
  featureValues[static_cast<size_t>(MLIRInlineFeatureIndex::inlining_decision)] =
      recommendation ? 1 : 0;

  // Log the observation if training logger is configured.
  if (logger) {
    logger->startObservation();
    for (size_t i = 0; i < featureValues.size(); ++i)
      logger->logTensorValue(
          i, reinterpret_cast<const char *>(&featureValues[i]));
    logger->endObservation();
    // Log reward so training log has outcome.
    int64_t logReward = currentTotalOps;
    logger->logReward(logReward);
    logger->flush();
  }
  return std::make_unique<MLIRInlineAdvice>(this, callOpAsOp, callerOp,
                                            calleeRegion, recommendation,
                                            std::move(featureValues));
}

bool MLIRInlineAdvisor::evaluateModel() {
  if (!runner)
    return true;
  return static_cast<bool>(runner->evaluate<int64_t>());
}

// ---------------------------------------------------------------------------
// MLIRInlineAdvice implementation
// ---------------------------------------------------------------------------

MLIRInlineAdvice::MLIRInlineAdvice(MLIRInlineAdvisor *advisor,
                                   Operation *callOp, Operation *callerOp,
                                   Region *calleeRegion, bool recommendation,
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
std::unique_ptr<MLIRInlineAdvisor>
createMLIRInlineAdvisor(Operation *op, CallGraph &cg,
                        std::function<std::unique_ptr<llvm::MLModelRunner>(
                            const std::vector<llvm::TensorSpec> &)>
        runnerFactory,
    llvm::Logger *logger) {
  return std::make_unique<MLIRInlineAdvisor>(op, cg, std::move(runnerFactory),
                                             logger);
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
  // no-op model runner (nullptr), so the advisor defaults to always inlining.
  auto runnerFactory = [&](const std::vector<llvm::TensorSpec> &inputFeatures)
      -> std::unique_ptr<llvm::MLModelRunner> {
    // If no AOT model is compiled and no interactive channel is configured,
    // return nullptr to signal "not available".
    return nullptr;
  };
  auto advisor = std::make_unique<MLIRInlineAdvisor>(op, cg, runnerFactory);
  return advisor;
}
