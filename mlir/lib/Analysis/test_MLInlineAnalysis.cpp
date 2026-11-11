//===- test_MLInlineAnalysis.cpp - machine learned InlineAdvisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface between the inliner and a learned model.
// It delegates model evaluation to either the AOT compiled model (the
// 'release' mode) or a runtime-loaded model (the 'development' case).
//
//===----------------------------------------------------------------------===//
// #include "llvm/Analysis/MLInlineAdvisor.h"
// #include "llvm/Analysis/AnalysisOptions.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InlineModelFeatureMaps.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
// #include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
// #include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TensorSpec.h"
// #include "llvm/Analysis/Utils/MLGOUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Analysis/test_MLInlineAnalysis.h"
#include "mlir/Analysis/AnalysisOptions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Analysis/MLGOUtils.h"
#include "mlir/Analysis/MLModelRunner.h"
#include "mlir/Analysis/ReleaseModeModelRunner.h"



// #include "mlir/IR/MLIRContext.h"
// #include "mlir/Interfaces/CallInterfaces.h"


using namespace mlir;

#if defined(LLVM_HAVE_TF_AOT_INLINERSIZEMODEL)
// codegen-ed file
#include "InlinerSizeModel.h" // NOLINT
using CompiledModelType = llvm::InlinerSizeModel;
#else
using CompiledModelType = NoopSavedModelImpl;
#endif

// #include "llvm/Analysis/EmitCModelRunner.h"
#include "mlir/Analysis/EmitCModelRunner.h"
// #include "mlir/Analysis/InlinerModels.h"
#include "llvm/Analysis/InlinerModels.h"

enum class EmitCModelChoice {
  Default,
#define MLGO_MODEL(CLASS_NAME, CLI_FLAG) CLASS_NAME,
// #include "mlir/Analysis/InlinerModels.def"
#include "llvm/Analysis/InlinerModels.def"
};

static EmitCModelChoice getSelectedMLGOModel() {
  static llvm::cl::opt<EmitCModelChoice> SelectedMLGOModel(
    "mlgo-model", llvm::cl::desc("Select the MLGO model to execute:"),
    llvm::cl::init(EmitCModelChoice::Default),
    llvm::cl::values(clEnumValN(EmitCModelChoice::Default, "default",
                                "Use standard heuristic")
#define MLGO_MODEL(CLASS_NAME, CLI_FLAG)                                       \
  , clEnumValN(EmitCModelChoice::CLASS_NAME, CLI_FLAG,                         \
               "Use the " CLI_FLAG " MLGO model")
// #include "mlir/Analysis/InlinerModels.def"
#include "llvm/Analysis/InlinerModels.def"
                         ));
  return SelectedMLGOModel;
}

// static std::unique_ptr<llvm::MLModelRunner> getCreateEmitCModelRunner() {
  // static std::unique_ptr<llvm::MLModelRunner>
  static std::unique_ptr<MLModelRunner>
  // createEmitCModelRunner(LLVMContext &Ctx,
  createEmitCModelRunner(MLIRContext &Ctx,
                        const std::vector<llvm::TensorSpec> &InputFeatures) {
    switch (getSelectedMLGOModel()) {
    case EmitCModelChoice::Default:
      return nullptr;
  #define MLGO_MODEL(CLASS_NAME, CLI_FLAG)                                       \
    case EmitCModelChoice::CLASS_NAME:                                           \
      return std::make_unique<EmitCModelRunner<llvm::CLASS_NAME>>(Ctx, InputFeatures);
      // return std::make_unique<EmitCModelRunner<CLASS_NAME>>(Ctx, InputFeatures);
      // return std::make_unique<llvm::EmitCModelRunner<CLASS_NAME>>(Ctx, InputFeatures);
  // #include "mlir/Analysis/InlinerModels.def"
  #include "llvm/Analysis/InlinerModels.def"
    }
    llvm_unreachable("Unknown MLGO model type!");
  }
  // return createEmitCModelRunner;
// }

std::unique_ptr<MLInlineAdvisor>
getReleaseModeAdvisor(ModuleOp &moduleOp, AnalysisManager &AM,
                            bool defaultAdvice) {
  const AnalysisOptions &Opts = getAnalysisOptions(moduleOp);
  if (!isReleaseModelValid<CompiledModelType>(
          Opts.inliner_interactive_channel_base, getSelectedMLGOModel()))
    return nullptr;
  auto RunnerFactory = [&](const std::vector<llvm::TensorSpec> &InputFeatures)
      -> std::unique_ptr<MLModelRunner> {
    return createReleaseModeModelRunner<CompiledModelType>(
        *moduleOp.getContext(), InputFeatures, llvm::DecisionName,
        Opts.inliner_interactive_channel_base.str(), llvm::InlineDecisionSpec,
        createEmitCModelRunner,
        EmbeddedModelRunnerOptions().setModelSelector(
            Opts.ml_inliner_model_selector));
  };
  return std::make_unique<MLInlineAdvisor>(moduleOp, AM, RunnerFactory,
                                           defaultAdvice);
}

#define DEBUG_TYPE "inline-ml"

const std::vector<llvm::TensorSpec> &MLInlineAdvisor::getInitialFeatureMap() {
  // clang-format off
static std::vector<llvm::TensorSpec> FeatureMap{
#define POPULATE_NAMES(DTYPE, SHAPE, NAME, __) llvm::TensorSpec::createSpec<DTYPE>(#NAME, SHAPE),
// InlineCost features - these must come first
  INLINE_COST_FEATURE_ITERATOR(POPULATE_NAMES)

// Non-cost features
  INLINE_FEATURE_ITERATOR(POPULATE_NAMES)
#undef POPULATE_NAMES
};
  // clang-format on
  return FeatureMap;
}

// struct MyOperationAnalysis {
  // MyOperationAnalysis(Operation *op);

const char *const DecisionName = "inlining_decision";
const llvm::TensorSpec &getInlineDecisionSpec() {
  static const llvm::TensorSpec decisionNameSpec =
      llvm::TensorSpec::createSpec<int64_t>(DecisionName, {1});
  return decisionNameSpec;
}
const char *const DefaultDecisionName = "inlining_default";
const llvm::TensorSpec &getDefaultDecisionSpec() {
  static const llvm::TensorSpec defaultDecisionNameSpec =
      llvm::TensorSpec::createSpec<int64_t>(DefaultDecisionName, {1});
  return defaultDecisionNameSpec;
}
const char *const RewardName = "delta_size";

CallOpInterface getInlinableCS(CallOpInterface callOp, 
                                      SymbolTableCollection &symbolTable) {
  // 1. Resolve the callee. 
  // resolveCallable returns generic Operation*, so we can't avoid that implicit upcast,
  // but we don't need manual static_casts here.
  // Operation *calleeOp = callOp.resolveCallable(&symbolTable);
  Operation *calleeOp = call_interface_impl::resolveCallable(callOp, &symbolTable);

  // 2. Use dyn_cast_or_null.
  // This handles two checks in one:
  //   a) Is calleeOp valid (not null)?
  //   b) Is calleeOp a CallableOpInterface?
  auto callableCallee = dyn_cast_or_null<CallableOpInterface>(calleeOp);

  // 3. Check if the callee has a region (body).
  // If getCallableRegion() returns nullptr, it's an external declaration.
  if (!callableCallee || !callableCallee.getCallableRegion())
    return nullptr;

  return callOp;
}


// };

MLInlineAnalysis::MLInlineAnalysis(Operation *op, AnalysisManager &AM)
    : moduleOp(cast<ModuleOp>(op)), AM(AM) {
}

// bool getMLInlineAdvisorDecision(Operation *op, AnalysisManager &AM){
std::function<std::unique_ptr<MLInlineAdvisor>(bool)>
MLInlineAnalysis::getReleaseModeAdvisorFunction() {

  // Initialize the MLInlineAdvisor with the provided operation and analysis manager.
  // MLInlineAdvisor(moduleOp, AM);
//   InliningAdvisorMode Mode;
//   auto GetDefaultAdvice = [&AM, Params](CallBase &CB) {
//     auto OIC = getDefaultInlineAdvice(CB, AM, Params);
//     return OIC.has_value();
//   };
//   switch (Mode) {
//   case InliningAdvisorMode::Default:
//     LLVM_DEBUG(dbgs() << "Using default inliner heuristic.\n");
//     Advisor.reset(new DefaultInlineAdvisor(M, FAM, Params, IC));
//     // Restrict replay to default advisor, ML advisors are stateful so
//     // replay will need augmentations to interleave with them correctly.
//     if (!ReplaySettings.ReplayFile.empty()) {
//       Advisor = llvm::getReplayInlineAdvisor(M, FAM, M.getContext(),
//                                              std::move(Advisor), ReplaySettings,
//                                              /* EmitRemarks =*/true, IC);
//     }
//     break;
//     // Run IR2VecVocabAnalysis once per module to get the vocabulary.
//     // We run it here because it is immutable and we want to avoid running it
//     // multiple times.
//   case InliningAdvisorMode::Development:
// #ifdef LLVM_HAVE_TFLITE
//     LLVM_DEBUG(dbgs() << "Using development-mode inliner policy.\n");
//     if (!InlineAdvisorAnalysis::initializeIR2VecVocabIfRequested(M, MAM))
//       return false;
//     Advisor = llvm::getDevelopmentModeAdvisor(M, MAM, GetDefaultAdvice);
// #endif
//     break;
//   case InliningAdvisorMode::Release:
//     LLVM_DEBUG(dbgs() << "Using release-mode inliner policy.\n");
//     if (!InlineAdvisorAnalysis::initializeIR2VecVocabIfRequested(M, MAM))
//       return false;
//     Advisor = llvm::getReleaseModeAdvisor(M, MAM, GetDefaultAdvice);
//     break;
//   }

//   return !!Advisor;
    // return MLInlineAdvisor(moduleOp, AM, defaultAdvice);
  // std::unique_ptr<MLInlineAdvisor>
  // getReleaseModeAdvisor(ModuleOp &moduleOp, AnalysisManager &AM,
                      // std::function<bool(CallOpInterface &)> GetDefaultAdvice);

  auto releaseModeAdvisorFunc = [this](bool defaultAdvice) {
           return getReleaseModeAdvisor(moduleOp, AM, defaultAdvice);
  }; 
  return releaseModeAdvisorFunc;
}

std::function<std::unique_ptr<MLInlineAdvisor>(bool)>
MLInlineAnalysis::getDevelopmentModeAdvisorFunction() {
  auto developmentModeAdvisorFunc = [this](bool defaultAdvice) {
          //  return llvm::getDevelopmentModeAdvisor(moduleOp, AM, defaultAdvice);
                      return getReleaseModeAdvisor(moduleOp, AM, defaultAdvice);

  }; 
  return developmentModeAdvisorFunc;
}

MLInlineAdvisor::MLInlineAdvisor(
    ModuleOp &moduleOp, AnalysisManager &AM,
    std::function<
        std::unique_ptr<MLModelRunner>(const std::vector<llvm::TensorSpec> &)>
        GetModelRunner,
    bool defaultAdvice)
    : moduleOp(moduleOp), AM(AM),
      Opts(getAnalysisOptions(moduleOp)), defaultAdvice(defaultAdvice),
      FeatureMap(getInitialFeatureMap()),
      CG(AM.getAnalysis<CallGraph>()),
      InitialIRSize(getModuleIRSize()), CurrentIRSize(InitialIRSize),
      advisorSymbolTable(AM.getAnalysis<SymbolTableAnalysis>().getSymbolTables()) {

  // Extract the 'call site height' feature - the position of a call site
  // relative to the farthest statically reachable SCC node. We don't mutate
  // this value while inlining happens. Empirically, this feature proved
  // critical in behavioral cloning - i.e. training a model to mimic the manual
  // heuristic's decisions - and, thus, equally important for training for
  // improvement.
  // CallGraph CGraph(M);
  // for (auto I = scc_begin(&CGraph); !I.isAtEnd(); ++I) {
  // SymbolTableCollection advisorSymbolTable;
  const CallGraph CGraph(moduleOp.getOperation());
  // const CallGraph *cgPtr = &CGraph;
  for (auto I = llvm::scc_begin(&CGraph); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &CGNodes = *I;
    unsigned Level = 0;
    for (auto *CGNode : CGNodes) {
      // llvm::Function *F = CGNode->getFunction();

      Region *region = CGNode->getCallableRegion();
      // 2. Get the parent op (the function op).
      auto funcOp = dyn_cast<FunctionOpInterface>(region->getParentOp());
      // auto func = dyn_cast_or_null<FunctionOpInterface>(funcOp);
      // if (!funcOp || funcOp->isDeclaration())
      if (!funcOp || funcOp.isExternal())
        continue;
      // for (auto &I : instructions(funcOp)) {
      // funcOp->walk([&](Operation *op) {
      funcOp.walk([&](CallOpInterface callOp) {

        // if (auto *CS = getInlinableCS(I)) {
        // if (auto CS = MyOperationAnalysis::getInlinableCS(*op)) {
        if (getInlinableCS(callOp, advisorSymbolTable)) {

          // auto *Called = CS->getCalledFunction();
          // auto Called = CS.getCallableForCallee();
          // auto Pos = FunctionLevels.find(&CG.get(*Called));
          // Operation *Callee = callOp.resolveCallable(&symbolTable);
          // auto Callable = dyn_cast<CallableOpInterface>(Callee);
          // CallGraphNode *calleeNode = constCG.lookupNode(Callable.getCallableRegion());
          CallGraphNode *calleeNode = CGraph.resolveCallable(callOp, advisorSymbolTable);

          // auto Pos = FunctionLevels.find(&CG.get(*Called));
          auto Pos = FunctionLevels.find(calleeNode);
          // CallGraphNode *node = CG.lookupNode(&funcOp.getFunctionBody());

          // In bottom up traversal, an inlinable callee is either in the
          // same SCC, or to a function in a visited SCC. So not finding its
          // level means we haven't visited it yet, meaning it's in this SCC.
          if (Pos == FunctionLevels.end())
            return WalkResult::advance();
            // continue;
          Level = std::max(Level, Pos->second + 1);
        }
        return WalkResult::advance();
      });
    }
    for (auto *CGNode : CGNodes) {
      // llvm::Function *F = CGNode->getFunction();
      Region *region = CGNode->getCallableRegion();

      // if (F && !F->isDeclaration())
      if (region && !region->empty()) {
        FunctionLevels[CGNode] = Level;
        // FunctionLevels[&CG.get(*F)] = Level;
      }
    }
  }
  for (auto KVP : FunctionLevels) {
    CallGraphNode *node = KVP.first;
    AllNodes.insert(node);
    Region *region = node->getCallableRegion();
    if (region) {
        auto funcOp = dyn_cast<FunctionOpInterface>(region->getParentOp());
        EdgeCount += getLocalCalls(funcOp);
    }
    // EdgeCount += getLocalCalls(KVP.first->getFunction());
  }
  NodeCount = AllNodes.size();

  // if (auto *IR2VecVocabResult = MAM.getCachedResult<IR2VecVocabAnalysis>(M)) {
  //   if (!IR2VecVocabResult->isValid()) {
  //     M.getContext().emitError("IR2VecVocabAnalysis is not valid");
  //     return;
  //   }
  //   // Add the IR2Vec features to the feature map
  //   auto IR2VecDim = IR2VecVocabResult->getDimension();
  //   FeatureMap.push_back(
  //       TensorSpec::createSpec<float>("callee_embedding", {IR2VecDim}));
  //   FeatureMap.push_back(
  //       TensorSpec::createSpec<float>("caller_embedding", {IR2VecDim}));
  // }
  if (Opts.inliner_interactive_include_default)
    FeatureMap.push_back(llvm::DefaultDecisionSpec);

  ModelRunner = GetModelRunner(getFeatureMap());
  if (!ModelRunner) {
    // M.getContext().emitError("Could not create model runner");
    moduleOp.emitError("Could not create model runner");
    return;
  }
  ModelRunner->switchContext("");
  ForceStop = Opts.ml_inliner_stop_immediately;
}

/*HERE*/
// unsigned MLInlineAdvisor::getInitialFunctionLevel(const Function &F) const {
unsigned MLInlineAdvisor::getInitialFunctionLevel(FunctionOpInterface F) const {
  Region *region = F.getCallableRegion();
  CallGraphNode *node = region ? CG.lookupNode(region) : nullptr;
  return node ? FunctionLevels.lookup(node) : 0;
}

// void MLInlineAdvisor::onPassEntry(LazyCallGraph::SCC *CurSCC) {
//   if (!CurSCC || ForceStop)
//     return;
//   FPICache.clear();
//   // Function passes executed between InlinerPass runs may have changed the
//   // module-wide features.
//   // The cgscc pass manager rules are such that:
//   // - if a pass leads to merging SCCs, then the pipeline is restarted on the
//   // merged SCC
//   // - if a pass leads to splitting the SCC, then we continue with one of the
//   // splits
//   // This means that the NodesInLastSCC is a superset (not strict) of the nodes
//   // that subsequent passes would have processed
//   // - in addition, if new Nodes were created by a pass (e.g. CoroSplit),
//   // they'd be adjacent to Nodes in the last SCC. So we just need to check the
//   // boundary of Nodes in NodesInLastSCC for Nodes we haven't seen. We don't
//   // care about the nature of the Edge (call or ref). `FunctionLevels`-wise, we
//   // record them at the same level as the original node (this is a choice, may
//   // need revisiting).
//   // - nodes are only deleted at the end of a call graph walk where they are
//   // batch deleted, so we shouldn't see any dead nodes here.
//   while (!NodesInLastSCC.empty()) {
//     const auto *N = *NodesInLastSCC.begin();
//     assert(!N->isDead());
//     NodesInLastSCC.erase(N);
//     EdgeCount += getLocalCalls(N->getFunction());
//     const auto NLevel = FunctionLevels.at(N);
//     for (const auto &E : *(*N)) {
//       const auto *AdjNode = &E.getNode();
//       assert(!AdjNode->isDead() && !AdjNode->getFunction().isDeclaration());
//       auto I = AllNodes.insert(AdjNode);
//       // We've discovered a new function.
//       if (I.second) {
//         ++NodeCount;
//         NodesInLastSCC.insert(AdjNode);
//         FunctionLevels[AdjNode] = NLevel;
//       }
//     }
//   }

//   EdgeCount -= EdgesOfLastSeenNodes;
//   EdgesOfLastSeenNodes = 0;

//   // (Re)use NodesInLastSCC to remember the nodes in the SCC right now,
//   // in case the SCC is split before onPassExit and some nodes are split out
//   assert(NodesInLastSCC.empty());
//   for (const auto &N : *CurSCC)
//     NodesInLastSCC.insert(&N);
// }

// void MLInlineAdvisor::onPassExit(LazyCallGraph::SCC *CurSCC) {
//   // No need to keep this around - function passes will invalidate it.
//   if (!Opts.ml_advisor_keep_fpi_cache)
//     FPICache.clear();
//   if (!CurSCC || ForceStop)
//     return;
//   // Keep track of the nodes and edges we last saw. Then, in onPassEntry,
//   // we update the node count and edge count from the subset of these nodes that
//   // survived.
//   EdgesOfLastSeenNodes = 0;

//   // Check on nodes that were in SCC onPassEntry
//   for (const LazyCallGraph::Node *N : NodesInLastSCC) {
//     assert(!N->isDead());
//     EdgesOfLastSeenNodes += getLocalCalls(N->getFunction());
//   }

//   // Check on nodes that may have got added to SCC
//   for (const auto &N : *CurSCC) {
//     assert(!N.isDead());
//     auto I = NodesInLastSCC.insert(&N);
//     if (I.second)
//       EdgesOfLastSeenNodes += getLocalCalls(N.getFunction());
//   }
//   assert(NodeCount >= NodesInLastSCC.size());
//   assert(EdgeCount >= EdgesOfLastSeenNodes);
// }

int64_t MLInlineAdvisor::getLocalCalls(FunctionOpInterface &F) {
  return getCachedFPI(F).DirectCallsToDefinedFunctions;
}

// // Update the internal state of the advisor, and force invalidate feature
// // analysis. Currently, we maintain minimal (and very simple) global state - the
// // number of functions and the number of static calls. We also keep track of the
// // total IR size in this module, to stop misbehaving policies at a certain bloat
// // factor (-ml-advisor-size-increase-threshold)
// void MLInlineAdvisor::onSuccessfulInlining(const MLInlineAdvice &Advice,
//                                            bool CalleeWasDeleted) {
//   assert(!ForceStop);
//   Function *Caller = Advice.getCaller();
//   Function *Callee = Advice.getCallee();
//   // The caller features aren't valid anymore.
//   {
//     PreservedAnalyses PA = PreservedAnalyses::all();
//     PA.abandon<FunctionPropertiesAnalysis>();
//     PA.abandon<LoopAnalysis>();
//     FAM.invalidate(*Caller, PA);
//   }
//   Advice.updateCachedCallerFPI(FAM);
//   if (Caller == Callee) {
//     assert(!CalleeWasDeleted);
//     // We double-counted CallerAndCalleeEdges - since the caller and callee
//     // would be the same
//     assert(Advice.CallerAndCalleeEdges % 2 == 0);
//     CurrentIRSize += getIRSize(*Caller) - Advice.CallerIRSize;
//     EdgeCount += getCachedFPI(*Caller).DirectCallsToDefinedFunctions -
//                  Advice.CallerAndCalleeEdges / 2;
//     // The NodeCount would stay the same.
//   } else {
//     int64_t IRSizeAfter =
//         getIRSize(*Caller) + (CalleeWasDeleted ? 0 : Advice.CalleeIRSize);
//     CurrentIRSize += IRSizeAfter - (Advice.CallerIRSize + Advice.CalleeIRSize);

//     // We can delta-update module-wide features. We know the inlining only
//     // changed the caller, and maybe the callee (by deleting the latter). Nodes
//     // are simple to update. For edges, we 'forget' the edges that the caller
//     // and callee used to have before inlining, and add back what they currently
//     // have together.
//     int64_t NewCallerAndCalleeEdges =
//         getCachedFPI(*Caller).DirectCallsToDefinedFunctions;

//     // A dead function's node is not actually removed from the call graph until
//     // the end of the call graph walk, but the node no longer belongs to any
//     // valid SCC.
//     if (CalleeWasDeleted) {
//       --NodeCount;
//       NodesInLastSCC.erase(CG.lookup(*Callee));
//       DeadFunctions.insert(Callee);
//     } else {
//       NewCallerAndCalleeEdges +=
//           getCachedFPI(*Callee).DirectCallsToDefinedFunctions;
//     }
//     EdgeCount += (NewCallerAndCalleeEdges - Advice.CallerAndCalleeEdges);
//   }
//   if (CurrentIRSize > Opts.ml_advisor_size_increase_threshold * InitialIRSize)
//     ForceStop = true;

//   assert(CurrentIRSize >= 0 && EdgeCount >= 0 && NodeCount >= 0);
// }

int64_t MLInlineAdvisor::getModuleIRSize() const {
  int64_t Ret = 0;
  // for (auto &F : moduleOp)
    // for (func::FuncOp F : moduleOp.getOps<func::FuncOp>()) {
    // if (!F.isDeclaration())
      // Ret += getIRSize(F);
  // }
  for (Operation &op : moduleOp.getBody()->getOperations()) {
    if (auto F = dyn_cast<FunctionOpInterface>(&op))
      if (!F.isExternal())
        Ret += getIRSize(F);
  }
    // if (!F.isDeclaration())
      // Ret += getIRSize(F);
  return Ret;
}

//  for (Operation &op : moduleOp.getBody()->getOperations()) {
//     auto F = dyn_cast<FunctionOpInterface>(&op);

//     if (!F || F.isExternal())
//       continue;

//     Ret += getIRSize(F);
//   }

// int64_t MLInlineAdvisor::getModuleIRSize() const {
//   int64_t Ret = 0;

//   for (func::FuncOp F : moduleOp.getOps<func::FuncOp>()) {
//     if (!F.isDeclaration())
//       Ret += getIRSize(F);
//   }

//   return Ret;
// }
llvm::FunctionPropertiesInfo &MLInlineAdvisor::getCachedFPI(FunctionOpInterface F) const {
  auto InsertPair = FPICache.try_emplace(F.getOperation());
  if (!InsertPair.second)
    return InsertPair.first->second;
   
   auto &FPI = InsertPair.first->second;

   F.getOperation()->walk([&](Operation *op) {
    if (op != F.getOperation())
      ++FPI.TotalInstructionCount;
  });

  return FPI;
}

std::unique_ptr<MLInlineAdvice> MLInlineAdvisor::getAdviceImpl(//CallOpInterface CB) {
    const Inliner::ResolvedCall &resolvedCall) {

  CallOpInterface CB = resolvedCall.call;
  if (auto Skip = getSkipAdviceIfUnreachableCallsite(CB))
    return Skip;


  FunctionOpInterface Caller = CB->getParentOfType<FunctionOpInterface>();
  Operation *calleeOp = call_interface_impl::resolveCallable(CB, &advisorSymbolTable);
  // FunctionOpInterface Callee = dyn_cast_or_null<FunctionOpInterface>(calleeOp);
  FunctionOpInterface Callee = dyn_cast_or_null<FunctionOpInterface>(calleeOp);

  // if (Opts.ml_inliner_skip_policy == SkipMLPolicyCriteria::IfCallerIsNotCold) {
  //   if (!PSI.isFunctionEntryCold(&Caller)) {
  //     // Return a MLInlineAdvice, despite delegating to the default advice,
  //     // because we need to keep track of the internal state. This is different
  //     // from the other instances where we return a "default" InlineAdvice,
  //     // which happen at points we won't come back to the MLAdvisor for
  //     // decisions requiring that state.
  //     // return ForceStop ? std::make_unique<llvm::InlineAdvice>(this, CB, ORE,
  //     //                                                   GetDefaultAdvice(CB))
  //     //                  : std::make_unique<MLInlineAdvice>(this, CB, ORE,
  //     //                                                     GetDefaultAdvice(CB));

  //   }
  // }
  // auto MandatoryKind = InlineAdvisor::getMandatoryKind(CB, FAM, ORE);
  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same thing if this is a recursive case.
  // if (MandatoryKind == InlineAdvisor::MandatoryInliningKind::Never ||
      // &Caller == &Callee)
  if (Caller == Callee)
    return getMandatoryAdvice(CB, false);

  // bool Mandatory =
      // MandatoryKind == llvm::InlineAdvisor::MandatoryInliningKind::Always;

  // If we need to stop, we won't want to track anymore any state changes, so
  // we just return the base InlineAdvice, which acts as a noop.
  if (ForceStop) {
    // ORE.emit([&] {
    //   return OptimizationRemarkMissed(DEBUG_TYPE, "ForceStop", &CB)
    //          << "Won't attempt inlining because module size grew too much.";
    // });
    // return std::make_unique<llvm::InlineAdvice>(this, CB, ORE, Mandatory);
    return std::make_unique<MLInlineAdvice>(this, CB, false);
  }

  int CostEstimate = 0;
  // if (!Mandatory) {
  //   auto IsCallSiteInlinable =
  //       llvm::getInliningCostEstimate(CB, TIR, GetAssumptionCache);
  //   if (!IsCallSiteInlinable) {
  //     // We can't inline this for correctness reasons, so return the base
  //     // InlineAdvice, as we don't care about tracking any state changes (which
  //     // won't happen).
  //     return std::make_unique<llvm::InlineAdvice>(this, CB, ORE, false);
  //   }
  //   CostEstimate = *IsCallSiteInlinable;
  // }

  // const auto CostFeatures =
  //     llvm::getInliningCostFeatures(CB, TIR, GetAssumptionCache);
  // if (!CostFeatures) {
  //   return std::make_unique<llvm::InlineAdvice>(this, CB, ORE, false);
  // }

  // if (Mandatory)
  //   return getMandatoryAdvice(CB, true);

  auto NumCtantParams = 0;
  // for (auto I = CB.arg_begin(), E = CB.arg_end(); I != E; ++I) {
  //   NumCtantParams += (isa<Constant>(*I));
  // }
  for (Value arg : CB.getArgOperands()) {
    if (matchPattern(arg, m_Constant())) {
      NumCtantParams++;
    }
  }


  auto &CallerBefore = getCachedFPI(Caller);
  auto &CalleeBefore = getCachedFPI(Callee);

  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::callee_basic_block_count) =
      CalleeBefore.BasicBlockCount;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::callsite_height) =
      getInitialFunctionLevel(Caller);
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::node_count) = NodeCount;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::nr_ctant_params) =
      NumCtantParams;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::edge_count) = EdgeCount;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::caller_users) =
      CallerBefore.Uses;
  *ModelRunner->getTensor<int64_t>(
      llvm::FeatureIndex::caller_conditionally_executed_blocks) =
      CallerBefore.BlocksReachedFromConditionalInstruction;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::caller_basic_block_count) =
      CallerBefore.BasicBlockCount;
  *ModelRunner->getTensor<int64_t>(
      llvm::FeatureIndex::callee_conditionally_executed_blocks) =
      CalleeBefore.BlocksReachedFromConditionalInstruction;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::callee_users) =
      CalleeBefore.Uses;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::cost_estimate) = CostEstimate;
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::is_callee_avail_external) =
      false;
      // Callee.hasAvailableExternallyLinkage();
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::is_caller_avail_external) =
      false;
      // Caller.hasAvailableExternallyLinkage();

  // if (UseIR2Vec) {
  //   // Python side expects float embeddings. The IR2Vec embeddings are doubles
  //   // as of now due to the restriction of fromJSON method used by the
  //   // readVocabulary method in ir2vec::Embeddings.
  //   auto setEmbedding = [&](const ir2vec::Embedding &Embedding,
  //                           FeatureIndex Index) {
  //     llvm::transform(Embedding, ModelRunner->getTensor<float>(Index),
  //                     [](double Val) { return static_cast<float>(Val); });
  //   };

  //   setEmbedding(CalleeBefore.getFunctionEmbedding(),
  //                FeatureIndex::callee_embedding);
  //   setEmbedding(CallerBefore.getFunctionEmbedding(),
  //                FeatureIndex::caller_embedding);
  // }

  // Add the cost features
  // for (size_t I = 0;
  //      I < static_cast<size_t>(llvm::InlineCostFeatureIndex::NumberOfFeatures); ++I) {
  //   *ModelRunner->getTensor<int64_t>(inlineCostFeatureToMlFeature(
  //       static_cast<llvm::InlineCostFeatureIndex>(I))) = CostFeatures->at(I);
  // }
  // This one would have been set up to be right at the end.
  if (!Opts.inliner_interactive_channel_base.empty() &&
      Opts.inliner_interactive_include_default)
    *ModelRunner->getTensor<int64_t>(getFeatureMap().size() - 1) =
        defaultAdvice;
  return getAdviceFromModel(CB);
}

std::unique_ptr<MLInlineAdvice> MLInlineAdvisor::getAdvice(const Inliner::ResolvedCall &resolvedCall,
                                                       bool MandatoryOnly) {
  // if (!MandatoryOnly)
    return getAdviceImpl(resolvedCall);
  // bool Advice = CB.getCaller() != CB.getCalledFunction() &&
                // MandatoryInliningKind::Always ==
                    // getMandatoryKind(CB, FAM, getCallerORE(CB));
  // return getMandatoryAdvice(CB, Advice);
}

/*HERE*/
std::unique_ptr<MLInlineAdvice>
// MLInlineAdvisor::getAdviceFromModel(CallBase &CB,
MLInlineAdvisor::getAdviceFromModel(CallOpInterface CB) {
                                    // llvm::OptimizationRemarkEmitter &ORE) {
  return std::make_unique<MLInlineAdvice>(
      this, CB, static_cast<bool>(ModelRunner->evaluate<int64_t>()));
}

// /*HERE*/
std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getSkipAdviceIfUnreachableCallsite(CallOpInterface CB) {
  auto caller = CB->getParentOfType<FunctionOpInterface>();
  DominanceInfo domInfo(caller);

  if (!domInfo.isReachableFromEntry(CB->getBlock())) {
    // return std::make_unique<MLInlineAdvice>(this, CB, getCallerORE(CB), false);
    return std::make_unique<MLInlineAdvice>(this, CB, false);
  }
  return nullptr;
}

/*HERE*/
// std::unique_ptr<InlineAdvice> MLInlineAdvisor::getMandatoryAdvice(CallBase &CB,
std::unique_ptr<MLInlineAdvice> MLInlineAdvisor::getMandatoryAdvice(CallOpInterface CB,
                                                                  bool Advice) {
  // Make sure we track inlinings in all cases - mandatory or not.
  if (auto Skip = getSkipAdviceIfUnreachableCallsite(CB))
    return Skip;
  if (Advice && !ForceStop)
    return getMandatoryAdviceImpl(CB);

  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same if we are forced to stop - we don't track anymore.
  // return std::make_unique<llvm::InlineAdvice>(this, CB, getCallerORE(CB), Advice);
  return std::make_unique<MLInlineAdvice>(this, CB, Advice);
}

/*HERE*/
// MLInlineAdvisor::getMandatoryAdviceImpl(CallBase &CB) {
std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getMandatoryAdviceImpl(CallOpInterface &CB) {
  return std::make_unique<MLInlineAdvice>(this, CB, true);
  // return std::make_unique<MLInlineAdvice>(this, CB, getCallerORE(CB), true);
}

void MLInlineAdvisor::print(raw_ostream &OS) const {
  OS << "[MLInlineAdvisor] Nodes: " << NodeCount << " Edges: " << EdgeCount
     << " EdgesOfLastSeenNodes: " << EdgesOfLastSeenNodes << "\n";
  OS << "[MLInlineAdvisor] FPI:\n";
  for (auto I : FPICache) {
    OS << I.first->getName() << ":\n";
    I.second.print(OS);
    OS << "\n";
  }
  OS << "\n";
  OS << "[MLInlineAdvisor] FuncLevels:\n";
  for (auto I : FunctionLevels) {
    CallGraphNode *node = I.first;

    Region *region = node->getCallableRegion();
    Operation *op = region ? region->getParentOp() : nullptr;
    auto funcOp = dyn_cast_or_null<FunctionOpInterface>(op);
    bool isDead = funcOp && DeadFunctions.contains(funcOp);
    std::string name = "<external>";

    if (funcOp) {
      // Use SymbolOpInterface to get the printable name (e.g., "@main")
      if (auto symbol = dyn_cast<SymbolOpInterface>(op))
        name = symbol.getName().str();
    }

    OS << (isDead
               ? "<deleted>"
               : name)
       << " : " << I.second << "\n";
    }

  OS << "\n";
}

/*HERE*/
// MLInlineAdvice::MLInlineAdvice(MLInlineAdvisor *Advisor, CallBase &CB,
MLInlineAdvice::MLInlineAdvice(MLInlineAdvisor *Advisor, CallOpInterface CB,
                               bool Recommendation)
    : CallerIRSize(Advisor->isForcedToStop() ? 0 : Advisor->getIRSize(Caller)),
      CalleeIRSize(Advisor->isForcedToStop() ? 0 : Advisor->getIRSize(Callee)),
      CallerAndCalleeEdges(Advisor->isForcedToStop()
                               ? 0
                               : (Advisor->getLocalCalls(Caller) +
                                  Advisor->getLocalCalls(Callee))),
      IsInliningRecommended(Recommendation),
      PreInlineCallerFPI(Advisor->getCachedFPI(Caller)){
  // if (Recommendation)
    // FPU.emplace(Advisor->getCachedFPI(getCaller()), CB);
}

// /*HERE*/
// void MLInlineAdvice::reportContextForRemark(
//     DiagnosticInfoOptimizationBase &OR) {
//   using namespace llvm::ore;
//   OR << NV("Callee", Callee->getName());
//   for (size_t I = 0; I < getAdvisor()->getFeatureMap().size(); ++I)
//     OR << NV(getAdvisor()->getFeatureMap()[I].name(),
//              *getAdvisor()->getModelRunner().getTensor<int64_t>(I));
//   OR << NV("ShouldInline", isInliningRecommended());
// }

// /*HERE*/
// void MLInlineAdvice::updateCachedCallerFPI(FunctionAnalysisManager &FAM) const {
//   FPU->finish(FAM);
// }

// void MLInlineAdvice::recordInliningImpl() {
//   ORE.emit([&]() {
//     OptimizationRemark R(DEBUG_TYPE, "InliningSuccess", DLoc, Block);
//     reportContextForRemark(R);
//     return R;
//   });
//   getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ false);
// }

// void MLInlineAdvice::recordInliningWithCalleeDeletedImpl() {
//   ORE.emit([&]() {
//     OptimizationRemark R(DEBUG_TYPE, "InliningSuccessWithCalleeDeleted", DLoc,
//                          Block);
//     reportContextForRemark(R);
//     return R;
//   });
//   getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ true);
// }

// void MLInlineAdvice::recordUnsuccessfulInliningImpl(
//     const llvm::InlineResult &Result) {
//   getAdvisor()->getCachedFPI(*Caller) = PreInlineCallerFPI;
//   ORE.emit([&]() {
//     OptimizationRemarkMissed R(DEBUG_TYPE, "InliningAttemptedAndUnsuccessful",
//                                DLoc, Block);
//     reportContextForRemark(R);
//     return R;
//   });
// }
// void MLInlineAdvice::recordUnattemptedInliningImpl() {
//   assert(!FPU);
//   ORE.emit([&]() {
//     OptimizationRemarkMissed R(DEBUG_TYPE, "IniningNotAttempted", DLoc, Block);
//     reportContextForRemark(R);
//     return R;
//   });
// }
