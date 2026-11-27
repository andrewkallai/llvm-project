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
// #include "llvm/Analysis/OptimizationRemarkEmitter.h"
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
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ManagedStatic.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Analysis/CFGLoopInfo.h"





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



static EmitCModelChoice SelectedMLGOModel =
    EmitCModelChoice::Default;

void setSelectedMLGOModel(EmitCModelChoice model) {
  SelectedMLGOModel = model;
}
// static EmitCModelChoice getSelectedMLGOModel() {
// struct MLInlineAdvisorOptions {
// llvm::cl::opt<EmitCModelChoice> SelectedMLGOModel{
//     "mlir-mlgo-model", llvm::cl::desc("Select the MLGO model to execute:"),
//     llvm::cl::init(EmitCModelChoice::Default),
//     llvm::cl::values(clEnumValN(EmitCModelChoice::Default, "default",
//                                 "Use standard heuristic")
// #define MLGO_MODEL(CLASS_NAME, CLI_FLAG)                                       \
//   , clEnumValN(EmitCModelChoice::CLASS_NAME, CLI_FLAG,                         \
//                "Use the " CLI_FLAG " MLGO model")
// // #include "mlir/Analysis/InlinerModels.def"
// #include "llvm/Analysis/InlinerModels.def"
//                          )};
// };
  // return SelectedMLGOModel;
// }


// llvm::ManagedStatic<MLInlineAdvisorOptions> AdvisorOptions;

// void mlir::registerMLInlineAdvisorCLOptions() {
  // (void)*AdvisorOptions;
// }

// void registerMLInlineAdvisorCLOptions() {
//   *AdvisorOptions;
// }

// AdvisorOptions->SelectedMLGOModel.getValue()



// static std::unique_ptr<llvm::MLModelRunner> getCreateEmitCModelRunner() {
  // static std::unique_ptr<llvm::MLModelRunner>
  static std::unique_ptr<MLModelRunner>
  // createEmitCModelRunner(LLVMContext &Ctx,
  createEmitCModelRunner(MLIRContext &Ctx,
                        const std::vector<llvm::TensorSpec> &InputFeatures) {
    // switch (getSelectedMLGOModel()) {
    switch (SelectedMLGOModel) {
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
getReleaseModeAdvisor(Operation *moduleOp, AnalysisManager &AM,
                            std::function<bool(const Inliner::ResolvedCall &)> getDefaultAdvice) {
                            // bool defaultAdvice) {
  const AnalysisOptions &Opts = getAnalysisOptions(*moduleOp);
  if (!isReleaseModelValid<CompiledModelType>(
          // Opts.inliner_interactive_channel_base, getSelectedMLGOModel()))
          Opts.inliner_interactive_channel_base, SelectedMLGOModel))
    return nullptr;
  auto RunnerFactory = [&](const std::vector<llvm::TensorSpec> &InputFeatures)
      -> std::unique_ptr<MLModelRunner> {
    return createReleaseModeModelRunner<CompiledModelType>(
        *moduleOp->getContext(), InputFeatures, llvm::DecisionName,
        Opts.inliner_interactive_channel_base.str(), llvm::InlineDecisionSpec,
        createEmitCModelRunner,
        EmbeddedModelRunnerOptions().setModelSelector(
            Opts.ml_inliner_model_selector));
  };
  return std::make_unique<MLInlineAdvisor>(moduleOp, AM, RunnerFactory,
                                           getDefaultAdvice);
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
    : moduleOp(op), AM(AM) {
}

// bool getMLInlineAdvisorDecision(Operation *op, AnalysisManager &AM){
// std::function<std::unique_ptr<MLInlineAdvisor>(bool)>
// std::unique_ptr<MLInlineAdvisor>
void
MLInlineAnalysis::getReleaseModeAdvisorFunction(EmitCModelChoice model, std::function<bool(const Inliner::ResolvedCall &)> getDefaultAdvice, bool printInlineAdvisor) {
  setSelectedMLGOModel(model); 
  // auto releaseModeAdvisorFunc = [this](bool defaultAdvice) {

            auto advisor = getReleaseModeAdvisor(moduleOp, AM, std::move(getDefaultAdvice));
            setMLInlineAdvisor(std::move(advisor));
            setPrintInlineAdvisor(printInlineAdvisor);
            // if (!advisor){
                  // llvm_unreachable("getReleaseModeAdvisor returned nullpr");
            // }
            // return advisor;
  // }; 
  // return releaseModeAdvisorFunc;
  // return getReleaseModeAdvisor(moduleOp, AM, defaultAdvice)
}

// std::function<std::unique_ptr<MLInlineAdvisor>(bool)>
// std::unique_ptr<MLInlineAdvisor>
void
MLInlineAnalysis::getDevelopmentModeAdvisorFunction(EmitCModelChoice model, std::function<bool(const Inliner::ResolvedCall &)> getDefaultAdvice, bool printInlineAdvisor) {
  setSelectedMLGOModel(model); 
  // auto developmentModeAdvisorFunc = [this](bool defaultAdvice) {
          //  return llvm::getDevelopmentModeAdvisor(moduleOp, AM, defaultAdvice);
                      auto advisor = getReleaseModeAdvisor(moduleOp, AM, std::move(getDefaultAdvice));
                      setMLInlineAdvisor(std::move(advisor));
                      setPrintInlineAdvisor(printInlineAdvisor);
                      // return advisor;

  // }; 
  // return developmentModeAdvisorFunc;
}

MLInlineAdvisor::MLInlineAdvisor(
    Operation *moduleOp, AnalysisManager &AM,
    std::function<
        std::unique_ptr<MLModelRunner>(const std::vector<llvm::TensorSpec> &)>
        GetModelRunner,
    std::function<bool(const Inliner::ResolvedCall &)> getDefaultAdvice)
    // bool defaultAdvice)
    : advisorSymbolTable(AM.getAnalysis<SymbolTableAnalysis>().getSymbolTables()),
      moduleOp(moduleOp), AM(AM),
      Opts(getAnalysisOptions(*moduleOp)),
      FeatureMap(getInitialFeatureMap()),
      getDefaultAdvice(getDefaultAdvice),
      CG(AM.getAnalysis<CallGraph>()),
      InitialIRSize(getModuleIRSize()),
      CurrentIRSize(InitialIRSize) {

  // Extract the 'call site height' feature - the position of a call site
  // relative to the farthest statically reachable SCC node. We don't mutate
  // this value while inlining happens. Empirically, this feature proved
  // critical in behavioral cloning - i.e. training a model to mimic the manual
  // heuristic's decisions - and, thus, equally important for training for
  // improvement.
  // CallGraph CGraph(M);
  // for (auto I = scc_begin(&CGraph); !I.isAtEnd(); ++I) {
  // SymbolTableCollection advisorSymbolTable;
  // const CallGraph CGraph(moduleOp);
  CallGraph &CGraph = AM.getAnalysis<CallGraph>();
  // const CallGraph *cgPtr = &CGraph;
  const CallGraph *CG = &CGraph;

  for (auto I = llvm::scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &CGNodes = *I;
    unsigned Level = 0;
    for (auto *CGNode : CGNodes) {
      // llvm::Function *F = CGNode->getFunction();
      // External call graph nodes do not have callable regions.
      if (CGNode->isExternal()){
        // llvm::errs() << "  XExternal node: " << CGNode << "\n";
        continue;
      }
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
      if (CGNode->isExternal())
        continue;
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
    moduleOp->emitError("Could not create model runner");
    return;
  }
  ModelRunner->switchContext("");
  ForceStop = Opts.ml_inliner_stop_immediately;
}

unsigned MLInlineAdvisor::getInitialFunctionLevel(FunctionOpInterface F) const {
  Region *region = F.getCallableRegion();
  CallGraphNode *node = region ? CG.lookupNode(region) : nullptr;
  return node ? FunctionLevels.lookup(node) : 0;
}

void MLInlineAdvisor::onPassEntry(llvm::ArrayRef<CallGraphNode *> curSCC) {
  if (curSCC.empty() || ForceStop)
    return;
  FPICache.clear();
  // Function passes executed between InlinerPass runs may have changed the
  // module-wide features.
  // The cgscc pass manager rules are such that:
  // - if a pass leads to merging SCCs, then the pipeline is restarted on the
  // merged SCC
  // - if a pass leads to splitting the SCC, then we continue with one of the
  // splits
  // This means that the NodesInLastSCC is a superset (not strict) of the nodes
  // that subsequent passes would have processed
  // - in addition, if new Nodes were created by a pass (e.g. CoroSplit),
  // they'd be adjacent to Nodes in the last SCC. So we just need to check the
  // boundary of Nodes in NodesInLastSCC for Nodes we haven't seen. We don't
  // care about the nature of the Edge (call or ref). `FunctionLevels`-wise, we
  // record them at the same level as the original node (this is a choice, may
  // need revisiting).
  // - nodes are only deleted at the end of a call graph walk where they are
  // batch deleted, so we shouldn't see any dead nodes here.
  while (!NodesInLastSCC.empty()) {
    const auto *N = *NodesInLastSCC.begin();
    assert(!N->isExternal() && !DeadCallGraphNodes.contains(N));
    NodesInLastSCC.erase(N);
    // llvm::errs() << "Processing node: " << N
    //         << " external=" << N->isExternal()
    //         << " dead=" << DeadCallGraphNodes.contains(N)
    //         << "\n";
    auto F = dyn_cast<FunctionOpInterface>(
      N->getCallableRegion()->getParentOp());
    assert(F && !F.isExternal());
    EdgeCount += getLocalCalls(F);
    const auto NLevel = FunctionLevels.at(N);
    for (const auto &E : *N) {
      auto *AdjNode = E.getTarget();
      // llvm::errs() << "AdjNode external="
            //  << AdjNode->isExternal() << "\n";
      if (AdjNode->isExternal())
        continue;
      // assert(!AdjNode->isExternal());
      auto AdjFunc = dyn_cast<FunctionOpInterface>(
        AdjNode->getCallableRegion()->getParentOp());
      

      assert(!DeadCallGraphNodes.contains(AdjNode) && (AdjFunc && !AdjFunc.isExternal()));
      auto I = AllNodes.insert(AdjNode);
      // We've discovered a new function.
      if (I.second) {
        ++NodeCount;
        NodesInLastSCC.insert(AdjNode);
        FunctionLevels[AdjNode] = NLevel;
      }
    }
  }

  EdgeCount -= EdgesOfLastSeenNodes;
  EdgesOfLastSeenNodes = 0;

  // (Re)use NodesInLastSCC to remember the nodes in the SCC right now,
  // in case the SCC is split before onPassExit and some nodes are split out
  assert(NodesInLastSCC.empty());
  for (auto *N : curSCC)
    NodesInLastSCC.insert(N);
}

void MLInlineAdvisor::onPassExit(llvm::ArrayRef<CallGraphNode *> curSCC) {
  // No need to keep this around - function passes will invalidate it.
  if (!Opts.ml_advisor_keep_fpi_cache)
    FPICache.clear();
  if (curSCC.empty() || ForceStop)
    return;
  // Keep track of the nodes and edges we last saw. Then, in onPassEntry,
  // we update the node count and edge count from the subset of these nodes that
  // survived.
  EdgesOfLastSeenNodes = 0;

  // Check on nodes that were in SCC onPassEntry
  for (const CallGraphNode *N : NodesInLastSCC) {
    auto F = dyn_cast<FunctionOpInterface>(
      N->getCallableRegion()->getParentOp());
    assert(!DeadCallGraphNodes.contains(N));
    EdgesOfLastSeenNodes += getLocalCalls(F);
  }

  // Check on nodes that may have got added to SCC
  for (auto *N : curSCC) {
    assert(!DeadCallGraphNodes.contains(N));
    auto I = NodesInLastSCC.insert(N);
    if (I.second){
      auto F = dyn_cast<FunctionOpInterface>(
        N->getCallableRegion()->getParentOp());
      assert(F && !F.isExternal());
      EdgesOfLastSeenNodes += getLocalCalls(F);
    }
  }
  assert(NodeCount >= NodesInLastSCC.size());
  assert(EdgeCount >= EdgesOfLastSeenNodes);
}

int64_t MLInlineAdvisor::getLocalCalls(FunctionOpInterface &F) {
  return getCachedFPI(F).DirectCallsToDefinedFunctions;
}

// Update the internal state of the advisor, and force invalidate feature
// analysis. Currently, we maintain minimal (and very simple) global state - the
// number of functions and the number of static calls. We also keep track of the
// total IR size in this module, to stop misbehaving policies at a certain bloat
// factor (-ml-advisor-size-increase-threshold)
void MLInlineAdvisor::onSuccessfulInlining(const MLInlineAdvice &Advice,
                                           bool CalleeWasDeleted) {
  assert(!ForceStop);
  FunctionOpInterface Caller = Advice.getCaller();
  FunctionOpInterface Callee = Advice.getCallee();
  // The caller features aren't valid anymore.
  {
    // AnalysisManager::PreservedAnalyses PA = PreservedAnalyses::all();
    // PA.abandon<FunctionPropertiesAnalysis>();
    // PA.abandon<LoopAnalysis>();
    // MAM.invalidate(*Caller, PA);
    AnalysisManager::PreservedAnalyses PA;
    AM.nest(Caller.getOperation()).invalidate(PA);
  }
  // Advice.updateCachedCallerFPI(AM);
  Advice.updateCachedCallerFPI();
  if (Caller == Callee) {
    assert(!CalleeWasDeleted);
    // We double-counted CallerAndCalleeEdges - since the caller and callee
    // would be the same
    assert(Advice.CallerAndCalleeEdges % 2 == 0);
    CurrentIRSize += getIRSize(Caller) - Advice.CallerIRSize;
    EdgeCount += getCachedFPI(Caller).DirectCallsToDefinedFunctions -
                 Advice.CallerAndCalleeEdges / 2;
    // The NodeCount would stay the same.
  } else {
    int64_t IRSizeAfter =
        getIRSize(Caller) + (CalleeWasDeleted ? 0 : Advice.CalleeIRSize);
    CurrentIRSize += IRSizeAfter - (Advice.CallerIRSize + Advice.CalleeIRSize);

    // We can delta-update module-wide features. We know the inlining only
    // changed the caller, and maybe the callee (by deleting the latter). Nodes
    // are simple to update. For edges, we 'forget' the edges that the caller
    // and callee used to have before inlining, and add back what they currently
    // have together.
    int64_t NewCallerAndCalleeEdges =
        getCachedFPI(Caller).DirectCallsToDefinedFunctions;

    // A dead function's node is not actually removed from the call graph until
    // the end of the call graph walk, but the node no longer belongs to any
    // valid SCC.
    if (CalleeWasDeleted) {
      --NodeCount;
      auto callable =
      dyn_cast<CallableOpInterface>(Callee.getOperation());

      if (callable){
        if (auto *node = CG.lookupNode(callable.getCallableRegion())) {
          NodesInLastSCC.erase(node);
          DeadCallGraphNodes.insert(node);
        }
      }
      DeadFunctions.insert(Callee);
    } else {
      NewCallerAndCalleeEdges +=
          getCachedFPI(Callee).DirectCallsToDefinedFunctions;
    }
    EdgeCount += (NewCallerAndCalleeEdges - Advice.CallerAndCalleeEdges);
  }
  if (CurrentIRSize > Opts.ml_advisor_size_increase_threshold * InitialIRSize)
    ForceStop = true;

  // llvm::errs() << "MLGO successful-inline accounting:\n"
  //            << "CurrentIRSize: " << CurrentIRSize << "\n"
  //            << "InitialIRSize: " << InitialIRSize << "\n"
  //            << "CallerIRSize: " << Advice.CallerIRSize << "\n"
  //            << "CalleeIRSize: " << Advice.CalleeIRSize << "\n"
  //            << "CallerAndCalleeEdges: "
  //            << Advice.CallerAndCalleeEdges << "\n"
  //            << "EdgeCount: " << EdgeCount << "\n"
  //            << "NodeCount: " << NodeCount << "\n"
  //            << "CalleeWasDeleted: " << CalleeWasDeleted << "\n";

  assert(CurrentIRSize >= 0 && EdgeCount >= 0 && NodeCount >= 0);
}

int64_t MLInlineAdvisor::getModuleIRSize() const {
  int64_t Ret = 0;
  // for (auto &F : moduleOp)
    // for (func::FuncOp F : moduleOp.getOps<func::FuncOp>()) {
    // if (!F.isDeclaration())
      // Ret += getIRSize(F);
  // }


  Block &body = moduleOp->getRegion(0).front();

  // for (Operation &op : moduleOp.getBody()->getOperations()) {
  for (Operation &op : body.getOperations()) {
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
    if (auto CB = dyn_cast<CallOpInterface>(op)) {
      Operation *calleeOp =
          call_interface_impl::resolveCallable(CB, &advisorSymbolTable);

      auto Callee =
          dyn_cast_or_null<FunctionOpInterface>(calleeOp);

      if (Callee && !Callee.isExternal())
        ++FPI.DirectCallsToDefinedFunctions;
    }
  });

  Region &region = F.getFunctionBody();

  if (!region.hasOneBlock()) {
    DominanceInfo domInfo(F.getOperation());
    CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

    for (Block &block : region)
      FPI.MaxLoopDepth =
          std::max(FPI.MaxLoopDepth,  static_cast<int64_t>(loopInfo.getLoopDepth(&block)));
  }

  return FPI;
}

std::unique_ptr<MLInlineAdvice> MLInlineAdvisor::getAdviceImpl(//CallOpInterface CB) {
    const Inliner::ResolvedCall &resolvedCall) {

  CallOpInterface CB = resolvedCall.call;
  if (auto Skip = getSkipAdviceIfUnreachableCallsite(CB))
    return Skip;


  FunctionOpInterface Caller = CB->getParentOfType<FunctionOpInterface>();
  Operation *calleeOp = call_interface_impl::resolveCallable(CB, &advisorSymbolTable);
  FunctionOpInterface Callee = dyn_cast_or_null<FunctionOpInterface>(calleeOp);

  if (Opts.ml_inliner_skip_policy == SkipMLPolicyCriteria::IfCallerIsNotCold) {
    if (!Caller.getOperation()->hasAttr("cold")) {

      // Return a MLInlineAdvice, despite delegating to the default advice,
      // because we need to keep track of the internal state. This is different
      // from the other instances where we return a "default" InlineAdvice,
      // which happen at points we won't come back to the MLAdvisor for
      // decisions requiring that state.
      return std::make_unique<MLInlineAdvice>(this, CB, getDefaultAdvice(resolvedCall));
    }
  }
  auto MandatoryKind = MLInlineAdvisor::getMandatoryKind(CB, AM);
  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same thing if this is a recursive case.
  if (MandatoryKind == MLInlineAdvisor::MandatoryInliningKind::Never ||
    (Caller == Callee) || (!Caller || !Callee)){
    return getMandatoryAdvice(CB, false);
  }


  bool Mandatory =
      MandatoryKind == MLInlineAdvisor::MandatoryInliningKind::Always;

  // If we need to stop, we won't want to track anymore any state changes, so
  // we just return the base InlineAdvice, which acts as a noop.
  if (ForceStop) {
    // ORE.emit([&] {
    //   return OptimizationRemarkMissed(DEBUG_TYPE, "ForceStop", &CB)
    //          << "Won't attempt inlining because module size grew too much.";
    // });
    auto opts = remark::RemarkOpts::name("ForceStop")
                .category("Inline");

    remark::missed(CB.getOperation()->getLoc(), opts)
      << "Won't attempt inlining because module size grew too much.";
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

  if (Mandatory)
    return getMandatoryAdvice(CB, true);

  auto NumCtantParams = 0;

  for (Value arg : CB.getArgOperands()) {
    if (matchPattern(arg, m_Constant())) {
      NumCtantParams++;
    }
  }


  auto &CallerBefore = getCachedFPI(Caller);
  auto &CalleeBefore = getCachedFPI(Callee);

  auto isAvailableExternally = [](FunctionOpInterface F) {
    auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(F.getOperation());

    return llvmFunc &&
          llvmFunc.getLinkage() == LLVM::Linkage::AvailableExternally;
  };

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
          isAvailableExternally(Callee);
  *ModelRunner->getTensor<int64_t>(llvm::FeatureIndex::is_caller_avail_external) =
          isAvailableExternally(Caller);

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


  Region *CallerRegion =
      resolvedCall.sourceNode->getCallableRegion();
  Region *CalleeRegion =
      resolvedCall.targetNode->getCallableRegion();

  if (!CallerRegion || !CalleeRegion)
    return std::make_unique<MLInlineAdvice>(this, CB, false);

  auto countOps = [](Region *region) -> int64_t {
    unsigned count = 0;
    region->walk([&](Operation *) {
      ++count;
    });
    return count;
  };

  unsigned CallerOps = countOps(CallerRegion);
  unsigned CalleeOps = countOps(CalleeRegion);
  unsigned ratio = CalleeOps * 100 / CallerOps;

  // Add the cost features
  for (size_t I = 0;
       I < static_cast<size_t>(llvm::InlineCostFeatureIndex::NumberOfFeatures); ++I) {
    *ModelRunner->getTensor<int64_t>(inlineCostFeatureToMlFeature(
        static_cast<llvm::InlineCostFeatureIndex>(I))) = ratio;
  }
 
  // This one would have been set up to be right at the end.
  if (!Opts.inliner_interactive_channel_base.empty() &&
      Opts.inliner_interactive_include_default)
    *ModelRunner->getTensor<int64_t>(getFeatureMap().size() - 1) =
        getDefaultAdvice(resolvedCall);
  return getAdviceFromModel(CB);
}

MLInlineAdvisor::MandatoryInliningKind
MLInlineAdvisor::getMandatoryKind(CallOpInterface &CB,
                                  AnalysisManager &am) {

  Operation *callee =
      call_interface_impl::resolveCallable(CB, &am.getAnalysis<SymbolTableAnalysis>().getSymbolTables());

  if (!callee)
    return MandatoryInliningKind::NotMandatory;

  auto hasAttr = [](Operation *op, StringRef name) {
    auto attrs = op->getAttrOfType<ArrayAttr>("passthrough");
    return attrs &&
           llvm::is_contained(
               attrs, StringAttr::get(op->getContext(), name));
  };

  // auto hasPassthroughAttr = [](Operation *op, StringRef name) {
  //   auto attrs = op->getAttrOfType<ArrayAttr>("passthrough");
  //   return attrs && llvm::is_contained(
  //                       attrs, StringAttr::get(op->getContext(), name));
  // };

  if (hasAttr(callee, "noinline"))
    return MandatoryInliningKind::Never;

  // if (hasAttr(callee, "alwaysinline") || hasPassthroughAttr(callee, "alwaysinline"))
  if (hasAttr(callee, "alwaysinline"))
    return MandatoryInliningKind::Always;

  return MandatoryInliningKind::NotMandatory;
}

std::unique_ptr<MLInlineAdvice> MLInlineAdvisor::getAdvice(const Inliner::ResolvedCall &resolvedCall,
                                                       bool MandatoryOnly){

  CallOpInterface CB = resolvedCall.call;

  if (!MandatoryOnly)
    return getAdviceImpl(resolvedCall);

  bool advice =
      resolvedCall.sourceNode != resolvedCall.targetNode &&
      MandatoryInliningKind::Always == getMandatoryKind(CB, AM);

  return getMandatoryAdvice(CB, advice);
}

std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getAdviceFromModel(CallOpInterface CB) {
  return std::make_unique<MLInlineAdvice>(
      this, CB, static_cast<bool>(ModelRunner->evaluate<int64_t>()));
}

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
  return std::make_unique<MLInlineAdvice>(this, CB, Advice);
}

std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getMandatoryAdviceImpl(CallOpInterface &CB) {
  return std::make_unique<MLInlineAdvice>(this, CB, true);
}

void MLInlineAdvisor::print(raw_ostream &OS) const {
  OS << "[MLInlineAdvisor] Nodes: " << NodeCount << " Edges: " << EdgeCount
     << " EdgesOfLastSeenNodes: " << EdgesOfLastSeenNodes << "\n";
  OS << "[MLInlineAdvisor] FPI:\n";
  for (auto I : FPICache) {
    OS << SymbolTable::getSymbolName(I.first).getValue() << ":\n";
    // OS << I.first->getName() << ":\n";
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

MLInlineAdvice::MLInlineAdvice(MLInlineAdvisor *Advisor, CallOpInterface CB,
                               bool Recommendation)
    : Advisor(Advisor), Caller(CB->getParentOfType<FunctionOpInterface>()),
      Callee(dyn_cast_or_null<FunctionOpInterface>(call_interface_impl::resolveCallable(CB, &Advisor->advisorSymbolTable))),
      CallerIRSize(Advisor->isForcedToStop() ? 0 : Advisor->getIRSize(Caller)),
      CalleeIRSize(Advisor->isForcedToStop() ? 0 : Advisor->getIRSize(Callee)),
      CallerAndCalleeEdges(Advisor->isForcedToStop()
                               ? 0
                               : (Advisor->getLocalCalls(Caller) +
                                  Advisor->getLocalCalls(Callee))),
      IsInliningRecommended(Recommendation),
      PreInlineCallerFPI(Advisor->getCachedFPI(Caller)),
      callLoc(CB->getLoc()){

  if (Recommendation)
    Advisor->getCachedFPI(getCaller());
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

void MLInlineAdvice::updateCachedCallerFPI() const {
  // Advisor->updateCachedFPI(Caller);
  Advisor->FPICache.erase(getCaller().getOperation());
  Advisor->getCachedFPI(Caller);
}

void MLInlineAdvice::recordInliningImpl() {
  // ORE.emit([&]() {
    // OptimizationRemark R(DEBUG_TYPE, "InliningSuccess", DLoc, Block);
    // reportContextForRemark(R);
    // return R;
  // });
  auto opts = remark::RemarkOpts::name("InliningSuccess")
                  .category("Inline");
  remark::passed(callLoc, opts)
      << "Successfully inlined call";
  getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ false);
}

void MLInlineAdvice::recordInliningWithCalleeDeletedImpl() {
  // ORE.emit([&]() {
  //   OptimizationRemark R(DEBUG_TYPE, "InliningSuccessWithCalleeDeleted", DLoc,
  //                        Block);
  //   reportContextForRemark(R);
  //   return R;
  // });
  auto opts =
    remark::RemarkOpts::name("InliningSuccessWithCalleeDeleted")
        .category("Inline");

  remark::passed(callLoc, opts)
    << "Successfully inlined call and deleted callee";
  getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ true);
}

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
