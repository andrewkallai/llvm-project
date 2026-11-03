//===- MLInlineAdvisor.h - ML - based InlineAdvisor factories ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MLINLINEADVISOR_H
#define LLVM_ANALYSIS_MLINLINEADVISOR_H

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/IR/PassManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/CallInterfaces.h"
// #include "mlir/Pass/PassManager.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Analysis/CallGraph.h"




#include <map>
#include <memory>
#include <optional>

// namespace llvm {
namespace mlir {
class DiagnosticInfoOptimizationBase;
// class Module;
class MLInlineAdvice;
class ProfileSummaryInfo;

// class LLVM_ABI MLInlineAdvisor : public llvm::InlineAdvisor {
class LLVM_ABI MLInlineAdvisor {
public:
/*HERE*/
  // MLInlineAdvisor(Module &M, ModuleAnalysisManager &MAM,
  MLInlineAdvisor(ModuleOp &M, ModuleAnalysisManager &MAM,
                  std::function<std::unique_ptr<llvm::MLModelRunner>(
                      const std::vector<llvm::TensorSpec> &)>
                      GetModelRunner,
                  std::function<bool(CallOpInterface &)> GetDefaultAdvice);
                  // std::function<bool(CallOpInterface &)> GetDefaultAdvice);
                  // std::function<bool(CallBase &)> GetDefaultAdvice);

  ~MLInlineAdvisor() = default;
  // ~MLInlineAdvisor() override = default;

  void onPassEntry(llvm::LazyCallGraph::SCC *SCC);
  void onPassExit(llvm::LazyCallGraph::SCC *SCC);
  // void onPassEntry(llvm::LazyCallGraph::SCC *SCC) override;
  // void onPassExit(llvm::LazyCallGraph::SCC *SCC) override;

  // int64_t getIRSize(Function &F) const {
  int64_t getIRSize(FunctionOpInterface &F) const {
    return getCachedFPI(F).TotalInstructionCount;
  }
  void onSuccessfulInlining(const MLInlineAdvice &Advice,
                            bool CalleeWasDeleted);

  bool isForcedToStop() const { return ForceStop; }
  // int64_t getLocalCalls(Function &F);
  int64_t getLocalCalls(FunctionOpInterface &F);
  const llvm::MLModelRunner &getModelRunner() const { return *ModelRunner; }
  // FunctionPropertiesInfo &getCachedFPI(Function &) const;
  llvm::FunctionPropertiesInfo &getCachedFPI(FunctionOpInterface) const;
  const std::vector<llvm::TensorSpec> &getFeatureMap() const { return FeatureMap; };
  static const std::vector<llvm::TensorSpec> &getInitialFeatureMap();

protected:
  // std::unique_ptr<InlineAdvice> getAdviceImpl(CallBase &CB) override;
  std::unique_ptr<MLInlineAdvice> getAdviceImpl(CallOpInterface CB);

  // std::unique_ptr<InlineAdvice> getMandatoryAdvice(CallBase &CB,
  std::unique_ptr<MLInlineAdvice> getMandatoryAdvice(CallOpInterface CB,
                                                   bool Advice);

  // virtual std::unique_ptr<MLInlineAdvice> getMandatoryAdviceImpl(CallBase &CB);
  // virtual std::unique_ptr<MLInlineAdvice> getMandatoryAdviceImpl(CallOpInterface &CB);
  std::unique_ptr<MLInlineAdvice> getMandatoryAdviceImpl(CallOpInterface &CB);

  // virtual std::unique_ptr<MLInlineAdvice>
  std::unique_ptr<MLInlineAdvice>
  getAdviceFromModel(CallOpInterface CB);
  // getAdviceFromModel(CallBase &CB, OptimizationRemarkEmitter &ORE);

  // Get the initial 'level' of the function, or 0 if the function has been
  // introduced afterwards.
  // TODO: should we keep this updated?
  // unsigned getInitialFunctionLevel(const Function &F) const;
  unsigned getInitialFunctionLevel(FunctionOpInterface F) const;

  std::unique_ptr<llvm::MLModelRunner> ModelRunner;
  // std::function<bool(CallBase &)> GetDefaultAdvice;
  std::function<bool(CallOpInterface &)> GetDefaultAdvice;
  std::vector<llvm::TensorSpec> FeatureMap;

private:
  int64_t getModuleIRSize() const;
  std::unique_ptr<MLInlineAdvice>
  getSkipAdviceIfUnreachableCallsite(CallOpInterface CB);
  // getSkipAdviceIfUnreachableCallsite(CallBase &CB);
  void print(raw_ostream &OS) const;
  // void print(raw_ostream &OS) const override;

  // Using std::map to benefit from its iterator / reference non-invalidating
  // semantics, which make it easy to use `getCachedFPI` results from multiple
  // calls without needing to copy to avoid invalidation effects.
  mutable std::map<const llvm::Function *, llvm::FunctionPropertiesInfo> FPICache;

  // llvm::LazyCallGraph &CG;
  CallGraph &CG;

  int64_t NodeCount = 0;
  int64_t EdgeCount = 0;
  int64_t EdgesOfLastSeenNodes = 0;
  // const bool UseIR2Vec;

  // std::map<const llvm::LazyCallGraph::Node *, unsigned> FunctionLevels;
  llvm::DenseMap<CallGraphNode *, unsigned> FunctionLevels;

  const int32_t InitialIRSize = 0;
  int32_t CurrentIRSize = 0;
  llvm::SmallPtrSet<const llvm::LazyCallGraph::Node *, 1> NodesInLastSCC;
  // DenseSet<const llvm::LazyCallGraph::Node *> AllNodes;
  llvm::DenseSet<CallGraphNode *> AllNodes;
  // DenseSet<Function *> DeadFunctions;
  // DenseSet<FunctionOpInterface *> DeadFunctions;
  llvm::DenseSet<Operation *> DeadFunctions;

  bool ForceStop = false;
  // ProfileSummaryInfo &PSI;
  SymbolTableCollection advisorSymbolTable;
};

/// InlineAdvice that tracks changes post inlining. For that reason, it only
/// overrides the "successful inlining" extension points.
// class LLVM_ABI MLInlineAdvice : public llvm::InlineAdvice {
class LLVM_ABI MLInlineAdvice {
public:
  // MLInlineAdvice(MLInlineAdvisor *Advisor, CallBase &CB,
  MLInlineAdvice(MLInlineAdvisor *Advisor, CallOpInterface CB,
                 bool Recommendation);
                //  llvm::OptimizationRemarkEmitter &ORE, bool Recommendation);
  // ~MLInlineAdvice() override = default;
  ~MLInlineAdvice() = default;

  // void recordInliningImpl() override;
  // void recordInliningWithCalleeDeletedImpl() override;
  // void recordUnsuccessfulInliningImpl(const llvm::InlineResult &Result) override;
  // void recordUnattemptedInliningImpl() override;
  void recordInliningImpl();
  void recordInliningWithCalleeDeletedImpl();
  void recordUnsuccessfulInliningImpl(const llvm::InlineResult &Result);
  void recordUnattemptedInliningImpl();

  // MLInlineAdvisor *const Advisor;

  FunctionOpInterface Caller;
  FunctionOpInterface Callee;

  FunctionOpInterface getCaller() const { return Caller; }
  FunctionOpInterface getCallee() const { return Callee; }

  const int64_t CallerIRSize;
  const int64_t CalleeIRSize;
  const int64_t CallerAndCalleeEdges;
  void updateCachedCallerFPI(llvm::FunctionAnalysisManager &FAM) const;

private:
  void reportContextForRemark(DiagnosticInfoOptimizationBase &OR);
  // MLInlineAdvisor *getAdvisor() const {
  //   // return static_cast<MLInlineAdvisor *>(Advisor);
  //   return Advisor;
  // };
  // Make a copy of the FPI of the caller right before inlining. If inlining
  // fails, we can just update the cache with that value.
  const llvm::FunctionPropertiesInfo PreInlineCallerFPI;
  std::optional<llvm::FunctionPropertiesUpdater> FPU;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MLINLINEADVISOR_H
