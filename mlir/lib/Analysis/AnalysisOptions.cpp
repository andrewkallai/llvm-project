//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AnalysisOptions.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Utils/ImportedFunctionsInliningStatistics.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/LibraryOptions.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ManagedStatic.h"



#define OPTIONS_STRUCT_DEFS
#include "mlir/Analysis/AnalysisOptions.inc"

using namespace mlir;
// using namespace llvm;

// static llvm::opt::RegisterLibraryOptions<llvm::AnalysisOptions> Registration;

// const AnalysisOptions &llvm::getAnalysisOptions(const Module &M) {
//   return getAnalysisOptions(M.getContext());
// }

// const AnalysisOptions &llvm::getAnalysisOptions(const Loop &L) {
//   return getAnalysisOptions(L.getHeader()->getContext());
// }


// namespace {

// struct AnalysisOptions {
//   llvm::cl::opt<std::string> inlinerInteractiveChannelBase{
//       "inliner-interactive-channel-base", llvm::cl::Hidden,
//       llvm::cl::desc("Base file path for the interactive model runner"),
//       llvm::cl::init("")};

//   llvm::cl::opt<SkipMLPolicyCriteria> mlInlinerSkipPolicy{
//       "ml-inliner-skip-policy", llvm::cl::Hidden,
//       llvm::cl::init(SkipMLPolicyCriteria::Never),
//       llvm::cl::values(
//           clEnumValN(SkipMLPolicyCriteria::Never, "never", "never"),
//           clEnumValN(SkipMLPolicyCriteria::IfCallerIsNotCold,
//                      "if-caller-not-cold",
//                      "Skip ML policy when the caller is not cold"))};
// };

// } // namespace

// static llvm::ManagedStatic<mlir::AnalysisOptions> clOptions;

// void mlir::registerAnalysisCLOptions() {
//   (void)*clOptions;
// }

// const AnalysisOptions &mlir::getAnalysisOptions() {
//   assert(clOptions.isConstructed() &&
//          "Analysis command-line options were not registered");

//   static const AnalysisOptions opts = [] {
//     AnalysisOptions result;
//     result.inliner_interactive_channel_base =
//         clOptions->inlinerInteractiveChannelBase;
//     result.ml_inliner_skip_policy = clOptions->mlInlinerSkipPolicy;
//     return result;
//   }();

//   return opts;
// }


// llvm::ManagedStatic<AnalysisOptions> clOptions;
llvm::ManagedStatic<AnalysisOptions> options;

// static llvm::opt::RegisterLibraryOptions<mlir::AnalysisOptions>
    // Registration;

// void mlir::registerAnalysisCLOptions() {
//   // Construct the options and register them with the command-line parser.
//   (void)*clOptions;
// }

void mlir::setAnalysisOptions(const AnalysisOptions &Opts) {
  *options = Opts;
}
const AnalysisOptions &mlir::getAnalysisOptions() {
  return *options;
}
// const AnalysisOptions &mlir::getAnalysisOptions() {
//   return *clOptions;
// }


const AnalysisOptions &mlir::getAnalysisOptions(Operation &op) {
  return getAnalysisOptions();
}
const AnalysisOptions &mlir::getAnalysisOptions(ModuleOp &moduleOp) {
  return getAnalysisOptions();
}

const AnalysisOptions &mlir::getAnalysisOptions(MLIRContext &Ctx) {
  return getAnalysisOptions();
}



// const mlir::AnalysisOptions &
// mlir::getAnalysisOptions(Operation &op) {
//   return getAnalysisOptions(*op.getContext());
// }

// const mlir::AnalysisOptions &
// mlir::getAnalysisOptions(ModuleOp &moduleOp) {
//   return getAnalysisOptions(*moduleOp.getContext());
// }

// const mlir::AnalysisOptions &
// mlir::getAnalysisOptions(MLIRContext &ctx) {
//   (void)ctx;
//   return getAnalysisOptions();
// }