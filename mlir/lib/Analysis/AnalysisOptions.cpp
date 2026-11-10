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




llvm::ManagedStatic<AnalysisOptions> clOptions;

void mlir::registerAnalysisCLOptions() {
  // Construct the options and register them with the command-line parser.
  (void)*clOptions;
}

const AnalysisOptions &mlir::getAnalysisOptions() {
  return *clOptions;
}


const AnalysisOptions &mlir::getAnalysisOptions(Operation &op) {
  return getAnalysisOptions();
}
const AnalysisOptions &mlir::getAnalysisOptions(ModuleOp &moduleOp) {
  return getAnalysisOptions();
}

const AnalysisOptions &mlir::getAnalysisOptions(MLIRContext &Ctx) {
  return getAnalysisOptions();
}