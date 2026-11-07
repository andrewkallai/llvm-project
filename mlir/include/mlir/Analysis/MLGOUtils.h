//===- MLGOUtils.h - Utilities for MLGO Release Mode ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides helper functions for creating MLModelRunners and checking
/// model validity in release mode.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_UTILS_MLGOUTILS_H
#define LLVM_ANALYSIS_UTILS_MLGOUTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/InteractiveModelRunner.h"
// #include "llvm/Analysis/MLModelRunner.h"
#include "mlir/Analysis/MLModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
// #include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <string>
#include <vector>
#include "mlir/IR/MLIRContext.h"


namespace mlir {

/// Helper to check if a release-mode ML advisor has a valid model to execute.
/// Overload for cl::opt<EnumType>.
template <class CompiledModelType, class EnumType, bool ExternalStorage,
          class ParserClass>
bool isReleaseModelValid(
    llvm::StringRef InteractiveChannelBaseName,
    const llvm::cl::opt<EnumType, ExternalStorage, ParserClass> &SelectedModel,
    EnumType DefaultModelVal = EnumType::Default) {
  return llvm::isEmbeddedModelEvaluatorValid<CompiledModelType>() ||
         !InteractiveChannelBaseName.empty() ||
         SelectedModel != DefaultModelVal;
}

/// Helper to check if a release-mode ML advisor has a valid model to execute.
/// Overload for plain EnumType.
template <class CompiledModelType, class EnumType>
bool isReleaseModelValid(llvm::StringRef InteractiveChannelBaseName,
                         EnumType SelectedModel,
                         EnumType DefaultModelVal = EnumType::Default) {
  return llvm::isEmbeddedModelEvaluatorValid<CompiledModelType>() ||
         !InteractiveChannelBaseName.empty() ||
         SelectedModel != DefaultModelVal;
}

/// Helper to construct the appropriate MLModelRunner in release mode:
/// 1. InteractiveModelRunner if an interactive channel is specified.
/// 2. EmitCModelRunner if an EmitC model is selected.
/// 3. ReleaseModeModelRunner<CompiledModelType> if an embedded AOT model is
///    available.
template <class CompiledModelType, class CreateEmitCFunc>
std::unique_ptr<llvm::MLModelRunner> createReleaseModeModelRunner(
    MLIRContext &Ctx, const std::vector<llvm::TensorSpec> &InputFeatures,
    // LLVMContext &Ctx, const std::vector<TensorSpec> &InputFeatures,
    llvm::StringRef DecisionName, const std::string &InteractiveChannelBaseName,
    const llvm::TensorSpec &InteractiveDecisionSpec,
    CreateEmitCFunc &&CreateEmitCModelRunner,
    const llvm::EmbeddedModelRunnerOptions &Options = {}) {
  if (!InteractiveChannelBaseName.empty()) {
    return std::make_unique<llvm::InteractiveModelRunner>(
        Ctx, InputFeatures, InteractiveDecisionSpec,
        InteractiveChannelBaseName + ".out",
        InteractiveChannelBaseName + ".in");
  }
  if (auto EmitCRunner = CreateEmitCModelRunner(Ctx, InputFeatures))
    return EmitCRunner;
  if (llvm::isEmbeddedModelEvaluatorValid<CompiledModelType>()) {
    auto Runner = std::make_unique<llvm::ReleaseModeModelRunner<CompiledModelType>>(
        Ctx, InputFeatures, DecisionName, Options);
    // Shapes the model was compiled for do not match the requested ones. The
    // error is already reported, so let the caller fall back.
    if (!Runner->isValid())
      return nullptr;
    return Runner;
  }
  return nullptr;
}

} // namespace llvm

#endif // LLVM_ANALYSIS_UTILS_MLGOUTILS_H
