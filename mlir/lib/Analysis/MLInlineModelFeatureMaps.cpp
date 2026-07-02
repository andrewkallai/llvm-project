//===- MLInlineModelFeatureMaps.cpp - feature map definitions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/MLInlineModelFeatureMaps.h"

using namespace mlir;

const char *const mlir::MLIRDecisionName = "inlining_decision";
const char *const mlir::MLIRRewardName = "inlining_reward";

// Use a function-local static to avoid global destructor.
const llvm::TensorSpec &getMLIRInlineDecisionSpec() {
  static const llvm::TensorSpec Spec =
      llvm::TensorSpec::createSpec<int64_t>(MLIRDecisionName, {1});
  return Spec;
}
