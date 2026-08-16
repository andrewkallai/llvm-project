//===- CIRInlinerInterface.cpp - CIR Inliner Interface -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MLIR inliner interface for the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRInlinerInterface.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIRInlinerInterface
//===----------------------------------------------------------------------===//

bool CIRInlinerInterface::isLegalToInline(Operation *call,
                                          Operation *callable,
                                          bool wouldBeCloned) const {
  // Only direct CIR calls to CIR functions can be inlined.
  auto funcOp = dyn_cast<FuncOp>(callable);
  auto callOp = dyn_cast<CallOp>(call);
  if (!funcOp || !callOp)
    return false;

  // Never inline functions explicitly marked `no_inline`.
  if (auto inlineKind = funcOp.getInlineKind())
    if (*inlineKind == InlineKind::NoInline)
      return false;

  // The caller must be a CIR function so that we can compare return types.
  auto callerOp = call->getParentOfType<FuncOp>();
  Region *funcBody = funcOp.getCallableRegion();
  if (!callerOp || !funcBody)
    return false;

  // `cir.return` ops nested inside sub-regions (early returns inside
  // `cir.if`/`cir.scope`, ...) survive inlining verbatim and become returns of
  // the *caller*, so every such nested return must match the caller's return
  // types. Top-level block terminators are rewritten to branches by
  // handleTerminator and are therefore always safe.
  llvm::ArrayRef<Type> callerResultTypes = callerOp.getResultTypes();
  bool compatible = true;
  funcBody->walk([&](ReturnOp ret) {
    // `cir.return` is a Terminator, so any return living directly in the
    // function body region is a top-level block terminator (handled by
    // handleTerminator). Only nested returns need the type check below.
    if (ret->getParentRegion() == funcBody)
      return WalkResult::advance();
    if (ret.getNumOperands() != callerResultTypes.size() ||
        !llvm::equal(ret.getOperandTypes(), callerResultTypes)) {
      compatible = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return compatible;
}

bool CIRInlinerInterface::isLegalToInline(Operation *op, Region *dest,
                                          bool wouldBeCloned,
                                          IRMapping &valueMapping) const {
  // Every CIR operation (including those nested in `cir.scope`, `cir.if` and
  // `cir.while` regions) can be inlined into a CIR function body.
  return true;
}

bool CIRInlinerInterface::isLegalToInline(Region *dest, Region *src,
                                          bool wouldBeCloned,
                                          IRMapping &valueMapping) const {
  return true;
}

void CIRInlinerInterface::handleTerminator(Operation *op,
                                           Block *newDest) const {
  // Only `cir.return` needs to be handled here.
  auto returnOp = dyn_cast<ReturnOp>(op);
  if (!returnOp)
    return;

  // Replace the return with an unconditional branch to the continuation block,
  // forwarding the returned values.
  OpBuilder builder(op);
  BrOp::create(builder, op->getLoc(), newDest, returnOp.getOperands());
  op->erase();
}

void CIRInlinerInterface::handleTerminator(Operation *op,
                                           ValueRange valuesToRepl) const {
  // Only `cir.return` needs to be handled here (single-block fast path).
  auto returnOp = dyn_cast<ReturnOp>(op);
  if (!returnOp)
    return;

  // Replace the inlined call results directly with the return operands.
  assert(returnOp.getNumOperands() == valuesToRepl.size() &&
         "expected return operand count to match call result count");
  for (const auto &it : llvm::enumerate(returnOp.getOperands()))
    valuesToRepl[it.index()].replaceAllUsesWith(it.value());
}
