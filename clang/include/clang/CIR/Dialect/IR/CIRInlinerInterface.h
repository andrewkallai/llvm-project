//===- CIRInlinerInterface.h - CIR Inliner Interface ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR inliner interface for the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRINLINERINTERFACE_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRINLINERINTERFACE_H

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
class Block;
class Operation;
class Region;
class ValueRange;
} // namespace mlir

namespace cir {

/// Interface for inlining CIR operations, registered on the CIR dialect.
///
/// Without a `DialectInlinerInterface` for CIR, the generic MLIR `-inline`
/// pass (and therefore the ML-guided inliner that wraps it) reports every
/// `cir.call` as not legal to inline, logging decisions but never changing the
/// IR. This interface makes both the default and the ML inliner actually
/// inline CIR calls.
struct CIRInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===------------------------------------------------------------------===//
  // Analysis Hooks
  //===------------------------------------------------------------------===//

  /// Direct CIR calls are inlinable unless the callee is marked `no_inline`.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final;

  /// All CIR operations can be inlined into a CIR function body.
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final;

  /// CIR function bodies can be inlined into CIR function bodies.
  bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final;

  //===------------------------------------------------------------------===//
  // Transformation Hooks
  //===------------------------------------------------------------------===//

  /// Replace an inlined `cir.return` with a branch to the continuation block.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final;

  /// Replace uses of the inlined call results with the return operands (used
  /// by the single-block fast path).
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final;
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRINLINERINTERFACE_H
