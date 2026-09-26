// RUN: mlir-opt %s -inline='enable-ml-inliner=true' | FileCheck %s
// RUN: mlir-opt %s --mlir-disable-threading -inline='enable-ml-inliner=true' | FileCheck %s

// Test that the ML inliner is wired up and doesn't crash. With a null model
// runner (no compiled model available), the advisor defaults to always
// recommending inlining, so inlining should proceed as normal.

func.func @callee(%arg0 : i32) -> i32 {
  %b = arith.addi %arg0, %arg0 : i32
  return %b : i32
}

// CHECK-LABEL: func @caller
func.func @caller(%arg0 : i32) -> i32 {
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: return
  %0 = call @callee(%arg0) : (i32) -> i32
  return %0 : i32
}
