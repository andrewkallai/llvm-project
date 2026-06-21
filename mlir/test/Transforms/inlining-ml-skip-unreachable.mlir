// RUN: mlir-opt %s -inline='enable-ml-inliner=true' | FileCheck %s

// Test that the ML inliner skips unreachable callsites. The call in the
// unreachable block is not inlined (and the block is cleaned up), but the
// ML inliner should not crash.

func.func @callee(%arg0 : i32) -> i32 {
  %b = arith.addi %arg0, %arg0 : i32
  return %b : i32
}

// CHECK-LABEL: func.func @caller
func.func @caller(%arg0 : i32) -> i32 {
  cf.br ^exit
^dead:
  %0 = call @callee(%arg0) : (i32) -> i32
  return %0 : i32
^exit:
  return %arg0 : i32
}
