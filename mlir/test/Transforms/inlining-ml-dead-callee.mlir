// RUN: mlir-opt %s -inline='enable-ml-inliner=true' | FileCheck %s

// Test that the ML inliner correctly handles dead callees. The dead functions
// are mutually recursive and can't be inlined, but the ML inliner should not
// crash or produce errors.

func.func @live() -> i32 {
  %c = arith.constant 1 : i32
  return %c : i32
}

// CHECK-LABEL: func.func @dead_a
func.func @dead_a() -> i32 {
  // CHECK: call @dead_b
  %r = call @dead_b() : () -> i32
  return %r : i32
}

func.func @dead_b() -> i32 {
  %r = call @dead_a() : () -> i32
  return %r : i32
}
