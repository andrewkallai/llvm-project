// RUN: mlir-opt %s -inline='enable-ml-inliner=true' | FileCheck %s

// Test that the ML inliner handles a module with recursive functions gracefully.
// The recursion prevents full inlining, but the ML inliner should not crash.
// A non-recursive caller should still have its call inlined.

func.func @factorial(%arg0 : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cmp = arith.cmpi eq, %arg0, %c0 : i32
  cf.cond_br %cmp, ^base, ^recurse(%arg0 : i32)
^base:
  cf.br ^exit(%c1 : i32)
^recurse(%n : i32):
  %sub = arith.subi %n, %c1 : i32
  %rec = call @factorial(%sub) : (i32) -> i32
  %result = arith.muli %n, %rec : i32
  cf.br ^exit(%result : i32)
^exit(%ret : i32):
  return %ret : i32
}

func.func @call_factorial() -> i32 {
  %c5 = arith.constant 5 : i32
  %r = call @factorial(%c5) : (i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @factorial
// CHECK: call @factorial
// CHECK-LABEL: func.func @call_factorial
// CHECK: return
