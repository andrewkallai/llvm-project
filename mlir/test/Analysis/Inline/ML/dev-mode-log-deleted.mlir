// REQUIRES: have_tflite, wow
// RUN: mlir-opt -enable-ml-inliner=development --inline \
// RUN:     -training-log=%t < %s 
// RUN: %python %S/../../../../../llvm/lib/Analysis/models/log_reader.py %t | FileCheck %s 

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.module_asm = [], llvm.target_triple = ""} {
  llvm.func @top() -> i32 {
    %0 = llvm.call @to_be_deleted() : () -> i32
    %1 = llvm.call @externally_visible() : () -> i32
    %2 = llvm.add %0, %1 : i32
    llvm.return %2 : i32
  }
  llvm.func internal @to_be_deleted() -> i32 attributes {dso_local} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @externally_visible() -> i32 {
    %0 = llvm.mlir.constant(2 : i32) : i32
    llvm.return %0 : i32
  }
}

// CHECK: observation: 0
// CHECK: inlining_decision: 1
// CHECK: observation: 1
// CHECK: inlining_decision: 1