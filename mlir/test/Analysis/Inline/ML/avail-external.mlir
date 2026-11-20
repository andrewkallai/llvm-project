// REQUIRES: x86_64-linux, wow
// RUN: rm -rf %t.rundir
// RUN: rm -rf %t.channel-basename.*
// RUN: mkdir %t.rundir
// RUN: cp %S/../../../../../llvm/lib/Analysis/models/log_reader.py %t.rundir
// RUN: cp %S/../../../../../llvm/lib/Analysis/models/interactive_host.py %t.rundir
// RUN: cp %S/Inputs/interactive_main.py %t.rundir
// RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
// RUN:    mlir-opt --inline -interactive-model-runner-echo-reply \
// RUN:    -enable-ml-inliner=release -inliner-interactive-channel-base=%t.channel-basename %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.module_asm = [], llvm.target_triple = ""} {
  llvm.func available_externally @g() {
    llvm.return
  }
  llvm.func @f() {
    llvm.call @g() : () -> ()
    llvm.return
  }
}

// CHECK: is_callee_avail_external: 1
// CHECK: is_caller_avail_external: 0