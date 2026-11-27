module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-grtev4-linux-gnu"} {
  llvm.func @external_fct(i32)
  llvm.func @top() -> i32 attributes {dso_local, minsize} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.call @multiplier(%0) : (i32) -> i32
    %3 = llvm.call @adder(%1) : (i32) -> i32
    %4 = llvm.add %2, %3 overflow<nsw> : i32
    llvm.call @external_fct(%4) : (i32) -> ()
    llvm.return %4 : i32
  }
  llvm.func internal @adder(%arg0: i32) -> i32 attributes {dso_local, minsize} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 <alignment = 4> : i32, !llvm.ptr
    %2 = llvm.load %1 <alignment = 4> : !llvm.ptr -> i32
    %3 = llvm.call @multiplier(%2) : (i32) -> i32
    %4 = llvm.load %1 <alignment = 4> : !llvm.ptr -> i32
    %5 = llvm.call @switcher(%0) : (i32) -> i32
    %6 = llvm.add %3, %5 overflow<nsw> : i32
    llvm.return %6 : i32
  }
  llvm.func internal @multiplier(%arg0: i32) -> i32 attributes {dso_local, minsize} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 <alignment = 4> : i32, !llvm.ptr
    %2 = llvm.load %1 <alignment = 4> : !llvm.ptr -> i32
    %3 = llvm.load %1 <alignment = 4> : !llvm.ptr -> i32
    %4 = llvm.mul %2, %3 overflow<nsw> : i32
    llvm.return %4 : i32
  }
  llvm.func @switcher(%arg0: i32) -> i32 attributes {minsize} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 <alignment = 4> : i32, !llvm.ptr
    %4 = llvm.load %3 <alignment = 4> : !llvm.ptr -> i32
    llvm.switch %4 : i32, ^bb3 [
      1: ^bb1,
      2: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    llvm.store %1, %2 <alignment = 4> : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb2:  // pred: ^bb0
    %5 = llvm.load %3 <alignment = 4> : !llvm.ptr -> i32
    %6 = llvm.load %3 <alignment = 4> : !llvm.ptr -> i32
    %7 = llvm.call @multiplier(%6) : (i32) -> i32
    %8 = llvm.add %5, %7 overflow<nsw> : i32
    llvm.store %8, %2 <alignment = 4> : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb3:  // pred: ^bb0
    %9 = llvm.call @adder(%1) : (i32) -> i32
    llvm.store %9, %2 <alignment = 4> : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
    %10 = llvm.load %2 <alignment = 4> : !llvm.ptr -> i32
    llvm.return %10 : i32
  }
}

// Previously check-not, defaul-next on the last
// CHECK: @multiplier
// DEFAULT-LABEL:        llvm.func internal @adder(%arg0: i32)
// DEFAULT:         llvm.mul %2