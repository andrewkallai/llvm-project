// The default inliner doesn't elide @adder, it believes it's too costly to inline 
// adder into switcher. The ML inliner carries out that inlining, resulting in
// a smaller result (part of it is that adder gets elided).
//
// This test uses Inputs/test-module.mlir, as it shares it with a similar test
// for the 'release' mode.
//
// REQUIRES: have_tflite, wow
// RUN: rm -rf %t
// RUN: rm -rf %t_savedmodel
// RUN: %python %S/../../../../../llvm/lib/Analysis/models/gen-inline-oz-test-model.py %t_savedmodel
// RUN: %python %S/../../../../../llvm/lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
// RUN: mlir-opt --inline='enable-ml-inliner=release' < %S/Inputs/test-module.mlir 2>&1 | FileCheck %S/Inputs/test-module.mlir --check-prefix=DEFAULT
// RUN: mlir-opt --inline='enable-ml-inliner=development' -ml-inliner-model-under-training=%t < %S/Inputs/test-module.mlir 2>&1 | FileCheck %S/Inputs/test-module.mlir --check-prefix=CHECK
// RUN: mlir-opt --inline='enable-ml-inliner=development' -training-log=%t.log < %S/Inputs/test-module.mlir 2>&1
// RUN: %python %S/../../../../../llvm/lib/Analysis/models/log_reader.py %t.log | FileCheck %s --check-prefix=CHECK-LOG

CHECK-LOG: observation: 0
CHECK-LOG-NEXT: {{^sroa_savings:}} 0
CHECK-LOG: {{^cost_estimate:}} -30
CHECK-LOG: {{^inlining_decision:}} 1
CHECK-LOG-NEXT: observation: 1
CHECK-LOG: observation: 5
