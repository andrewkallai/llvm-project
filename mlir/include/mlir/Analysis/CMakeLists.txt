set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS AnalysisOptions.td)
llvm_tablegen(AnalysisOptions.inc -gen-opt-parser-defs)
add_mlir_generic_tablegen_target(MLIRAnalysisOptions)