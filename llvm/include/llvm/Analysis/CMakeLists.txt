set(LLVM_TARGET_DEFINITIONS TargetLibraryInfo.td)
llvm_tablegen(TargetLibraryInfo.inc -gen-target-library-info)

set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS AnalysisOptions.td)
llvm_tablegen(AnalysisOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(analysis_gen)
