add_subdirectory(Presburger)

add_mlir_unittest(MLIRMLGOUtilsTests
  MLGOUtilsTest.cpp
  MLModelRunnerTest.cpp
)

mlir_target_link_libraries(MLIRMLGOUtilsTests
  PRIVATE
  MLIRAnalysis
)