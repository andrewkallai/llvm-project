//===- MLInlineModelFeatureMaps.h - MLIR model runner feature defs -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the feature maps for the ML-based MLIR inliner.
// It is analogous to llvm/include/llvm/Analysis/InlineModelFeatureMaps.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_MLINLINEMODELFEATUREMAPS_H
#define MLIR_ANALYSIS_MLINLINEMODELFEATUREMAPS_H

#include "llvm/Analysis/TensorSpec.h"

namespace mlir {

// ---------------------------------------------------------------------------
// Feature macros
// ---------------------------------------------------------------------------

// Features derived from the callable's region structure (the "callee" side).
#define CALLEE_REGION_FEATURES(M)                                              \
  M(int64_t, {1}, callee_block_count,                                          \
    "Number of blocks in the callee region")                                   \
  M(int64_t, {1}, callee_region_count,                                         \
    "Number of nested regions in the callee")                                  \
  M(int64_t, {1}, callee_operand_count,                                        \
    "Total operand count across all ops in the callee")                        \
  M(int64_t, {1}, callee_result_count,                                         \
    "Total result count across all ops in the callee")                         \
  M(int64_t, {1}, callee_arg_count,                                            \
    "Number of arguments to the callee's entry block")                         \
  M(int64_t, {1}, callee_is_isolated_from_above,                               \
    "Whether the callee region is isolated from above")

// Features derived from the caller's region structure.
#define CALLER_REGION_FEATURES(M)                                              \
  M(int64_t, {1}, caller_block_count,                                          \
    "Number of blocks in the caller region")                                   \
  M(int64_t, {1}, caller_region_count,                                         \
    "Number of nested regions in the caller")                                  \
  M(int64_t, {1}, caller_operand_count,                                        \
    "Total operand count across all ops in the caller")                        \
  M(int64_t, {1}, caller_result_count,                                         \
    "Total result count across all ops in the caller")                         \
  M(int64_t, {1}, caller_arg_count,                                            \
    "Number of arguments to the caller's entry block")                         \
  M(int64_t, {1}, caller_is_isolated_from_above,                               \
    "Whether the caller region is isolated from above")

// Call-site-level features.
#define CALL_SITE_FEATURES(M)                                                  \
  M(int64_t, {1}, call_site_operand_count,                                     \
    "Number of operands at the call site")                                     \
  M(int64_t, {1}, call_site_num_ctant_args,                                    \
    "Number of constant arguments at the call site")                           \
  M(int64_t, {1}, callsite_height,                                              \
    "Position of the call site in the call graph (0 = leaf)")

// Module-level graph features.
#define GRAPH_FEATURES(M)                                                      \
  M(int64_t, {1}, graph_node_count,                                            \
    "Total number of call-graph nodes (externals excluded)")                   \
  M(int64_t, {1}, graph_edge_count,                                            \
    "Total number of call edges (externals excluded)")                         \
  M(int64_t, {1}, graph_callee_region_level,                                   \
    "Depth of the callee region in the call graph (leaf == 0)")                \
  M(int64_t, {1}, graph_initial_total_ops,                                     \
    "Total operation count before any inlining")                               \
  M(int64_t, {1}, graph_current_total_ops_ratio,                               \
    "Ratio of current to initial total operation count, scaled by 100")

// All features combined.
#define ALL_FEATURES(M)                                                        \
  CALLEE_REGION_FEATURES(M)                                                    \
  CALLER_REGION_FEATURES(M)                                                    \
  CALL_SITE_FEATURES(M)                                                        \
  GRAPH_FEATURES(M)

// ---------------------------------------------------------------------------
// Feature index enum
// ---------------------------------------------------------------------------

enum class MLIRInlineFeatureIndex : size_t {
#define POPULATE_INDICES(DTYPE, SHAPE, NAME, DOC) NAME,
  ALL_FEATURES(POPULATE_INDICES)
#undef POPULATE_INDICES
      NumFeatures
};

// ---------------------------------------------------------------------------
// Decision name and spec, reward name
// ---------------------------------------------------------------------------

extern const char *const MLIRDecisionName;
extern const llvm::TensorSpec MLIRInlineDecisionSpec;

extern const char *const MLIRRewardName;

} // namespace mlir

#endif // MLIR_ANALYSIS_MLINLINEMODELFEATUREMAPS_H
