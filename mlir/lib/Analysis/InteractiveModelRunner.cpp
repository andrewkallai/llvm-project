//===- InteractiveModelRunner.cpp - noop ML model runner   ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A runner that communicates with an external agent via 2 file descriptors.
//===----------------------------------------------------------------------===//
#include "mlir/Analysis/InteractiveModelRunner.h"
#include "mlir/Analysis/AnalysisOptions.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

InteractiveModelRunner::InteractiveModelRunner(
    MLIRContext &Ctx, const std::vector<llvm::TensorSpec> &Inputs,
    const llvm::TensorSpec &Advice, StringRef OutboundName, StringRef InboundName)
    : MLModelRunner(Ctx, MLModelRunner::Kind::Interactive, Inputs.size()),
      InputSpecs(Inputs), OutputSpec(Advice),
      InEC(llvm::sys::fs::openFileForRead(InboundName, Inbound)),
      OutputBuffer(OutputSpec.getTotalTensorBufferSize()) {
  if (InEC) {
    emitError(UnknownLoc::get(&Ctx), "Cannot open inbound file: " + InEC.message());

    return;
  }
  {
    auto OutStream = std::make_unique<llvm::raw_fd_ostream>(OutboundName, OutEC);
    if (OutEC) {
      emitError(UnknownLoc::get(&Ctx), "Cannot open outbound file: " + OutEC.message());
      return;
    }
    Log = std::make_unique<llvm::Logger>(std::move(OutStream), InputSpecs, Advice,
                                   /*IncludeReward=*/false, Advice);
  }
  // Just like in the no inference case, this will allocate an appropriately
  // sized buffer.
  for (size_t I = 0; I < InputSpecs.size(); ++I)
    setUpBufferForTensor(I, InputSpecs[I], nullptr);
  Log->flush();
}

InteractiveModelRunner::~InteractiveModelRunner() {
  llvm::sys::fs::file_t FDAsOSHandle = llvm::sys::fs::convertFDToNativeFile(Inbound);
  llvm::sys::fs::closeFile(FDAsOSHandle);
}

void *InteractiveModelRunner::evaluateUntyped() {
  Log->startObservation();
  for (size_t I = 0; I < InputSpecs.size(); ++I)
    Log->logTensorValue(I, reinterpret_cast<const char *>(getTensorUntyped(I)));
  Log->endObservation();
  Log->flush();

  size_t InsPoint = 0;
  char *Buff = OutputBuffer.data();
  const size_t Limit = OutputBuffer.size();
  while (InsPoint < Limit) {
    auto ReadOrErr = llvm::sys::fs::readNativeFile(
        llvm::sys::fs::convertFDToNativeFile(Inbound),
        {Buff + InsPoint, OutputBuffer.size() - InsPoint});
    if (ReadOrErr.takeError()) {
      // Ctx.emitError("Failed reading from inbound file");
      emitError(UnknownLoc::get(&Ctx), "Failed reading from inbound file");
      break;
    }
    InsPoint += *ReadOrErr;
  }
  if (getAnalysisOptions(Ctx).interactive_model_runner_echo_reply)
    llvm::dbgs() << OutputSpec.name() << ": "
           << tensorValueToString(OutputBuffer.data(), OutputSpec) << "\n";
  return OutputBuffer.data();
}
