#!/usr/bin/env bash

# ONNX와 출력 TRT 엔진 경로
ONNX_PATH="output/models/lseg_img_enc_vit_ade20k.onnx"

# 워크스페이스 (1GiB), FP16 on, sparse on, strict types on,
# timing cache off, precision constraints on, GPU fallback on,
# tactic replay on, debug on, verbose profiling on
python3 conversion/onnx_to_trt.py \
  --onnx   "${ONNX_PATH}" \
  --workspace $((1<<30)) \
  --fp16 \
  --sparse \
  --disable-timing-cache \
  --gpu-fallback \
  --debug