#!/usr/bin/env bash
set -euo pipefail

# 변환할 ONNX 파일 경로
ONNX_PATH="output/models/lseg_img_enc_vit_ade20k_clean.onnx"

# (옵션 없이) 순수 기본값으로 TRT 엔진 생성
python3 conversion/onnx_to_trt.py \
    --onnx "${ONNX_PATH}"

echo "✅ 모든 최적화 옵션이 비활성화된 엔진이 생성되었습니다."