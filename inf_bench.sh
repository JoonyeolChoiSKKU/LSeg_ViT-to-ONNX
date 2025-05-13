#!/usr/bin/env bash
# inference_benchmark.sh

# 설정
ENGINE=output/models/lseg_img_enc_vit_ade20k.trt
ITERS=1000
# 테스트할 사이즈 리스트 (H W)
SIZES=(260)

echo "***** C++ TensorRT Multi-Scale Benchmark *****"
for S in "${SIZES[@]}"; do
  echo
  echo ">> Testing ${S}x${S}"
  ./build/trt_cpp_infer_time_tester "$ENGINE" "$ITERS" "$S" "$S"
done
