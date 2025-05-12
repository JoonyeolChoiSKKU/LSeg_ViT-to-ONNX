import tensorrt as trt, os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_dynamic_engine(onnx_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # FP16, Sparse Weights 등 옵션
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<29)

        # ONNX 파싱
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        # Dynamic axes 프로파일 설정: height·width 모두 [260, 480, 910] 범위
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (1, 3, 260, 260),  # MIN
            (1, 3, 480, 480),  # OPT
            (1, 3, 910, 910),  # MAX
        )
        config.add_optimization_profile(profile)

        # 엔진 생성 및 저장
        serialized = builder.build_serialized_network(network, config)
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"✅ Dynamic TRT 엔진 저장: {engine_path}")
    return engine

if __name__ == "__main__":
    onnx_path   = "output/models/lseg_img_enc_vit_ade20k.onnx"
    engine_path = "output/models/lseg_img_enc_vit_ade20k.trt"
    build_dynamic_engine(onnx_path, engine_path)