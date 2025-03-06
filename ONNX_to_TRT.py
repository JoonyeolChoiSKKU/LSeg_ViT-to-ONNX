import tensorrt as trt

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# EXPLICIT_BATCH 플래그 설정 (TensorRT 10.x에서 필요)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_file_path, max_workspace_size=1 << 20):
    """
    ONNX 모델을 TensorRT 엔진으로 변환하는 함수.
    - onnx_file_path: 변환할 ONNX 파일 경로
    - max_workspace_size: 빌드 시 사용할 워크스페이스 크기 (1MB 기본 설정)
    """
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:
        
        # 🛠 TensorRT 10.x에서 최신 API 사용
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        # ONNX 모델 로드
        with open(onnx_file_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                print("❌ ONNX 파싱 실패!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 동적 Shape 최적화 프로파일 설정 (입력: [N, 3, 480, 480])
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name,
                  (1, 3, 480, 480),  # MIN
                  (1, 3, 480, 480),  # OPT
                  (1, 3, 480, 480))  # MAX

        config.add_optimization_profile(profile)

        # TensorRT 엔진 직렬화
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("❌ TensorRT 엔진 직렬화 실패!")
            return None

        # Runtime 생성 및 엔진 역직렬화
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

def save_engine(engine, engine_file_path):
    """ TensorRT 엔진을 파일로 저장하는 함수 """
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    onnx_path = "lseg_image_encoder.onnx"  # 변환할 ONNX 파일
    engine_path = "lseg_image_encoder.trt"  # 변환된 TensorRT 엔진 저장 경로
    
    engine = build_engine(onnx_path)
    if engine:
        save_engine(engine, engine_path)
        print(f"✅ TensorRT 엔진이 {engine_path} 로 저장되었습니다.")
    else:
        print("❌ TensorRT 변환에 실패했습니다.")
