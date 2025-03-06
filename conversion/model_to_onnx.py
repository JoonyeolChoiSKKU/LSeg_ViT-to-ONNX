import torch
import torch.nn as nn
import torch.onnx
from lseg.lseg_module import LSegModule
from lseg.image_encoder import LSegImageEncoder

# Image Encoder 래퍼 모듈:
# 원본 LSegNet의 forward에서 텍스트 인코딩과 segmentation head(output_conv) 관련 부분을 제거하고,
# 백본, 중간 feature 추출, 그리고 projection layer(head1)까지의 경로만 그대로 구현합니다.


if __name__ == "__main__":
    # 체크포인트 파일 경로 (lseg_app.py와 동일)
    checkpoint_path = "models/demo_e200.ckpt"
    
    # LSegModule을 사용해 체크포인트에서 모델을 로드합니다.
    lseg_module = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_path='../datasets/',
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu'
    )
    
    # lseg_module.net가 존재하면 해당 네트워크를 사용
    if hasattr(lseg_module, 'net'):
        lseg_net = lseg_module.net
    else:
        lseg_net = lseg_module

    # 이미지 인코딩 래퍼 모듈 생성
    image_encoder = LSegImageEncoder(lseg_net)
    image_encoder.eval()

    # ONNX 내보내기: 입력 이미지 크기는 (1, 3, 480, 480)로 고정합니다.
    dummy_input = torch.randn(1, 3, 480, 480)
    onnx_filename = "models/lseg_image_encoder.onnx"
    
    torch.onnx.export(
        image_encoder,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes=None
    )
    
    print("ONNX export complete:", onnx_filename)
