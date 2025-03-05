import torch
import torch.nn as nn
import torch.onnx
from langSeg.lseg_module import LSegModule
from langSeg.lseg_vit import forward_vit

# Image Encoder 래퍼 모듈:
# 원본 LSegNet의 forward에서 텍스트 인코딩과 segmentation head(output_conv) 관련 부분을 제거하고,
# 백본, 중간 feature 추출, 그리고 projection layer(head1)까지의 경로만 그대로 구현합니다.
class LSegImageEncoder(nn.Module):
    def __init__(self, lseg_net):
        super(LSegImageEncoder, self).__init__()
        self.lseg_net = lseg_net

    def forward(self, x):
        # 백본을 통해 중간 feature들을 추출 (lseg_vit.py의 forward_vit 사용)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.lseg_net.pretrained, x)
        
        # 각 feature에 대해 scratch 모듈의 RN 변환 적용
        layer_1_rn = self.lseg_net.scratch.layer1_rn(layer_1)
        layer_2_rn = self.lseg_net.scratch.layer2_rn(layer_2)
        layer_3_rn = self.lseg_net.scratch.layer3_rn(layer_3)
        layer_4_rn = self.lseg_net.scratch.layer4_rn(layer_4)
        
        # RefineNet 스타일의 fusion block 적용
        path_4 = self.lseg_net.scratch.refinenet4(layer_4_rn)
        path_3 = self.lseg_net.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.lseg_net.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.lseg_net.scratch.refinenet1(path_2, layer_1_rn)
        
        # Projection layer: CLIP 기반 head1
        image_features = self.lseg_net.scratch.head1(path_1)
        imshape = image_features.shape
        
        # 재배열 및 정규화: feature map을 (N, out_c, H, W) 형태로 변환
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.lseg_net.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(imshape[0], imshape[2], imshape[3], self.lseg_net.out_c).permute(0, 3, 1, 2)
        
        # arch_option이 1 또는 2이면 head_block 적용 (원본 그대로)
        if self.lseg_net.arch_option in [1, 2] and self.lseg_net.block_depth > 0:
            for _ in range(self.lseg_net.block_depth - 1):
                image_features = self.lseg_net.scratch.head_block(image_features)
            image_features = self.lseg_net.scratch.head_block(image_features, activate=False)
        
        # segmentation head (output_conv)는 제외합니다.
        return image_features

if __name__ == "__main__":
    # 체크포인트 파일 경로 (lseg_app.py와 동일)
    checkpoint_path = "demo_e200.ckpt"
    
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
    onnx_filename = "lseg_image_encoder.onnx"
    
    torch.onnx.export(
        image_encoder,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    print("ONNX export complete:", onnx_filename)
