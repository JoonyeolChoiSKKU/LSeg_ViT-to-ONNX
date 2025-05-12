import os
import torch
import torch.nn as nn
import numpy as np
# ★ 필요에 따라 대체할 upsampling 파라미터를 정의하세요.
# 원래 encoding 라이브러리에서 사용한 up_kwargs의 내용을 참고하여 동일한 파라미터를 제공해야 합니다.
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

from .models.lseg_net import LSegNet
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class LSegModule:
    def __init__(self, **kwargs):
        # 필수 하이퍼파라미터
        self.crop_size = kwargs["crop_size"]
        # sliding_inference 와 동일한 upsample 옵션
        self._up_kwargs = up_kwargs
        # sliding window 평가용 기본값
        self.base_size = kwargs.get("base_size", 520)   # 원본 기본값 520
        self.mean      = [0.5, 0.5, 0.5]                # 원본 normalize 값
        self.std       = [0.5, 0.5, 0.5]

        # ← 여기를 추가합니다: 원본과 동일한 logit_scale 정의
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        ).exp()
        # minimal label loading (필요한 경우, 혹은 고정된 라벨 리스트로 대체 가능)
        labels = self.get_labels('ade20k')
        self.num_class = len(labels)           # ← 클래스 개수
        self.base_size  = kwargs.get('base_size', 520)  # ← 원본 테스트에선 520 고정
        self.mean, self.std = [0.5]*3, [0.5]*3  # ← Normalize 파라미터
        self.net = LSegNet(
            labels=labels,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=kwargs["crop_size"],
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
            readout=kwargs["readout"],
        )
        # Patch embedding의 이미지 사이즈를 설정 (Image Encoder 추출에 필요)
        #self.net.pretrained.model.patch_embed.img_size = (self.crop_size, self.crop_size)
        self._up_kwargs = up_kwargs

    def get_labels(self, dataset):
        labels = []
        path = 'data/{}_objectInfo150.txt'.format(dataset)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                label = line.strip().split(',')[-1].split(';')[0]
                labels.append(label)
            if dataset in ['ade20k'] and labels:
                labels = labels[1:]
        return labels

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location, **kwargs):
        # 안전하게 checkpoint 내 ModelCheckpoint 글로벌을 허용
        torch.serialization.add_safe_globals([ModelCheckpoint])
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        # checkpoint 구조가 {"model": state_dict, ...} 또는 {"state_dict": ...}인 경우를 모두 처리합니다.
        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        instance = cls(**checkpoint.get("hyper_parameters", {}), **kwargs)
        # hyper_parameters에 없을 수 있는 필드들 보강
        if not hasattr(instance, "crop_size"):
            instance.crop_size = kwargs.get("crop_size", 480)
        if not hasattr(instance, "base_size"):
            instance.base_size = 520
        instance.mean = getattr(instance, "mean", [0.5,0.5,0.5])
        instance.std  = getattr(instance, "std",  [0.5,0.5,0.5])
        instance._up_kwargs = up_kwargs

        # state_dict의 각 키에서 "net." 접두어를 제거합니다.
        new_state = {}
        for key, value in state.items():
            new_key = key
            if key.startswith("net."):
                new_key = key[len("net."):]
            new_state[new_key] = value

        # 모델에 state_dict를 로드합니다.
        instance.net.load_state_dict(new_state, strict=False)
        # 체크포인트에 logit_scale 이 있으면 덮어씌웁니다.
        if "logit_scale" in state:
            instance.logit_scale = state["logit_scale"]
        elif "module.logit_scale" in state:
            instance.logit_scale = state["module.logit_scale"]
        return instance

class LSegModuleWrapper:
    def __init__(self, model):
        self.net = model
