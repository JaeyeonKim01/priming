from dataclasses import dataclass
from typing import List


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int # number of past frames given to the U-Net as context
    cond_channels: int # dim of unified conditioning vector
    depths: List[int] # num ResNet blocks at each downsampling resolution level
    channels: List[int] # num feature map channels at each U-Net resolution level
    attn_depths: List[bool] # whether self-attention/cross-attention should be applied at each specific resolution block
    num_actions: int = 7 # dim of action vector


def get_inner_model_config(config_name: str) -> InnerModelConfig:
    if config_name == "Bridge_UNet":
        return InnerModelConfig(
            img_channels=16,
            num_steps_conditioning=4,
            cond_channels=256,
            depths=[2, 2, 2, 2],
            channels=[160, 320, 640, 640],
            attn_depths=[0, 0, 1, 1],
            num_actions=7,
        )
    else:
        raise ValueError(f"Unknown InnerModelConfig name: {config_name}")
