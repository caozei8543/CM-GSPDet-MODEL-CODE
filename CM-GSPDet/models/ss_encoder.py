import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpectralSpatialDualBranchEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Transformer基础层（局部高频特征提取）
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.local_encoder = TransformerEncoder(encoder_layers, num_layers=3)
        
        # 全局零频上下文提取（全局平均池化+MLP调制）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.modulation_mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model),
            nn.Sigmoid()  # 门控机制，动态调制局部特征
        )

    def _flatten_feature(self, rgb_feat, depth_feat):
        """
        融合RGB和补全后的深度特征，展平为Transformer输入格式
        rgb_feat: (B, d_model, H, W) ResNet50输出特征
        depth_feat: (B, d_model, H, W) 深度特征编码后
        return: feat_flat (B, N, d_model) N=H*W
        """
        B, C, H, W = rgb_feat.shape
        # 跨模态特征融合
        fusion_feat = rgb_feat + depth_feat  # 逐元素融合（论文验证有效）
        # 展平为序列
        feat_flat = fusion_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return feat_flat

    def forward(self, rgb_feat, depth_feat):
        """
        前向传播：双分支特征融合+动态调制
        rgb_feat: (B, d_model, H, W) RGB骨干特征
        depth_feat: (B, d_model, H, W) 补全深度的骨干特征
        return: final_feat (B, d_model, H, W) 调制后的融合特征
        """
        B, C, H, W = rgb_feat.shape
        # 特征展平
        feat_flat = self._flatten_feature(rgb_feat, depth_feat)  # (B, N, C)
        # 1. 局部高频特征提取（Transformer）
        local_feat = self.local_encoder(feat_flat)  # (B, N, C)
        # 2. 全局零频上下文提取+调制权重
        global_feat = self.global_pool(local_feat.permute(0,2,1)).squeeze(-1)  # (B, C)
        mod_weight = self.modulation_mlp(global_feat).unsqueeze(1)  # (B, 1, C)
        # 3. 动态调制：全局上下文门控局部特征
        modulated_feat = local_feat * mod_weight  # (B, N, C)
        # 恢复原特征形状
        final_feat = modulated_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return final_feat