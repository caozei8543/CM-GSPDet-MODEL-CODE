import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from models.cm_gsp import CrossModalGSPFilter
from models.ss_encoder import SpectralSpatialDualBranchEncoder
from models.dynamic_gwd import DynamicGWD

# 点云骨干：PointNet++（轻量版，适配室内点云）
class PointNetPPFeat(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        """x: (B, 3, N) 点云坐标"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=-1)[0]  # 全局池化 (B, 256)
        return x

class ARCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = cfg['model']['d_model']
        
        # 1. 骨干网络
        # RGB骨干：ResNet50（截断到layer4，输出2048→256）
        self.rgb_backbone = resnet50(pretrained=True)
        self.rgb_feat_proj = nn.Sequential(
            nn.Conv2d(2048, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        # 点云骨干：PointNet++（投影到2D深度特征）
        self.pc_backbone = PointNetPPFeat(out_dim=d_model)
        self.depth_feat_proj = nn.Conv2d(d_model, d_model, 1)
        
        # 2. 论文三大创新模块
        self.cm_gsp = CrossModalGSPFilter(
            sigma_s=cfg['model']['sigma_s'],
            sigma_c=cfg['model']['sigma_c'],
            alpha=cfg['model']['alpha']
        )
        self.ss_encoder = SpectralSpatialDualBranchEncoder(
            d_model=d_model,
            nhead=cfg['model']['nhead'],
            dim_feedforward=cfg['model']['dim_feedforward']
        )
        self.dynamic_gwd = DynamicGWD(tau=cfg['model']['tau'])
        
        # 3. 3D检测头（预测7维3D框：cx,cy,cz,dx,dy,dz,θ）
        self.det_head = nn.Sequential(
            nn.Conv2d(d_model, d_model//2, 3, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model//2, 7)
        )

    def _pc2depth(self, pc_feat, H, W):
        """点云特征投影为2D深度特征图"""
        B, C = pc_feat.shape
        depth_feat = pc_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        depth_feat = self.depth_feat_proj(depth_feat)
        return depth_feat

    def forward(self, rgb, depth_sparse, pc, gt_bboxes_3d=None):
        """
        前向传播：训练/测试一体化
        rgb: (B, 3, H, W) 归一化RGB
        depth_sparse: (B, 1, H, W) 稀疏深度
        pc: (B, 3, N) 点云坐标
        gt_bboxes_3d: (B, 7) 真实3D框（训练时传入）
        return: 训练时返回loss，测试时返回pred_bboxes_3d
        """
        B, _, H, W = rgb.shape
        # 1. 骨干特征提取
        rgb_feat = self.rgb_backbone.features(rgb)  # (B, 2048, H//32, W//32)
        rgb_feat = self.rgb_feat_proj(rgb_feat)    # (B, 256, H//32, W//32)
        pc_feat = self.pc_backbone(pc)             # (B, 256)
        depth_feat = self._pc2depth(pc_feat, rgb_feat.shape[2], rgb_feat.shape[3])  # (B,256,H//32,W//32)
        
        # 2. CM-GSP：跨模态深度补全
        depth_sparse_down = F.interpolate(depth_sparse, size=rgb_feat.shape[2:], mode='nearest')
        depth_dense = self.cm_gsp(rgb_feat, depth_sparse_down)  # (B,1,H//32,W//32)
        # 深度特征融合
        depth_dense_feat = depth_feat * depth_dense  # 补全深度引导点云特征
        
        # 3. SS-Encoder：谱-空双分支特征调制
        fusion_feat = self.ss_encoder(rgb_feat, depth_dense_feat)  # (B,256,H//32,W//32)
        
        # 4. 检测头：预测3D框
        pred_bboxes_3d = self.det_head(fusion_feat)  # (B,7)
        pred_bboxes_3d = pred_bboxes_3d.unsqueeze(1)  # (B,1,7) 适配GWD输入格式
        
        # 训练时计算GWD损失，测试时返回预测框
        if self.training and gt_bboxes_3d is not None:
            gt_bboxes_3d = gt_bboxes_3d.unsqueeze(1)  # (B,1,7)
            loss = self.dynamic_gwd(pred_bboxes_3d, gt_bboxes_3d)
            return loss
        else:
            return pred_bboxes_3d.squeeze(1)