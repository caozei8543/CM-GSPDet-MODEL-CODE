import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from numba import jit

class CrossModalGSPFilter(nn.Module):
    def __init__(self, sigma_s=0.1, sigma_c=0.05, alpha=0.1):
        super().__init__()
        self.sigma_s = sigma_s  # 空间相似度缩放（论文经验值）
        self.sigma_c = sigma_c  # 颜色相似度缩放（论文经验值）
        self.alpha = alpha      # 图正则化权重（论文2.1节设为0.1）

    def _build_adjacency_matrix(self, rgb):
        """
        构建邻接矩阵W，对应论文公式(1)：W_ij = exp(-||p_i-p_j||²/2σs² - ||c_i-c_j||²/2σc²)
        rgb: (B, 3, H, W) 归一化后的RGB图像
        return: W (B, H*W, H*W) 邻接矩阵
        """
        B, C, H, W = rgb.shape
        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)  # (B, N, 3), N=H*W
        # 构建空间坐标p_i (B, N, 2)
        x = torch.linspace(0, 1, W, device=rgb.device).repeat(H, 1)
        y = torch.linspace(0, 1, H, device=rgb.device).unsqueeze(1).repeat(1, W)
        pos = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).reshape(1, H*W, 2).repeat(B, 1, 1)
        
        # 计算空间距离||p_i-p_j||²
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 2)
        pos_dist = torch.sum(pos_diff ** 2, dim=-1)     # (B, N, N)
        # 计算颜色距离||c_i-c_j||²
        rgb_diff = rgb_flat.unsqueeze(2) - rgb_flat.unsqueeze(1)  # (B, N, N, 3)
        rgb_dist = torch.sum(rgb_diff ** 2, dim=-1)               # (B, N, N)
        # 公式(1)计算邻接矩阵
        W = torch.exp(-pos_dist/(2*self.sigma_s**2) - rgb_dist/(2*self.sigma_c**2))
        # 自环置0
        mask = torch.eye(H*W, device=W.device).unsqueeze(0).repeat(B, 1, 1)
        W = W * (1 - mask)
        return W

    def _graph_laplacian(self, W):
        """计算非归一化图拉普拉斯矩阵L=D-W，D为度矩阵"""
        B, N, _ = W.shape
        D = torch.sum(W, dim=-1)  # (B, N) 度向量
        D = torch.diag_embed(D)   # (B, N, N) 度矩阵
        L = D - W
        return L

    def _graph_regularized_recovery(self, depth_sparse, W):
        """
        图正则化信号恢复，对应论文公式(2)(3)：min||Mx-y||² + αx^TLx → (M+αL)x=My
        depth_sparse: (B, 1, H, W) 稀疏深度图（点云投影得到）
        W: (B, N, N) 邻接矩阵
        return: depth_dense (B, 1, H, W) 恢复后的稠密深度图
        """
        B, C, H, W = depth_sparse.shape
        N = H * W
        # 构建掩码矩阵M：有深度值为1，无则为0
        depth_flat = depth_sparse.reshape(B, 1, N).permute(0, 2, 1)  # (B, N, 1)
        M = (depth_flat != 0).float().squeeze(-1)  # (B, N)
        M = torch.diag_embed(M)                    # (B, N, N) 掩码对角矩阵
        y = depth_flat.squeeze(-1)                 # (B, N) 稀疏深度信号
        
        # 计算拉普拉斯矩阵L
        L = self._graph_laplacian(W)
        # 公式(3)：求解(M + αL)x = My
        A = M + self.alpha * L  # (B, N, N)
        b = torch.bmm(M, y.unsqueeze(-1)).squeeze(-1)  # (B, N)
        # 求解线性方程组Ax=b（PyTorch批处理求解）
        x = torch.linalg.solve(A, b)  # (B, N) 恢复的稠密深度信号
        depth_dense = x.reshape(B, H, W).unsqueeze(1)  # (B, 1, H, W)
        return depth_dense

    @jit(nopython=True)  # 加速计算
    def forward(self, rgb, depth_sparse):
        """
        前向传播：RGB引导的稀疏深度补全
        rgb: (B, 3, H, W) 归一化RGB图像（ResNet50输入格式）
        depth_sparse: (B, 1, H, W) 点云投影的稀疏深度图
        return: depth_dense (B, 1, H, W) 各向异性平滑的稠密深度图
        """
        # 构建邻接矩阵
        W = self._build_adjacency_matrix(rgb)
        # 图正则化深度恢复
        depth_dense = self._graph_regularized_recovery(depth_sparse, W)
        # 限制深度值非负
        depth_dense = F.relu(depth_dense)
        return depth_dense