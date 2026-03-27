import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicGWD(nn.Module):
    def __init__(self, tau=0.3):
        super().__init__()
        self.tau = tau  # 损失灵敏度超参数（论文2.3节设为0.3）

    def _box2gaussian(self, bboxes_3d):
        """
        3D框转3D高斯分布N(μ, Σ)，对应论文公式(4)(5)
        bboxes_3d: (B, N, 7) 3D框格式：(cx, cy, cz, dx, dy, dz, θ)
        return: mu (B, N, 3), sigma (B, N, 3, 3) 均值和协方差矩阵
        """
        B, N, _ = bboxes_3d.shape
        cx, cy, cz = bboxes_3d[...,0], bboxes_3d[...,1], bboxes_3d[...,2]
        dx, dy, dz = bboxes_3d[...,3], bboxes_3d[...,4], bboxes_3d[...,5]
        theta = bboxes_3d[...,6]  # yaw角（绕z轴旋转）
        
        # 公式(4)：均值向量μ
        mu = torch.stack([cx, cy, cz], dim=-1)  # (B, N, 3)
        
        # 构建旋转矩阵Rz(θ)（绕z轴）
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        Rz = torch.zeros(B, N, 3, 3, device=bboxes_3d.device)
        Rz[...,0,0] = cos_theta
        Rz[...,0,1] = -sin_theta
        Rz[...,1,0] = sin_theta
        Rz[...,1,1] = cos_theta
        Rz[...,2,2] = 1.0
        
        # 公式(5)：协方差矩阵Σ = Rz * diag(dx²/4, dy²/4, dz²/4) * Rz^T
        diag = torch.stack([dx**2/4, dy**2/4, dz**2/4], dim=-1)  # (B, N, 3)
        diag_mat = torch.diag_embed(diag)  # (B, N, 3, 3)
        sigma = torch.matmul(torch.matmul(Rz, diag_mat), Rz.transpose(-2, -1))  # (B, N, 3, 3)
        return mu, sigma

    def _gaussian_wasserstein2(self, mu_p, sigma_p, mu_t, sigma_t):
        """
        计算二阶高斯沃瑟斯坦距离W²，对应论文公式(6)
        W² = ||μp-μt||² + Tr(Σp + Σt - 2*(Σp^0.5 Σt Σp^0.5)^0.5)
        mu_p/sigma_p: 预测框的均值/协方差
        mu_t/sigma_t: 真实框的均值/协方差
        return: w2_sq (B, N) 每个框的W²距离
        """
        # 均值差的L2平方
        mu_diff = mu_p - mu_t
        mu_dist_sq = torch.sum(mu_diff ** 2, dim=-1)  # (B, N)
        
        # 计算协方差项：Tr(Σp + Σt - 2*(Σp^0.5 Σt Σp^0.5)^0.5)
        B, N = mu_p.shape[:2]
        cov_term = torch.zeros(B, N, device=mu_p.device)
        for b in range(B):
            for n in range(N):
                sp = sigma_p[b, n]  # (3,3)
                st = sigma_t[b, n]  # (3,3)
                # 计算Σp^0.5（特征值分解）
                eigval, eigvec = torch.linalg.eig(sp)
                eigval = torch.clamp(eigval.real, min=1e-8)  # 避免负特征值
                sp_sqrt = torch.matmul(torch.matmul(eigvec.real, torch.diag(torch.sqrt(eigval))), eigvec.real.T)
                # 计算Σp^0.5 Σt Σp^0.5
                spstsp = torch.matmul(torch.matmul(sp_sqrt, st), sp_sqrt)
                # 计算平方根
                eigval2, eigvec2 = torch.linalg.eig(spstsp)
                eigval2 = torch.clamp(eigval2.real, min=1e-8)
                spstsp_sqrt = torch.matmul(torch.matmul(eigvec2.real, torch.diag(torch.sqrt(eigval2))), eigvec2.real.T)
                # 迹计算
                tr = torch.trace(sp + st - 2 * spstsp_sqrt)
                cov_term[b, n] = tr.real
        
        # 总W²距离
        w2_sq = mu_dist_sq + cov_term
        return w2_sq

    def forward(self, pred_bboxes_3d, gt_bboxes_3d):
        """
        前向传播：计算动态GWD损失，对应论文公式(7)：L_GWD = 1 - exp(-W²/(τ*(det(Σt))^(1/3)))
        pred_bboxes_3d: (B, N, 7) 预测3D框
        gt_bboxes_3d: (B, N, 7) 真实3D框（已对齐）
        return: loss (scalar) 批量平均GWD损失
        """
        # 3D框转高斯分布
        mu_p, sigma_p = self._box2gaussian(pred_bboxes_3d)
        mu_t, sigma_t = self._box2gaussian(gt_bboxes_3d)
        
        # 计算W²距离
        w2_sq = self._gaussian_wasserstein2(mu_p, sigma_p, mu_t, sigma_t)  # (B, N)
        
        # 计算det(Σt)^(1/3)（目标协方差的行列式的立方根，尺度自适应项）
        det_sigma_t = torch.linalg.det(sigma_t)  # (B, N)
        det_sigma_t = torch.clamp(det_sigma_t, min=1e-8)  # 避免行列式为0
        det_term = torch.pow(det_sigma_t, 1/3)  # (B, N)
        
        # 公式(7)计算GWD损失
        loss_per_box = 1 - torch.exp(-w2_sq / (self.tau * det_term))  # (B, N)
        # 过滤无效框（gt为0的位置）
        valid_mask = (gt_bboxes_3d[...,0] != 0).float()  # (B, N)
        loss = (loss_per_box * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        return loss