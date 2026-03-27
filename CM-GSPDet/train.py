import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
from data.sunrgbd.dataset import SUNRGBDDataset
from models.arc_net import ARCNet

# 加载配置
with open("./config/sunrgbd.yaml", 'r') as f:
    cfg = yaml.safe_load(f)
device = torch.device(cfg['train']['device'])

# 创建数据集和加载器
train_dataset = SUNRGBDDataset(cfg, split='train')
train_loader = DataLoader(
    train_dataset, batch_size=cfg['train']['batch_size'],
    shuffle=True, num_workers=4, pin_memory=True
)

# 初始化模型、优化器
model = ARCNet(cfg).to(device)
optimizer = optim.AdamW(
    model.parameters(), lr=cfg['train']['lr'],
    weight_decay=cfg['train']['weight_decay']
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

# 创建保存目录
os.makedirs(cfg['train']['save_path'], exist_ok=True)

# 训练主循环
model.train()
best_loss = float('inf')
for epoch in range(cfg['train']['epochs']):
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        depth_sparse = batch['depth_sparse'].to(device)
        pc = batch['pc'].to(device)
        gt_bboxes_3d = batch['gt_bboxes_3d'].to(device)
        
        # 前向传播
        loss = model(rgb, depth_sparse, pc, gt_bboxes_3d)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['train']['grad_clip'])  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item(), "Avg Loss": total_loss/(len(pbar))})
    
    # 学习率更新
    lr_scheduler.step()
    
    # 保存最优模型
    avg_loss = total_loss / len(train_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(cfg['train']['save_path'], "arc_net_best.pth"))
    # 保存最新模型
    torch.save(model.state_dict(), os.path.join(cfg['train']['save_path'], f"arc_net_epoch{epoch+1}.pth"))
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}")

print("Training Finished!")