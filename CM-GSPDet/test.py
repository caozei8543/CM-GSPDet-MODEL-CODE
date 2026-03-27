# 核心测试逻辑（test.py关键片段）
model.eval()
with torch.no_grad():
    for batch in test_loader:
        rgb = batch['rgb'].to(device)
        depth_sparse = batch['depth_sparse'].to(device)
        pc = batch['pc'].to(device)
        gt_bboxes_3d = batch['gt_bboxes_3d'].cpu().numpy()
        # 预测3D框
        pred_bboxes_3d = model(rgb, depth_sparse, pc).cpu().numpy()
        # 计算3D IoU和AP/AR（utils/metrics.py实现）
        metrics.update(pred_bboxes_3d, gt_bboxes_3d)
# 打印论文同款指标
ap_novel, ap_base, ap_avg = metrics.get_ap()
ar_novel, ar_base, ar_avg = metrics.get_ar()
print(f"APnovel: {ap_novel:.2f}%, APbase: {ap_base:.2f}%, APavg: {ap_avg:.2f}%")
print(f"ARnovel: {ar_novel:.2f}%, ARbase: {ar_base:.2f}%, ARavg: {ar_avg:.2f}%")