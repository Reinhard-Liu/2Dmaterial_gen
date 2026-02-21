import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


# ==========================================
# 1. 多任务联合损失函数 (Multi-Task Loss)
# ==========================================
class JointDiffusionLoss(nn.Module):
    def __init__(self, lambda_diff=1.0, lambda_her=1.0, lambda_stab=0.5, lambda_synth=0.5):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_her = lambda_her
        self.lambda_stab = lambda_stab
        self.lambda_synth = lambda_synth

        # 使用 reduction='none' 以便我们根据 CFG mask 灵活屏蔽无条件样本的属性 Loss
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_noise, target_noise, pred_her, target_her,
                pred_stab, target_stab, pred_synth, target_synth, cfg_mask):
        """
        :param cfg_mask: 形状 [batch_size]，值为 1 代表该样本被随机 Mask（无条件生成）
        """
        # 有效条件掩码：被 Mask 的样本不参与属性 Loss 的反向传播
        valid_mask = 1.0 - cfg_mask

        # A. 扩散重构损失 (无论是否无条件，都要学习去噪)
        # 将 [num_nodes, 3] 展平计算 MSE
        loss_diff = self.mse(pred_noise, target_noise).mean()

        # B. 属性预测损失 (仅针对有条件的样本)
        loss_her = (self.mse(pred_her, target_her) * valid_mask).mean()
        loss_stab = (self.mse(pred_stab, target_stab) * valid_mask).mean()
        loss_synth = (self.bce(pred_synth, target_synth) * valid_mask).mean()

        # C. 联合总损失
        total_loss = (self.lambda_diff * loss_diff +
                      self.lambda_her * loss_her +
                      self.lambda_stab * loss_stab +
                      self.lambda_synth * loss_synth)

        return total_loss, loss_diff, loss_her, loss_stab, loss_synth


# ==========================================
# 2. 学习率调度器 (Cosine Annealing with Warmup)
# ==========================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    带预热 (Warmup) 的余弦退火学习率调度器。
    早期训练容易梯度爆炸，Warmup 能让学习率从 0 稳步上升，之后再按余弦曲线衰减。
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ==========================================
# 3. 梯度裁剪工具 (Gradient Clipping)
# ==========================================
def clip_gradients(model, max_norm=1.0):
    """
    为防止 EGNN 在早期训练时因距离计算 (radial) 引发梯度爆炸，限制梯度的最大范数。
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)