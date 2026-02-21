import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增绘图库

from dataset.material_dataset import MaterialDataset
from models.diffusion_model import DenoisingEGNN
from models.optimization import JointDiffusionLoss, get_cosine_schedule_with_warmup, clip_gradients


def get_noise_schedule(num_steps=1000, device='cuda'):
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动 2D HER 生成器训练 (模块化版) | 设备: {device}")

    # 超参数设置
    epochs = 100
    batch_size = 32
    timesteps = 1000

    # 初始化数据与模型
    dataset = MaterialDataset(
        root='./dataset',
        db_path=r'D:\Programming Software\github_project\MachineLearning_MG\dataset\c2db\c2db.db',
        dimenet_weights_path=r'D:\Programming Software\github_project\MachineLearning_MG\models\weights\dimenet_best_ocp.pth'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DenoisingEGNN(num_node_features=1, hidden_dim=128).to(device)

    # 初始化优化组件
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    loss_fn = JointDiffusionLoss()
    alpha_bar = get_noise_schedule(timesteps, device)

    # 日志与保存
    os.makedirs('results/checkpoints', exist_ok=True)
    writer = SummaryWriter('results/tensorboard_logs')
    best_loss = float('inf')

    # 🌟 新增：用于记录绘图的列表
    history_epoch_loss = []

    print("开始训练...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 提取真实属性
            target_delta_g = batch.delta_g_h
            target_stability = batch.ehull
            target_synth = (batch.ehull <= 0.0).float()

            z = batch.z.long()
            pos = batch.pos

            # 加噪过程
            t = torch.randint(0, timesteps, (batch.num_graphs,), device=device).long()
            t_node = t[batch.batch]
            a_bar_t = alpha_bar[t_node].unsqueeze(-1)
            noise = torch.randn_like(pos)
            pos_noisy = torch.sqrt(a_bar_t) * pos + torch.sqrt(1 - a_bar_t) * noise

            # 模型预测
            outputs = model(z=z, pos=pos_noisy, edge_index=batch.edge_index, batch=batch.batch,
                            time_step=t,
                            context_delta_g=target_delta_g,
                            context_stability=target_stability,
                            p_uncond=0.15)

            # 计算联合损失
            total_loss, l_diff, l_her, l_stab, l_synth = loss_fn(
                pred_noise=outputs['eps_x'], target_noise=noise,
                pred_her=outputs['delta_g'], target_her=target_delta_g,
                pred_stab=outputs['stability'], target_stab=target_stability,
                pred_synth=outputs['synth_logits'], target_synth=target_synth,
                cfg_mask=outputs['cfg_mask']
            )

            # 反向传播与梯度裁剪
            total_loss.backward()
            clip_gradients(model, max_norm=0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_epoch_loss = epoch_loss / len(dataloader)
        history_epoch_loss.append(avg_epoch_loss)  # 记录 Loss

        writer.add_scalar('Loss/Total', avg_epoch_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'results/checkpoints/best_diffusion_model.pth')
            print(f"🌟 新的最佳模型已保存 (Loss: {best_loss:.4f})")

    writer.close()

    # 🌟 新增：训练结束后绘制并保存 Loss 曲线

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history_epoch_loss, color='b', linewidth=2, label='Training Loss')
    plt.title('Diffusion Model Joint Training Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=300)
    plt.close()
    print("✅ 训练全部结束！Loss 曲线已保存至 results/loss_curve.png")


if __name__ == "__main__":

    train()