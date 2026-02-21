import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean
import math


# ==========================================
# 1. 基础组件：正弦时间步嵌入
# ==========================================
class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==========================================
# 2. 核心：E(3) 等变图网络层 (EGNN Layer)
# ==========================================
class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=0, act_fn=nn.SiLU()):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim), act_fn,
            nn.Linear(hidden_dim, hidden_dim), act_fn
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        # 🚨 [关键修复 1]：将坐标更新的最后一层权重强制初始化为 0。
        # 这确保了模型在最开始不会乱动原子，而是从 0 开始平稳学习
        nn.init.zeros_(self.coord_mlp[2].weight)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index
        coord_diff = x[row] - x[col]

        # 🚨 [关键修复 2]：对平方距离做 max 截断 (比如 100 埃平方)
        # 防止周期性边界跨越导致的异常大距离炸毁网络
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        radial = torch.clamp(radial, max=100.0)

        # ... 后续代码保持不变 ...

        if edge_attr is not None:
            edge_inputs = torch.cat([h[row], h[col], radial, edge_attr], dim=1)
        else:
            edge_inputs = torch.cat([h[row], h[col], radial], dim=1)

        edge_messages = self.edge_mlp(edge_inputs)

        coord_weights = self.coord_mlp(edge_messages)
        coord_updates = coord_diff * coord_weights

        coord_update_sum = scatter_sum(coord_updates, row, dim=0, dim_size=x.size(0))
        x_new = x + coord_update_sum

        node_messages = scatter_sum(edge_messages, row, dim=0, dim_size=h.size(0))
        node_inputs = torch.cat([h, node_messages], dim=1)
        h_new = h + self.node_mlp(node_inputs)

        return h_new, x_new


# ==========================================
# 3. 2D 材料扩散主模型 (支持 CFG 条件引导)
# ==========================================
class DenoisingEGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 🚨 [关键修复 3]：使用 Embedding 替代 Linear 处理原子序数 (假设最大元素号为 100)
        self.node_embed = nn.Embedding(100, hidden_dim)
        self.time_embed = SinusoidalTimeEmbeddings(hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # 属性条件融合 MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.egnn_layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

        # A. 坐标噪声预测头
        self.x_noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )

        # B. 2D 晶格约束
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 9)
        )
        self.register_buffer("lattice_2d_mask", torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]]))

        # C. 多任务预测头 (3个分支)
        self.delta_g_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.stability_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.synthesizability_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, z, pos, edge_index, batch, time_step, context_delta_g=None, context_stability=None,
                p_uncond=0.15):
        # ... 前面的 cfg_mask 逻辑不变 ...
        """
        :param p_uncond: Classifier-Free Guidance 的无条件生成概率 (默认 15%)
        """
        batch_size = time_step.size(0)
        device = pos.device

        # ==========================================
        # 模块内聚：自动生成 CFG 掩码并丢弃条件
        # ==========================================
        # 仅在训练模式 (self.training) 下进行随机 Mask，推断时由外部手动控制
        if self.training and p_uncond > 0.0:
            cfg_mask = (torch.rand(batch_size, device=device) < p_uncond).float()
        else:
            cfg_mask = torch.zeros(batch_size, device=device)

        if context_delta_g is None: context_delta_g = torch.zeros(batch_size, device=device)
        if context_stability is None: context_stability = torch.zeros(batch_size, device=device)

        # 实施遮蔽：被 Mask 的样本，条件变为 0
        context_delta_g = context_delta_g * (1 - cfg_mask)
        context_stability = context_stability * (1 - cfg_mask)

        # ==========================================
        # 骨干网络前向传播
        # ==========================================
        h = self.node_embed(z.squeeze())
        t_emb = self.time_mlp(self.time_embed(time_step))

        ctx_cat = torch.stack([context_delta_g, context_stability], dim=-1)
        ctx_emb = self.context_mlp(ctx_cat)

        # 融合 时间+属性，注入节点
        t_ctx_emb = t_emb + ctx_emb
        h = h + t_ctx_emb[batch]

        # EGNN 消息传递
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index)

        # 预测坐标与 2D 晶格
        eps_x = self.x_noise_head(h)
        graph_latent = scatter_mean(h, batch, dim=0)

        eps_lattice_flat = self.lattice_head(graph_latent)
        eps_lattice = eps_lattice_flat.view(-1, 3, 3) * self.lattice_2d_mask.unsqueeze(0)

        # 预测三大属性
        pred_delta_g = self.delta_g_head(graph_latent).squeeze(-1)
        pred_stability = self.stability_head(graph_latent).squeeze(-1)
        pred_synth_logits = self.synthesizability_head(graph_latent).squeeze(-1)

        return {
            "eps_x": eps_x,
            "eps_lattice": eps_lattice,
            "delta_g": pred_delta_g,
            "stability": pred_stability,
            "synth_logits": pred_synth_logits,
            "cfg_mask": cfg_mask  # 返回 mask 供 optimization 中的联合损失函数使用
        }


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes = 12
    batch_size = 2
    hidden_dim = 128

    z_fake = torch.randn(num_nodes, 1).to(device)  # 测试时改为 1 维输入匹配 train.py
    pos_noisy = torch.randn(num_nodes, 3).to(device)
    edge_index_fake = torch.randint(0, num_nodes, (2, 30)).to(device)
    batch_fake = torch.cat([torch.zeros(6), torch.ones(6)]).long().to(device)
    time_fake = torch.randint(0, 1000, (batch_size,)).to(device)

    # 测试传入的引导条件 (例如，我们希望生成 ΔG_H 为 0.05 的完美催化剂)
    ctx_delta_g_fake = torch.tensor([0.05, -0.1], device=device)
    ctx_stab_fake = torch.tensor([0.0, 0.2], device=device)

    model = DenoisingEGNN(num_node_features=1, hidden_dim=hidden_dim).to(device)

    # 前向传播测试 (带条件)
    outputs = model(z_fake, pos_noisy, edge_index_fake, batch_fake, time_fake,
                    context_delta_g=ctx_delta_g_fake, context_stability=ctx_stab_fake)

    print("=== Denoising EGNN (Conditional) Forward Test ===")
    print(f"坐标噪声预测形状 (eps_x): {outputs['eps_x'].shape}")
    print(f"晶格噪声预测形状 (eps_lattice): {outputs['eps_lattice'].shape}")
    print(f"属性预测输出 (delta_g): {outputs['delta_g'].shape}")
    print("✅ 模型已成功支持带属性条件的输入！")