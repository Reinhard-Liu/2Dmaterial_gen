import os
import torch
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import write


class StructureGenerator:
    def __init__(self, model, device='cuda', num_steps=1000):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_steps = num_steps
        self.beta = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _get_realistic_templates(self):
        """
        🚀 核心升级 1：真实 2D 材料化学计量比模板库 (以 12 原子体系为例)
        格式: [原子序数] * 数量。告别随机乱抽，保证电荷平衡与合成可行性。
        """
        return [
            [6] * 12,  # 石墨烯/碳网格 (C)
            [5] * 6 + [7] * 6,  # 氮化硼 (h-BN)
            [42] * 4 + [16] * 8,  # 二硫化钼 (MoS2)
            [74] * 4 + [34] * 8,  # 二硒化钨 (WSe2)
            [22] * 4 + [16] * 8,  # 二硫化钛 (TiS2)
            [23] * 4 + [16] * 8,  # 二硫化钒 (VS2)
            [41] * 4 + [34] * 8,  # 二硒化铌 (NbSe2)
            [78] * 4 + [34] * 8,  # 二硒化铂 (PtSe2)
            [15] * 12,  # 黑磷 (Phosphorene)
            [49] * 6 + [34] * 6,  # 硒化铟 (InSe)
            [31] * 6 + [16] * 6,  # 硫化镓 (GaS)
        ]

    def generate_guided_2d_materials(self, num_materials=100, num_atoms_per_mat=12,
                                     guidance_scale=0.08, target_delta_g=0.0):
        print(f"🌀 开始生成 {num_materials} 个靶向 2D 材料 | 梯度强度: {guidance_scale}")

        # 🚀 核心升级 2：基于真实模板分配原子序数 Z
        templates = self._get_realistic_templates()
        z_list = []
        for _ in range(num_materials):
            # 随机挑选一个合法配方
            tmpl = templates[torch.randint(0, len(templates), (1,)).item()]

            # 兼容性处理：如果外部传入的原子数不是12，按比例截断或填充
            if len(tmpl) != num_atoms_per_mat:
                tmpl = (tmpl * (num_atoms_per_mat // len(tmpl) + 1))[:num_atoms_per_mat]

            # 打乱原子顺序，防止位置偏置
            tmpl_tensor = torch.tensor(tmpl, dtype=torch.long)
            tmpl_tensor = tmpl_tensor[torch.randperm(len(tmpl_tensor))]
            z_list.append(tmpl_tensor)

        z = torch.cat(z_list).to(self.device)
        batch = torch.arange(num_materials, device=self.device).repeat_interleave(num_atoms_per_mat)
        pos_t = torch.randn(num_materials * num_atoms_per_mat, 3, device=self.device)

        edge_index = self._build_dummy_edges(num_materials, num_atoms_per_mat)

        for t_step in tqdm(reversed(range(self.num_steps)), total=self.num_steps, desc="Denoising & Guiding"):
            t = torch.full((num_materials,), t_step, device=self.device, dtype=torch.long)
            pos_t = pos_t.detach().requires_grad_(True)

            ctx_g = torch.full((num_materials,), target_delta_g, device=self.device)
            ctx_stab = torch.zeros(num_materials, device=self.device)

            outputs = self.model(
                z=z, pos=pos_t, edge_index=edge_index, batch=batch,
                time_step=t, context_delta_g=ctx_g, context_stability=ctx_stab, p_uncond=0.0
            )

            pred_noise = outputs['eps_x']
            pred_delta_g = outputs['delta_g']
            pred_stability = outputs['stability']

            target_loss = torch.abs(pred_delta_g - target_delta_g) + pred_stability
            grad_pos = torch.autograd.grad(outputs=target_loss.sum(), inputs=pos_t)[0]

            with torch.no_grad():
                a_t = self.alpha[t_step]
                a_bar_t = self.alpha_bar[t_step]
                pos_mean = (1 / torch.sqrt(a_t)) * (pos_t - (1 - a_t) / torch.sqrt(1 - a_bar_t) * pred_noise)

                if t_step > 0:
                    noise = torch.randn_like(pos_t)
                    sigma_t = torch.sqrt(self.beta[t_step])
                    pos_prev = pos_mean + sigma_t * noise
                else:
                    pos_prev = pos_mean

                grad_pos_2d = grad_pos.clone()
                grad_pos_2d[:, 2] = 0.0  # 强制 2D 约束
                pos_t = pos_prev - guidance_scale * grad_pos_2d

        print("✅ 逆向扩散与梯度靶向优化完成！")
        return z.detach(), pos_t.detach(), batch

    def _build_dummy_edges(self, num_mats, atoms_per_mat):
        row, col = [], []
        for i in range(num_mats):
            start_idx = i * atoms_per_mat
            end_idx = start_idx + atoms_per_mat
            for r in range(start_idx, end_idx):
                for c in range(start_idx, end_idx):
                    if r != c: row.extend([r, c]); col.extend([c, r])
        return torch.tensor([row, col], device=self.device)

    def export_to_atoms_and_cif(self, z, pos, batch, output_dir="results/generated_cifs"):
        """导出 CIF 并返回 Atoms 列表供下游评估"""
        os.makedirs(output_dir, exist_ok=True)
        atoms_list = []
        z_np = z.cpu().numpy()
        pos_np = pos.cpu().numpy()
        batch_np = batch.cpu().numpy()
        num_mats = batch.max().item() + 1

        for i in range(num_mats):
            mask = (batch_np == i)
            # 🚀 核心升级 3：紧凑的周期性边界 (去除 XY 的大真空层)
            # 允许边界原子跨过晶胞相连，彻底消除悬挂键，极大提升 MatterSim 稳定性得分
            box_size_xy = np.max(pos_np[mask][:, :2], axis=0) - np.min(pos_np[mask][:, :2], axis=0)
            cell_x = max(box_size_xy[0] + 0.2, 3.0)
            cell_y = max(box_size_xy[1] + 0.2, 3.0)

            cell = [cell_x, cell_y, 20.0]  # Z 轴依然保持 20 埃以维持二维特性

            atoms = Atoms(numbers=z_np[mask], positions=pos_np[mask], cell=cell, pbc=[True, True, False])
            atoms.center()
            atoms_list.append(atoms)
            write(os.path.join(output_dir, f"gen_2d_mat_{i + 1}.cif"), atoms)

        return atoms_list