import torch
from torch_geometric.nn import DimeNetPlusPlus


def build_dimenet_plus_plus(device='cuda'):
    """
    构建用于预测吸附能 Delta E_H 的 DimeNet++ 模型
    """
    model = DimeNetPlusPlus(
        hidden_channels=128,  # 隐藏层维度
        out_channels=1,  # 输出维度：1（预测单一标量能量 Delta E_H）
        num_blocks=4,  # 交互模块的数量
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,  # 球面谐波频率数
        num_radial=6,  # 径向基函数频率数
        cutoff=5.0,  # 截断半径（需与你 dataset 中的 5.0 埃保持一致）
        max_num_neighbors=32,  # 最大邻居数
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3
    ).to(device)

    return model


# 测试前向传播
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_dimenet_plus_plus(device)

    # 模拟我们在 dataset 中构建的一个 batch 的数据
    num_atoms = 10
    z = torch.randint(1, 100, (num_atoms,)).to(device)  # 原子序数
    pos = torch.randn(num_atoms, 3).to(device)  # 三维坐标
    batch = torch.zeros(num_atoms, dtype=torch.long).to(device)  # Batch 索引

    # DimeNet++ 只需要 z, pos 和 batch 就能自动在内部构建距离图并计算能量
    energy_prediction = model(z, pos, batch)
    print(f"Predicted Delta E_H: {energy_prediction.item():.4f} eV")