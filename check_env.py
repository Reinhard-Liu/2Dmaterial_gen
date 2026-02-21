import sys
import platform


def print_separator(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


print_separator("System Info")
print(f"Python Version: {sys.version}")
print(f"OS: {platform.system()} {platform.release()}")

try:
    import torch

    print_separator("PyTorch & CUDA")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Current Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
except ImportError:
    print("\n[ERROR] PyTorch is not installed.")

try:
    import torch_geometric

    print_separator("PyTorch Geometric")
    print(f"PyG Version: {torch_geometric.__version__}")
except ImportError:
    print("\n[ERROR] PyTorch Geometric is not installed.")

try:
    import e3nn

    print_separator("e3nn (Equivariant Ops)")
    print(f"e3nn Version: {e3nn.__version__}")
except ImportError:
    print("\n[ERROR] e3nn is not installed.")

try:
    import ase
    from ase.build import graphene_nanoribbon

    print_separator("ASE (Atomic Simulation Environment)")
    print(f"ASE Version: {ase.__version__}")

    # 测试 ASE 结构构建功能
    atoms = graphene_nanoribbon(2, 2, type='zigzag', saturated=True)
    print(f"Test Material Generation: Successfully built a {len(atoms)}-atom 2D graphene nanoribbon.")
except ImportError:
    print("\n[ERROR] ASE is not installed or functioning correctly.")

try:
    import pymatgen.core as pmg

    print_separator("Pymatgen")
    print(f"Pymatgen Version: {pmg.__version__}")
except ImportError:
    print("\n[ERROR] Pymatgen is not installed.")

print_separator("Environment Check Complete")