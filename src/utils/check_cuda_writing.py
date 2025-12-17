from pathlib import Path
import torch

def check_h100():
    try:
        x = torch.randn(1, device='cuda')
    except Exception as e:
        print(e)
        return False
    
    print("✓ Writing a cuda tensor works")
    return True

if __name__ == "__main__":
    cuda_available = check_h100()
    if not cuda_available:
        Path("cuda_not_available").touch()

    import sys
    sys.exit(0 if check_h100() else 1)
