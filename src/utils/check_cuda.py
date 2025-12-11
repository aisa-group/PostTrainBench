#!/usr/bin/env python3
import torch

def check_h100():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {device_count} device(s)")
    
    h100_found = False
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {name} ({props.total_memory / 1e9:.1f} GB)")
        
        if "H100" in name:
            h100_found = True
    
    if h100_found:
        print("✓ H100 detected")
    else:
        print("❌ No H100 found")
    
    return h100_found

if __name__ == "__main__":
    import sys
    sys.exit(0 if check_h100() else 1)