import sys
import os
import torch
from pathlib import Path

# 1. Setup Paths
ROOT_DIR = "/home/yanivgra/dinov3"
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 2. Match the specific imports
try:
    from dinov3.configs import setup_config
    from dinov3.train.ssl_meta_arch import SSLMetaArch
    import dinov3.distributed as distributed  # Import the distributed utility
    print("--- Step 1: Core DINOv3 modules imported ---")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def check_alignment():
    # Mocking arguments for setup_config
    class Args:
        config_file = f"{ROOT_DIR}/dinov3/configs/ssl_default_config.yaml"
        opts = [] 
        output_dir = "/home/yanivgra//dinov3/tmp_check"

    args = Args()
    
    # FIX 1: Create the output directory so the logger doesn't crash
    os.makedirs(args.output_dir, exist_ok=True)

    # FIX 2: "Fool" DINOv3 into thinking we are in a single-GPU distributed setup
    distributed.is_enabled = lambda: True
    distributed.get_world_size = lambda: 1
    distributed.get_rank = lambda: 0
    distributed.get_process_subgroup = lambda: None # DINOv3 needs this for some layers
    
    ckpt_path = "/home/yanivgra/dinov3/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    print("--- Step 2: Loading Config (Bypassing scaling checks) ---")
    try:
        cfg = setup_config(args, strict_cfg=False)
    except Exception as e:
        print(f"Config setup failed: {e}")
        return

    print("--- Step 3: Building SSLMetaArch ---")
    try:
        # Using CPU to avoid needing a GPU for a simple weight check
        with torch.device("cpu"): 
            model = SSLMetaArch(cfg)
        print("Model architecture initialized.")
    except Exception as e:
        print(f"Architecture build failed: {e}")
        return

    print("--- Step 4: Loading Checkpoint with Prefix Mapping ---")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # If the checkpoint is nested (e.g., from a previous DINO run), get the backbone
        if "teacher" in checkpoint:
            raw_weights = checkpoint["teacher"]
        elif "model" in checkpoint:
            raw_weights = checkpoint["model"]
        else:
            raw_weights = checkpoint

        # Remove 'backbone.' prefix if it exists in the checkpoint 
        # (so we start from a clean 'blocks.0...')
        clean_weights = {}
        for k, v in raw_weights.items():
            new_k = k.replace("backbone.", "")
            clean_weights[new_k] = v

        # Now, map these clean weights to BOTH student and teacher in our SSLMetaArch
        mapped_weights = {}
        for k, v in clean_weights.items():
            mapped_weights[f"student.backbone.{k}"] = v
            mapped_weights[f"teacher.backbone.{k}"] = v
            mapped_weights[f"model_ema.backbone.{k}"] = v

        # Load into the model
        msg = msg = model.load_state_dict(mapped_weights, strict=False, assign=True)
        
        print("\n" + "="*30)
        print("ALIGNMENT RESULTS:")
        print(f"Missing Keys: {len(msg.missing_keys)}")
        if msg.missing_keys:
            print("Missing keys list:")
            for k in msg.missing_keys:
                print(f"  - {k}")
        print(f"Unexpected Keys: {len(msg.unexpected_keys)}")
        if msg.unexpected_keys:
            print("Unexpected keys list:")
            for k in msg.unexpected_keys:
                print(f"  - {k}")
        
        # Check if the backbone keys are now handled
        backbone_missing = [k for k in msg.missing_keys if "backbone" in k]
        if len(backbone_missing) == 0:
            print("✔️ SUCCESS: Both Student and Teacher backbones are fully loaded!")
            print(f"Note: {len(msg.missing_keys)} head/loss keys remain uninitialized (this is expected).")
        else:
            print(f"❌ STILL MISSING {len(backbone_missing)} backbone keys. Sample: {backbone_missing[:3]}")
        print("="*30)

    except Exception as e:
        print(f"Mapping failed: {e}")


if __name__ == "__main__":
    check_alignment()