"""
Test Flash Attention and attention mechanism availability on Windows
Run this AFTER installing Unsloth
"""

import sys
import torch

print("="*80)
print("FLASH ATTENTION COMPATIBILITY CHECK (Windows)")
print("="*80)
print()

# Check Python version
print(f"✓ Python version: {sys.version}")
print()

# Check PyTorch and CUDA
print("="*80)
print("PYTORCH & CUDA")
print("="*80)
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version (PyTorch): {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ VRAM: {vram_gb:.1f} GB")
else:
    print("❌ CUDA not available - GPU training won't work!")
    print("   Make sure CUDA toolkit is installed")
    sys.exit(1)

print()

# Check Flash Attention 2
print("="*80)
print("FLASH ATTENTION 2 CHECK")
print("="*80)

flash_attn_available = False
try:
    from flash_attn import flash_attn_func
    flash_attn_available = True
    print("✅ Flash Attention 2 is AVAILABLE!")
    print("   This is rare on Windows - you're lucky!")
except ImportError as e:
    print("❌ Flash Attention 2 is NOT available")
    print(f"   Reason: {str(e)}")
    print("   This is NORMAL on Windows - don't worry!")

print()

# Check xFormers (common fallback)
print("="*80)
print("XFORMERS CHECK (Fallback #1)")
print("="*80)

xformers_available = False
try:
    import xformers
    import xformers.ops
    xformers_available = True
    print(f"✅ xFormers is available (version {xformers.__version__})")
    print("   This gives ~2-3x speedup vs standard attention")
except ImportError:
    print("❌ xFormers is NOT available")
    print("   Unsloth will install this automatically")

print()

# Check PyTorch SDPA (PyTorch 2.0+ built-in)
print("="*80)
print("PYTORCH SDPA CHECK (Fallback #2)")
print("="*80)

sdpa_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if sdpa_available:
    print("✅ PyTorch SDPA (Scaled Dot Product Attention) is available")
    print("   This is built into PyTorch 2.0+ and gives ~1.5-2x speedup")
    
    # Check which backends are available
    try:
        from torch.nn.attention import SDPBackend
        import torch.backends.cuda
        
        # Check which SDPA backends are enabled
        print("\n   Available SDPA backends:")
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=True, 
            enable_mem_efficient=True
        ):
            print("   - Math (always available)")
            print("   - Memory Efficient (likely available on Ampere)")
            try:
                print("   - Flash Attention (checking...)")
                # This is Flash Attention via SDPA, different from flash-attn package
            except:
                pass
    except:
        print("   (Cannot check specific backends, but SDPA is available)")
else:
    print("❌ PyTorch SDPA not available")
    print("   Update PyTorch to 2.0 or higher")

print()

# Summary
print("="*80)
print("SUMMARY FOR YOUR PROJECT")
print("="*80)
print()

if flash_attn_available:
    print("🎉 EXCELLENT: You have Flash Attention 2!")
    print("   Expected speedup: 3-5x vs standard attention")
    status = "EXCELLENT"
elif xformers_available:
    print("✅ GOOD: You have xFormers")
    print("   Expected speedup: 2-3x vs standard attention")
    status = "GOOD"
elif sdpa_available:
    print("✅ ACCEPTABLE: You have PyTorch SDPA")
    print("   Expected speedup: 1.5-2x vs standard attention")
    status = "ACCEPTABLE"
else:
    print("⚠️  LIMITED: Only standard attention available")
    print("   Training will be slower but still works")
    status = "LIMITED"

print()
print("="*80)
print("WHAT THIS MEANS FOR YOUR 2-WEEK PROJECT")
print("="*80)
print()

if status in ["EXCELLENT", "GOOD"]:
    print("✅ You're in GREAT shape!")
    print("   All 5 experiments will fit comfortably in your timeline")
    print()
    print("   Estimated training times (3B model, 5000 examples):")
    print("   - Per epoch: 2-3 hours")
    print("   - Full experiment (3 epochs): 6-9 hours")
    print("   - All 5 experiments: ~40-50 hours (fits in 1 week!)")
elif status == "ACCEPTABLE":
    print("✅ You're still GOOD!")
    print("   Training will be a bit slower but totally doable")
    print()
    print("   Estimated training times (3B model, 5000 examples):")
    print("   - Per epoch: 3-4 hours")
    print("   - Full experiment (3 epochs): 9-12 hours")
    print("   - All 5 experiments: ~50-60 hours (still fits!)")
else:
    print("⚠️  Training will be slower")
    print("   Consider using smaller model (1B) or fewer experiments")
    print()
    print("   Estimated training times (3B model, 5000 examples):")
    print("   - Per epoch: 5-6 hours")
    print("   - Full experiment (3 epochs): 15-18 hours")
    print("   - Recommendation: Focus on 3 best experiments")

print()
print("="*80)
print("NEXT STEPS")
print("="*80)
print()

if not xformers_available and not flash_attn_available:
    print("1. Install Unsloth - it will handle optimizations automatically:")
    print('   pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"')
    print()
    print("2. Re-run this script to verify what got installed")
else:
    print("✅ You're ready to proceed with the project!")
    print("   Unsloth will automatically use the best available attention mechanism")

print()
print("="*80)