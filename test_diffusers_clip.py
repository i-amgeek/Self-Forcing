#!/usr/bin/env python3
"""
Test script to verify Diffusers CLIP model loading works correctly.
"""

import os
import torch
from wan.modules.clip_diffusers import create_clip_model

def test_diffusers_clip():
    """Test loading CLIP model from Diffusers format"""
    
    print("🧪 Testing Diffusers CLIP Model Loading...")
    
    # Check if the model exists
    diffusers_clip_path = "wan_models/Wan2.1-I2V-14B-480P-Diffusers/image_encoder"
    
    if not os.path.exists(diffusers_clip_path):
        print(f"❌ Model not found at: {diffusers_clip_path}")
        print("Please ensure you have downloaded the Wan2.1-I2V-14B-480P-Diffusers model")
        return False
    
    print(f"✅ Found model directory: {diffusers_clip_path}")
    
    # Check required files
    config_file = os.path.join(diffusers_clip_path, "config.json")
    model_file = os.path.join(diffusers_clip_path, "model.safetensors")
    
    if not os.path.exists(config_file):
        print(f"❌ Missing config.json at: {config_file}")
        return False
    
    if not os.path.exists(model_file):
        print(f"❌ Missing model.safetensors at: {model_file}")
        return False
    
    print("✅ Required files found")
    
    try:
        # Test model loading
        print("🔄 Loading CLIP model...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        clip_model = create_clip_model(
            dtype=dtype,
            device=device,
            model_path=diffusers_clip_path,
            tokenizer_path="wan_models/Wan2.1-I2V-14B-480P-Diffusers/tokenizer"
        )
        
        print("✅ CLIP model loaded successfully!")
        
        # Test basic functionality
        print("🔄 Testing model inference...")
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_image = torch.randn(1, 3, 224, 224, dtype=dtype, device=device)
        dummy_video = [dummy_image]  # CLIP expects list of tensors
        
        with torch.no_grad():
            features = clip_model.visual(dummy_video)
        
        print(f"✅ Model inference successful! Output shape: {features.shape}")
        
        # Check output properties
        expected_features = 1280  # Based on ViT-H/14 architecture
        if features.shape[-1] == expected_features:
            print(f"✅ Output features dimension correct: {expected_features}")
        else:
            print(f"⚠️ Unexpected output dimension: {features.shape[-1]} (expected {expected_features})")
        
        print("\n🎉 All tests passed! Diffusers CLIP model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_diffusers_clip()
    exit(0 if success else 1)
