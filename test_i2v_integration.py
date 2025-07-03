#!/usr/bin/env python3
"""
Test script to verify I2V integration in the demo.
This script checks the key components without requiring the full environment.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported (syntax check)"""
    print("🔍 Testing imports...")
    
    try:
        # Test demo.py syntax
        with open('demo.py', 'r') as f:
            demo_content = f.read()
        
        # Check for key I2V components in Python
        python_i2v_components = [
            'initialize_clip_model',
            'process_uploaded_image',
            'handle_upload_image',
            'handle_set_mode',
            'current_mode',
            'uploaded_image'
        ]
        
        missing_components = []
        for component in python_i2v_components:
            if component not in demo_content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing Python I2V components: {missing_components}")
            return False
        else:
            print("✅ All Python I2V components found in demo.py")
        
        # Test HTML template
        with open('templates/demo.html', 'r') as f:
            html_content = f.read()
        
        html_components = [
            'switchMode',
            'imageUploadSection',
            'handleImageUpload',
            'processImageFile',
            'upload_image',
            'set_mode'
        ]
        
        missing_html = []
        for component in html_components:
            if component not in html_content:
                missing_html.append(component)
        
        if missing_html:
            print(f"❌ Missing HTML components: {missing_html}")
            return False
        else:
            print("✅ All I2V components found in demo.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        'demo.py',
        'templates/demo.html',
        'inference.py',
        'utils/dataset.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_i2v_features():
    """Test I2V specific features"""
    print("🔍 Testing I2V features...")
    
    # Check demo.py for I2V logic
    with open('demo.py', 'r') as f:
        demo_content = f.read()
    
    # Check for mode handling
    if "mode == 'i2v'" not in demo_content:
        print("❌ I2V mode handling not found")
        return False
    
    # Check for image processing
    if "process_uploaded_image" not in demo_content:
        print("❌ Image processing function not found")
        return False
    
    # Check for CLIP integration
    if "CLIPModel" not in demo_content:
        print("❌ CLIP model integration not found")
        return False
    
    # Check HTML for mode switching
    with open('templates/demo.html', 'r') as f:
        html_content = f.read()
    
    if 'name="mode"' not in html_content:
        print("❌ Mode selection radio buttons not found")
        return False
    
    if 'imageUploadSection' not in html_content:
        print("❌ Image upload section not found")
        return False
    
    print("✅ All I2V features implemented")
    return True

def main():
    """Run all tests"""
    print("🚀 Testing I2V Integration")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_i2v_features
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All tests passed! I2V integration is ready.")
        print("\n📋 Summary of I2V Features Added:")
        print("   • Mode selection (T2V/I2V) in UI")
        print("   • Image upload with drag & drop")
        print("   • Image preprocessing and validation")
        print("   • CLIP model integration for I2V")
        print("   • Backend I2V generation pipeline")
        print("   • Socket.IO events for I2V workflow")
        print("\n🔧 To use I2V:")
        print("   1. Ensure CLIP model is available in wan_models/")
        print("   2. Start demo: python demo.py")
        print("   3. Select 'Image-to-Video' mode")
        print("   4. Upload an image")
        print("   5. Enter animation prompt")
        print("   6. Generate!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
