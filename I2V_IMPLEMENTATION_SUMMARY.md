# Image-to-Video (I2V) Implementation Summary

## Overview

This repository now supports **Image-to-Video (I2V) generation** in addition to the existing Text-to-Video (T2V) functionality. The I2V feature allows users to upload an image and generate a video showing how that image should animate based on a text prompt.

## ✅ Confirmed: Repository Supports I2V Generation

Based on analysis of `inference.py` and related files, this repository **DOES support image-to-video generation** with the following evidence:

### Original I2V Support in inference.py
- `--i2v` command line argument for I2V mode
- `TextImagePairDataset` for handling image+text pairs
- Image preprocessing with transforms (resize to 480x832, normalize)
- Initial latent encoding from input image
- Conditional generation pipeline that uses the image as first frame

### Enhanced Demo Interface
The demo interface has been enhanced with full I2V support including:

## 🚀 New Features Added

### 1. **Mode Selection Interface**
- Radio buttons to switch between Text-to-Video and Image-to-Video modes
- Dynamic UI updates based on selected mode
- Real-time mode switching with backend synchronization

### 2. **Image Upload System**
- **Drag & Drop Support**: Users can drag images directly onto the upload area
- **Click to Upload**: Traditional file picker interface
- **Image Preview**: Shows uploaded image with file information
- **Format Validation**: Supports JPG, PNG, WebP formats
- **Size Validation**: Maximum 10MB file size limit
- **Image Processing**: Automatic resize to 480x832 and normalization

### 3. **Backend I2V Pipeline**
- **CLIP Model Integration**: For image feature extraction
- **Image Preprocessing**: Transforms images to model requirements
- **Conditional Generation**: Uses uploaded image as conditioning
- **Memory Management**: Dynamic model loading for low-memory systems
- **Error Handling**: Comprehensive validation and error reporting

### 4. **Socket.IO Events**
- `set_mode`: Switch between T2V and I2V modes
- `upload_image`: Handle image upload and processing
- `image_uploaded`: Confirm successful image upload
- `mode_changed`: Notify frontend of mode changes

## 🔧 Technical Implementation

### Frontend (templates/demo.html)
```javascript
// Mode switching
function switchMode(mode) {
    currentMode = mode;
    // Update UI elements
    // Notify backend
    socket.emit('set_mode', { mode: mode });
}

// Image upload handling
function processImageFile(file) {
    // Validate file type and size
    // Convert to base64
    // Send to backend
    socket.emit('upload_image', { image_data: imageData });
}
```

### Backend (demo.py)
```python
# CLIP model initialization
def initialize_clip_model():
    clip_model = CLIPModel(
        dtype=torch.float16,
        device=gpu,
        checkpoint_path="wan_models/Wan2.1-T2V-1.3B/clip_l14_336.pth",
        tokenizer_path="wan_models/Wan2.1-T2V-1.3B"
    )
    return clip_model

# Image processing
def process_uploaded_image(image_data):
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)

# I2V generation
if mode == 'i2v' and uploaded_image is not None:
    # Process image for conditioning
    # Extract CLIP features
    # Initialize generation with image
```

## 📁 Files Modified

### Core Files
- **`demo.py`**: Added I2V backend logic, CLIP integration, image processing
- **`templates/demo.html`**: Added I2V UI components, JavaScript handlers
- **`test_i2v_integration.py`**: Comprehensive test suite for I2V features

### Key Functions Added
- `initialize_clip_model()`: CLIP model setup for I2V
- `process_uploaded_image()`: Image preprocessing pipeline
- `handle_upload_image()`: Socket.IO image upload handler
- `handle_set_mode()`: Mode switching handler
- `switchMode()`: Frontend mode switching
- `processImageFile()`: Frontend image processing

## 🎯 Usage Instructions

### 1. **Setup Requirements**
```bash
# Ensure CLIP model is available
# Expected path: wan_models/Wan2.1-T2V-1.3B/clip_l14_336.pth
```

### 2. **Start the Demo**
```bash
python demo.py --port 5001
```

### 3. **Use I2V Generation**
1. Open the web interface
2. Select "🖼️ Image-to-Video" mode
3. Upload an image (drag & drop or click)
4. Enter animation prompt (e.g., "The person starts walking forward")
5. Configure generation settings
6. Click "🚀 Start I2V Generation"

### 4. **Example Prompts for I2V**
- "The person starts walking forward slowly"
- "The leaves begin to sway gently in the wind"
- "The water starts flowing and creating ripples"
- "The character turns their head and smiles"

## 🔍 Command Line I2V Usage

The original `inference.py` also supports I2V via command line:

```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path path/to/image_text_pairs.txt \
    --output_folder outputs/ \
    --i2v \
    --num_output_frames 21 \
    --seed 42
```

## 🧪 Testing

Run the integration test to verify I2V functionality:

```bash
python test_i2v_integration.py
```

Expected output:
```
🎉 All tests passed! I2V integration is ready.
```

## 📋 Feature Comparison

| Feature | T2V | I2V |
|---------|-----|-----|
| Input | Text prompt only | Image + text prompt |
| Output | 21-frame video | 21-frame video |
| Resolution | 480x832 | 480x832 |
| Conditioning | Text embeddings | Image + text embeddings |
| CLIP Model | Optional | Required |
| Use Cases | Creative generation | Animation from stills |

## 🚨 Requirements & Dependencies

### Model Requirements
- **CLIP Model**: `wan_models/Wan2.1-T2V-1.3B/clip_l14_336.pth`
- **VAE Decoder**: Standard or TAEHV variant
- **Transformer**: Self-forcing diffusion model

### Python Dependencies
- `torch`, `torchvision`
- `PIL` (Pillow)
- `flask`, `flask-socketio`
- `omegaconf`
- `numpy`

## 🎉 Conclusion

The Self-Forcing repository now provides comprehensive Image-to-Video generation capabilities through both:

1. **Web Interface**: User-friendly demo with drag & drop image upload
2. **Command Line**: Batch processing via `inference.py --i2v`

The implementation maintains compatibility with existing T2V functionality while adding robust I2V features with proper error handling, validation, and user experience considerations.
