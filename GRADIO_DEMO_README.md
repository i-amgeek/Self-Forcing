# 🚀 Gradio Demo for Self-Forcing

A user-friendly Gradio interface for Self-Forcing video generation with **public sharing support**.

## ✨ Features

- 🌐 **Public Sharing**: Create shareable links with `--share` flag
- 📝 **Text-to-Video**: Generate videos from text prompts
- 🖼️ **Image-to-Video**: Animate uploaded images
- ⚡ **Advanced Optimizations**: torch.compile, FP8 quantization, TAEHV VAE
- 📊 **Progress Tracking**: Real-time generation progress
- 📱 **Mobile Friendly**: Responsive design
- 🎨 **Example Gallery**: Pre-loaded example prompts

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install gradio>=3.50.0
```

### 2. Launch Demo

#### Local Access Only
```bash
python gradio_demo.py
```

#### Public Sharing (Creates shareable link)
```bash
python gradio_demo.py --share
```

#### Custom Port
```bash
python gradio_demo.py --port 8080 --share
```

## 🎯 Usage

### Text-to-Video (T2V)
1. Enter a descriptive prompt
2. Leave image field empty
3. Click "🚀 Generate Video"

### Image-to-Video (I2V)
1. Upload an image
2. Enter animation prompt (e.g., "The person starts walking")
3. Click "🚀 Generate Video"

## ⚙️ Advanced Settings

- **🔥 torch.compile**: Faster generation after first compilation
- **⚡ FP8 Quantization**: Lower memory usage
- **🎨 TAEHV VAE**: Higher quality decoder
- **🎞️ FPS**: Output video frame rate

## 📝 Example Prompts

### Text-to-Video
- "A stylish woman walks down a Tokyo street filled with warm glowing neon"
- "A white cat walking through a garden, cinematic lighting"
- "Ocean waves crashing against rocky cliffs at sunset"

### Image-to-Video
- "The person starts walking forward slowly"
- "The leaves begin to sway gently in the wind"
- "The water starts flowing and creating ripples"

## 🔧 Command Line Options

```bash
python gradio_demo.py [OPTIONS]

Options:
  --share              Create public Gradio link
  --port INTEGER       Port number (default: 7860)
  --host TEXT          Host address (default: 0.0.0.0)
  --checkpoint_path    Path to model checkpoint
  --config_path        Path to config file
  --trt               Use TensorRT optimization
```

## 🌐 Public Sharing

When using `--share`, Gradio creates a temporary public URL that:
- ✅ Works from anywhere on the internet
- ✅ Automatically expires after 72 hours
- ✅ Requires no additional setup
- ⚠️ Should only be used for demos/testing

Example public URL: `https://abc123.gradio.live`

## 📊 Output

- **Videos**: Saved to `gradio_outputs/` directory
- **Format**: MP4 with H.264 encoding
- **Resolution**: 480x832 pixels
- **Duration**: ~3.5 seconds (21 frames at 6 FPS)

## 🔍 Troubleshooting

### CLIP Model Not Found
```bash
# Ensure CLIP model exists at:
wan_models/Wan2.1-T2V-1.3B/clip_l14_336.pth
```

### Memory Issues
- Enable FP8 quantization
- Use lower FPS settings
- Ensure sufficient GPU VRAM

### FFmpeg Not Found
```bash
# Install FFmpeg for video creation
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

## 🆚 vs Flask Demo

| Feature | Gradio Demo | Flask Demo |
|---------|-------------|------------|
| Public Sharing | ✅ `--share` | ❌ Local only |
| Setup | Simple | Medium |
| Real-time Streaming | ❌ | ✅ |
| Progress Tracking | ✅ Progress bar | ✅ Live updates |
| Best For | Public demos | Development |

## 🎉 Tips for Best Results

1. **Detailed Prompts**: Use descriptive language with camera movements and lighting
2. **I2V Animation**: Focus on specific motions rather than scene changes
3. **Seed Control**: Use same seed for reproducible results
4. **Memory Management**: Enable optimizations for lower-end GPUs

## 📞 Support

For issues or questions:
1. Check the main repository documentation
2. Verify model files are properly downloaded
3. Ensure GPU drivers and CUDA are up to date
