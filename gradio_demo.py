"""
Gradio Demo for Self-Forcing with I2V support and public sharing.
"""

import os
import re
import random
import time
import argparse
import hashlib
import subprocess
import urllib.request
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import gradio as gr

from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.utils import generate_timestamp
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from wan.modules.clip import CLIPModel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true', help='Create public Gradio link')
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt')
parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml')
parser.add_argument('--trt', action='store_true')
args = parser.parse_args()

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

# Load models
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

text_encoder = WanTextEncoder()

# Initialize CLIP model for I2V support
clip_model = None

# Global variables
current_vae_decoder = None
current_use_taehv = False
fp8_applied = False
torch_compile_applied = False
models_compiled = False

def initialize_clip_model():
    """Initialize CLIP model for I2V support"""
    global clip_model
    
    if clip_model is None:
        print("🔧 Initializing CLIP model for I2V support...")
        try:
            clip_checkpoint_path = "wan_models/Wan2.1-T2V-1.3B/clip_l14_336.pth"
            clip_tokenizer_path = "wan_models/Wan2.1-T2V-1.3B"
            
            if not os.path.exists(clip_checkpoint_path):
                print(f"⚠️ CLIP checkpoint not found at {clip_checkpoint_path}")
                return None
                
            clip_model = CLIPModel(
                dtype=torch.float16,
                device=gpu,
                checkpoint_path=clip_checkpoint_path,
                tokenizer_path=clip_tokenizer_path
            )
            clip_model.eval()
            clip_model.requires_grad_(False)
            
            if low_memory:
                DynamicSwapInstaller.install_model(clip_model, device=gpu)
            else:
                clip_model.to(gpu)
                
            print("✅ CLIP model initialized successfully")
            return clip_model
            
        except Exception as e:
            print(f"❌ Failed to initialize CLIP model: {e}")
            return None
    
    return clip_model

def initialize_vae_decoder(use_taehv=False, use_trt=False):
    """Initialize VAE decoder based on the selected option"""
    global current_vae_decoder, current_use_taehv

    if use_trt:
        from demo_utils.vae import VAETRTWrapper
        current_vae_decoder = VAETRTWrapper()
        return current_vae_decoder

    if use_taehv:
        from demo_utils.taehv import TAEHV
        taehv_checkpoint_path = "checkpoints/taew2_1.pth"
        if not os.path.exists(taehv_checkpoint_path):
            print(f"Downloading taew2_1.pth...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class TAEHVDiffusersWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float16
                self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
                self.config = DotDict(scaling_factor=1.0)

            def decode(self, latents, return_dict=None):
                return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

        current_vae_decoder = TAEHVDiffusersWrapper()
    else:
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)

    current_vae_decoder.eval()
    current_vae_decoder.to(dtype=torch.float16)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.to(gpu)
    current_use_taehv = use_taehv

    return current_vae_decoder

# Initialize models
vae_decoder = initialize_vae_decoder(use_taehv=False, use_trt=args.trt)

transformer = WanDiffusionWrapper(is_causal=True)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
transformer.load_state_dict(state_dict['generator_ema'])

text_encoder.eval()
transformer.eval()

transformer.to(dtype=torch.float16)
text_encoder.to(dtype=torch.bfloat16)

text_encoder.requires_grad_(False)
transformer.requires_grad_(False)

pipeline = CausalInferencePipeline(
    config,
    device=gpu,
    generator=transformer,
    text_encoder=text_encoder,
    vae=vae_decoder
)

if low_memory:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
transformer.to(gpu)

def process_uploaded_image(image):
    """Process uploaded image for I2V generation"""
    if image is None:
        return None
    
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            return None
        
        # Define transform for I2V
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        image_tensor = transform(image)
        return image_tensor, image
        
    except Exception as e:
        print(f"❌ Error processing uploaded image: {e}")
        return None

def calculate_sha256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()

def generate_mp4_from_images(image_directory, output_video_path, fps=24):
    """Generate MP4 video from images"""
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite existing files
        '-framerate', str(fps),
        '-i', os.path.join(image_directory, '%03d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return False

@torch.no_grad()
def generate_video(prompt, image=None, seed=-1, enable_torch_compile=False, enable_fp8=False, use_taehv=False, fps=6, progress=gr.Progress()):
    """Generate video with progress tracking"""
    global current_vae_decoder, current_use_taehv, fp8_applied, torch_compile_applied, models_compiled, clip_model
    
    try:
        # Determine mode
        mode = 'i2v' if image is not None else 't2v'
        
        # Validate inputs
        if not prompt.strip():
            raise ValueError("Prompt is required")
        
        if mode == 'i2v':
            if clip_model is None:
                clip_model = initialize_clip_model()
                if clip_model is None:
                    raise ValueError("CLIP model not available for I2V mode")
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**32)
        
        # Create output directory
        words_up_to_punctuation = re.split(r'[^\w\s]', prompt)[0].strip()[:20]
        sha256_hash = calculate_sha256(prompt)[:10]
        anim_name = f"{mode.upper()}_{words_up_to_punctuation}_{seed}_{sha256_hash}"
        
        output_dir = f"./gradio_outputs/{anim_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        progress(0.05, desc="Initializing generation...")
        
        # Handle VAE decoder switching
        if use_taehv != current_use_taehv:
            progress(0.1, desc="Switching VAE decoder...")
            current_vae_decoder = initialize_vae_decoder(use_taehv=use_taehv)
            pipeline.vae = current_vae_decoder
        
        # Handle FP8 quantization
        if enable_fp8 and not fp8_applied:
            progress(0.15, desc="Applying FP8 quantization...")
            from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
            quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
            fp8_applied = True
        
        # Text encoding
        progress(0.2, desc="Encoding text prompt...")
        conditional_dict = text_encoder(text_prompts=[prompt])
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.float16)
        
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
        
        # Handle torch.compile
        if enable_torch_compile and not models_compiled:
            progress(0.25, desc="Compiling models (first time only)...")
            transformer.compile(mode="max-autotune-no-cudagraphs")
            if not current_use_taehv and not low_memory and not args.trt:
                current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
            models_compiled = True
        
        # Initialize generation
        progress(0.3, desc="Initializing generation...")
        rnd = torch.Generator(gpu).manual_seed(seed)
        
        pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)
        
        # Handle I2V vs T2V initialization
        if mode == 'i2v' and image is not None:
            progress(0.35, desc="Processing input image for I2V...")
            
            # Process the uploaded image
            image_result = process_uploaded_image(image)
            if image_result is None:
                raise ValueError("Failed to process uploaded image")
            
            image_tensor, _ = image_result
            image_tensor = image_tensor.to(device=gpu, dtype=torch.float16)
            
            # Extract CLIP features
            if clip_model is not None:
                clip_model.to(gpu)
                clip_image = image_tensor.unsqueeze(0)
                clip_features = clip_model.visual([clip_image])
                if low_memory:
                    clip_model.cpu()
        
        # Initialize noise
        noise = torch.randn([1, 21, 16, 60, 104], device=gpu, dtype=torch.float16, generator=rnd)
        
        # Generation parameters
        num_blocks = 7
        current_start_frame = 0
        num_input_frames = 0
        all_num_frames = [pipeline.num_frame_per_block] * num_blocks
        
        if current_use_taehv:
            vae_cache = None
        else:
            vae_cache = ZERO_VAE_CACHE
            for i in range(len(vae_cache)):
                vae_cache[i] = vae_cache[i].to(device=gpu, dtype=torch.float16)
        
        total_frames_generated = 0
        all_frames = []
        
        # Generation loop
        for idx, current_num_frames in enumerate(all_num_frames):
            block_progress = 0.4 + (idx / len(all_num_frames)) * 0.5
            progress(block_progress, desc=f"Generating block {idx+1}/{len(all_num_frames)}...")
            
            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            
            # Denoising loop
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                timestep = torch.ones([1, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                
                if index < len(pipeline.denoising_step_list) - 1:
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([1 * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )
            
            # Update KV cache for next block
            if idx != len(all_num_frames) - 1:
                transformer(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros_like(timestep),
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
            
            # Decode to pixels
            decode_progress = block_progress + 0.02
            progress(decode_progress, desc=f"Decoding block {idx+1} to pixels...")
            
            if args.trt:
                all_current_pixels = []
                for i in range(denoised_pred.shape[1]):
                    is_first_frame = torch.tensor(1.0).cuda().half() if idx == 0 and i == 0 else torch.tensor(0.0).cuda().half()
                    outputs = vae_decoder.forward(denoised_pred[:, i:i + 1, :, :, :].half(), is_first_frame, *vae_cache)
                    current_pixels, vae_cache = outputs[0], outputs[1:]
                    all_current_pixels.append(current_pixels.clone())
                pixels = torch.cat(all_current_pixels, dim=1)
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]
            else:
                if current_use_taehv:
                    if vae_cache is None:
                        vae_cache = denoised_pred
                    else:
                        denoised_pred = torch.cat([vae_cache, denoised_pred], dim=1)
                        vae_cache = denoised_pred[:, -3:, :, :, :]
                    pixels = current_vae_decoder.decode(denoised_pred)
                    if idx == 0:
                        pixels = pixels[:, 3:, :, :, :]
                    else:
                        pixels = pixels[:, 12:, :, :, :]
                else:
                    pixels, vae_cache = current_vae_decoder(denoised_pred.half(), *vae_cache)
                    if idx == 0:
                        pixels = pixels[:, 3:, :, :, :]
            
            # Save frames
            block_frames = pixels.shape[1]
            for frame_idx in range(block_frames):
                frame = torch.clamp(pixels[0, frame_idx].float(), -1., 1.) * 127.5 + 127.5
                frame = frame.to(torch.uint8).cpu().numpy()
                frame = np.transpose(frame, (1, 2, 0))
                
                frame_image = Image.fromarray(frame, 'RGB')
                frame_path = os.path.join(output_dir, f"{total_frames_generated:03d}.jpg")
                frame_image.save(frame_path)
                all_frames.append(frame_image)
                total_frames_generated += 1
            
            current_start_frame += current_num_frames
        
        # Create video
        progress(0.95, desc="Creating video file...")
        video_path = os.path.join(output_dir, f"{anim_name}.mp4")
        success = generate_mp4_from_images(output_dir, video_path, fps)
        
        progress(1.0, desc="Generation complete!")
        
        if success and os.path.exists(video_path):
            return video_path, f"✅ Generated {total_frames_generated} frames successfully!"
        else:
            return None, f"❌ Video creation failed, but {total_frames_generated} frames were generated"
    
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Self-Forcing: Text-to-Video & Image-to-Video Generation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Self-Forcing: Advanced Video Generation
        
        Generate high-quality videos from text prompts or animate images with AI.
        
        **Features:**
        - 📝 **Text-to-Video**: Create videos from descriptive text prompts
        - 🖼️ **Image-to-Video**: Animate uploaded images with text guidance
        - ⚡ **Optimizations**: torch.compile, FP8 quantization, TAEHV VAE
        """)
        
        with gr.Tab("🎬 Video Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="📝 Prompt",
                        placeholder="Describe the video you want to generate...",
                        lines=3,
                        value="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage."
                    )
                    
                    image_input = gr.Image(
                        label="🖼️ Upload Image (Optional - for Image-to-Video)",
                        type="pil",
                        height=300
                    )
                    
                    with gr.Row():
                        seed = gr.Number(
                            label="🎲 Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                        fps = gr.Slider(
                            label="🎞️ FPS",
                            minimum=2,
                            maximum=16,
                            value=6,
                            step=0.5
                        )
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        enable_torch_compile = gr.Checkbox(
                            label="🔥 Enable torch.compile (faster after first run)",
                            value=False
                        )
                        enable_fp8 = gr.Checkbox(
                            label="⚡ Enable FP8 Quantization (lower memory)",
                            value=False
                        )
                        use_taehv = gr.Checkbox(
                            label="🎨 Use TAEHV VAE (higher quality)",
                            value=False
                        )
                    
                    generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="📹 Generated Video",
                        height=400
                    )
                    status_output = gr.Textbox(
                        label="📊 Status",
                        lines=2,
                        interactive=False
                    )
        
        with gr.Tab("📖 Usage Guide"):
            gr.Markdown("""
            ## 🎯 How to Use
            
            ### Text-to-Video (T2V)
            1. Enter a detailed prompt describing your desired video
            2. Adjust settings if needed
            3. Click "Generate Video"
            
            ### Image-to-Video (I2V)
            1. Upload an image
            2. Enter a prompt describing how the image should animate
            3. Click "Generate Video"
            
            ## 💡 Tips for Better Results
            
            ### Text Prompts
            - Use detailed, descriptive language
            - Include camera movements, lighting, and style
            - Example: "A white cat walking through a garden, cinematic lighting, shallow depth of field"
            
            ### Image Animation Prompts
            - Describe the motion you want to see
            - Example: "The person starts walking forward slowly"
            - Example: "The leaves begin to sway gently in the wind"
            
            ## ⚙️ Settings Explained
            
            - **torch.compile**: Speeds up generation after first use (one-time compilation)
            - **FP8 Quantization**: Reduces memory usage with minimal quality loss
            - **TAEHV VAE**: Higher quality decoder (requires more memory)
            - **FPS**: Frame rate of output video (higher = smoother but longer generation)
            
            ## 🔧 Requirements
            
            - CUDA-compatible GPU with sufficient VRAM
            - For I2V: CLIP model in `wan_models/Wan2.1-T2V-1.3B/`
            """)
        
        # Event handlers
        generate_btn.click(
            fn=generate_video,
            inputs=[prompt, image_input, seed, enable_torch_compile, enable_fp8, use_taehv, fps],
            outputs=[video_output, status_output],
            show_progress=True
        )
        
        # Example prompts
        examples = [
            ["A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.", None, 42, False, False, False, 6],
            ["A white and orange tabby cat darting through a dense garden, cinematic with warm tones.", None, 123, False, False, False, 6],
            ["Ocean waves crashing against rocky cliffs at sunset, dramatic lighting.", None, 456, False, False, False, 6],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[prompt, image_input, seed, enable_torch_compile, enable_fp8, use_taehv, fps],
            outputs=[video_output, status_output],
            fn=generate_video,
            cache_examples=False
        )
    
    return demo

if __name__ == "__main__":
    # Create output directory
    os.makedirs("gradio_outputs", exist_ok=True)
    
    # Create and launch interface
    demo = create_interface()
    
    print(f"🚀 Starting Gradio demo...")
    if args.share:
        print("🌐 Creating public link...")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        show_tips=True
    )
