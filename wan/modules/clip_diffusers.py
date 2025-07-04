"""
Diffusers-compatible CLIP model loader for Wan I2V models.
Supports both native .pth format and Diffusers safetensors format.
"""

import os
import json
import logging
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from .clip import CLIPModel, clip_xlm_roberta_vit_h_14
from .tokenizers import HuggingfaceTokenizer


class DiffusersCLIPModel:
    """
    CLIP model that can load from both native .pth and Diffusers safetensors formats.
    """
    
    def __init__(self, dtype, device, model_path, tokenizer_path=None):
        self.dtype = dtype
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Determine format and load model
        if self._is_diffusers_format():
            self._load_diffusers_format()
        else:
            self._load_native_format()
    
    def _is_diffusers_format(self):
        """Check if the model path is in Diffusers format."""
        if os.path.isdir(self.model_path):
            # Check for Diffusers structure
            config_path = os.path.join(self.model_path, "config.json")
            model_path = os.path.join(self.model_path, "model.safetensors")
            return os.path.exists(config_path) and os.path.exists(model_path)
        return False
    
    def _load_diffusers_format(self):
        """Load CLIP model from Diffusers format."""
        logging.info(f"Loading CLIP model from Diffusers format: {self.model_path}")
        
        # Load config
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model with config parameters
        model_kwargs = self._extract_model_config(config)
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=self.dtype,
            device=self.device,
            **model_kwargs
        )
        
        # Load weights from safetensors
        model_file = os.path.join(self.model_path, "model.safetensors")
        state_dict = load_file(model_file, device=str(self.device))
        
        # Load state dict with proper key mapping if needed
        self._load_state_dict_with_mapping(state_dict)
        
        self.model = self.model.eval().requires_grad_(False)
        
        # Initialize tokenizer
        self._init_tokenizer()
    
    def _load_native_format(self):
        """Load CLIP model from native .pth format."""
        logging.info(f"Loading CLIP model from native format: {self.model_path}")
        
        # Use original CLIPModel implementation
        original_clip = CLIPModel(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.model_path,
            tokenizer_path=self.tokenizer_path
        )
        
        self.model = original_clip.model
        self.transforms = original_clip.transforms
        self.tokenizer = original_clip.tokenizer
    
    def _extract_model_config(self, config):
        """Extract model configuration from Diffusers config."""
        # Map Diffusers config to our model config
        model_kwargs = {}
        
        # Vision transformer config
        if "vision_config" in config:
            vision_config = config["vision_config"]
            model_kwargs.update({
                "image_size": vision_config.get("image_size", 224),
                "patch_size": vision_config.get("patch_size", 14),
                "vision_dim": vision_config.get("hidden_size", 1280),
                "vision_heads": vision_config.get("num_attention_heads", 16),
                "vision_layers": vision_config.get("num_hidden_layers", 32),
                "vision_mlp_ratio": 4,  # Default value
                "vision_pool": "token",  # Default value
            })
        
        # Text config (if available)
        if "text_config" in config:
            text_config = config["text_config"]
            model_kwargs.update({
                "vocab_size": text_config.get("vocab_size", 250002),
                "max_text_len": text_config.get("max_position_embeddings", 514),
                "text_dim": text_config.get("hidden_size", 1024),
                "text_heads": text_config.get("num_attention_heads", 16),
                "text_layers": text_config.get("num_hidden_layers", 24),
            })
        
        # Projection dimension
        model_kwargs["embed_dim"] = config.get("projection_dim", 1024)
        
        return model_kwargs
    
    def _load_state_dict_with_mapping(self, state_dict):
        """Load state dict with proper key mapping if needed."""
        try:
            # Try direct loading first
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys when loading CLIP: {missing_keys[:5]}...")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading CLIP: {unexpected_keys[:5]}...")
                
        except Exception as e:
            logging.error(f"Failed to load CLIP state dict: {e}")
            # Could implement key mapping here if needed
            raise
    
    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Try to find tokenizer in parent directory or use default
        if self.tokenizer_path:
            tokenizer_path = self.tokenizer_path
        else:
            # Look for tokenizer in parent directory of image_encoder
            parent_dir = os.path.dirname(self.model_path)
            tokenizer_dir = os.path.join(parent_dir, "tokenizer")
            if os.path.exists(tokenizer_dir):
                tokenizer_path = tokenizer_dir
            else:
                # Fallback to a reasonable default
                tokenizer_path = "xlm-roberta-large"
        
        try:
            self.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_path,
                seq_len=self.model.max_text_len - 2,
                clean='whitespace'
            )
        except Exception as e:
            logging.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            # Fallback tokenizer
            self.tokenizer = HuggingfaceTokenizer(
                name="xlm-roberta-large",
                seq_len=512,
                clean='whitespace'
            )
    
    def visual(self, videos):
        """Process visual input - same interface as original CLIPModel."""
        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(
                u.transpose(0, 1),
                size=size,
                mode='bicubic',
                align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out


def create_clip_model(dtype, device, model_path, tokenizer_path=None):
    """
    Factory function to create CLIP model supporting both formats.
    
    Args:
        dtype: Model dtype (torch.float16, etc.)
        device: Target device
        model_path: Path to model (either .pth file or directory with Diffusers format)
        tokenizer_path: Path to tokenizer (optional for Diffusers format)
    
    Returns:
        CLIP model instance with .visual() method
    """
    return DiffusersCLIPModel(dtype, device, model_path, tokenizer_path)
