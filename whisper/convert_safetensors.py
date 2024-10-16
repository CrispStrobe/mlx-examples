import argparse
import copy
import hashlib
import json
import os
import urllib
import warnings
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Any, List

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear, QuantizedLinear
from mlx.utils import tree_flatten
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration

from mlx_whisper.whisper import ModelDimensions, AudioEncoder, TextDecoder, Whisper
import safetensors.torch  # Ensure safetensors is installed

import gc
import re  # Moved import here for quantize_model

# Import for uploading
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, logging as hf_logging

# Initialize global verbose flag
VERBOSE = False

def debug_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def get_dtype(dtype_str):
    dtype_mapping = {
        "float16": (mx.float16, np.float16),
        "float32": (mx.float32, np.float32),
    }
    if dtype_str not in dtype_mapping:
        raise ValueError(f"dtype {dtype_str} not supported. Choose from {list(dtype_mapping.keys())}.")
    return dtype_mapping[dtype_str]

class FloatLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, original_layer=None, dtype=mx.float16):
        super().__init__(normalized_shape, eps=eps)
        self.dtype = dtype
        if original_layer is not None:
            # Cast weight and bias to the desired dtype
            self.weight = original_layer.weight.astype(dtype)
            self.bias = original_layer.bias.astype(dtype)

    def __call__(self, x):
        # Ensure input is cast to the desired dtype
        x = x.astype(self.dtype)
        return super().__call__(x)

class CustomAudioEncoder(AudioEncoder):
    def __init__(self, *args, dtype=mx.float16, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype  # Store the desired dtype

    def __call__(self, x):
        audio_features = super().__call__(x)
        audio_features = audio_features.astype(self.dtype)
        debug_print(f"CustomAudioEncoder output dtype: {audio_features.dtype}")
        return audio_features

class CustomWhisper(nn.Module):

    def __init__(self, dims: ModelDimensions, dtype=mx.float16):
        super().__init__()
        self.dims = dims
        self._dtype = dtype  # Initialize _dtype

        # Initialize encoder with CustomAudioEncoder
        self.encoder = CustomAudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype=dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype=dtype,
        )
        
        # Alignment heads remain unchanged
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    @property
    def dtype(self):
        return self._dtype

    def forward_with_cross_qk(self, mel, tokens):
        # Ensure mel is cast to the desired dtype
        mel = mel.astype(self._dtype)
        audio_features = self.encoder(mel).astype(self._dtype)
        return self.decoder(tokens, audio_features)[0].astype(self._dtype)

    def astype(self, dtype):
        """
        Casts model parameters to the specified dtype.

        Args:
            dtype (mx.Dtype): The target dtype (e.g., mx.float16, mx.float32).
        """
        self._dtype = dtype
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight = module.weight.astype(dtype)
                debug_print(f"Cast nn.Linear layer '{name}.weight' to {dtype}.")
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = module.bias.astype(dtype)
                    debug_print(f"Cast nn.Linear layer '{name}.bias' to {dtype}.")

            elif isinstance(module, QuantizedLinear):
                # QuantizedLinear.weight should remain as uint32, do not cast
                # Only cast bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = module.bias.astype(dtype)
                    debug_print(f"Cast QuantizedLinear layer '{name}.bias' to {dtype}.")

            elif isinstance(module, FloatLayerNorm):
                module.weight = module.weight.astype(dtype)
                debug_print(f"Cast FloatLayerNorm layer '{name}.weight' to {dtype}.")
                module.bias = module.bias.astype(dtype)
                debug_print(f"Cast FloatLayerNorm layer '{name}.bias' to {dtype}.")

            elif isinstance(module, nn.Conv1d):
                module.weight = module.weight.astype(dtype)
                debug_print(f"Cast nn.Conv1d layer '{name}.weight' to {dtype}.")
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = module.bias.astype(dtype)
                    debug_print(f"Cast nn.Conv1d layer '{name}.bias' to {dtype}.")

            elif isinstance(module, nn.Embedding):
                module.weight = module.weight.astype(dtype)
                debug_print(f"Cast nn.Embedding layer '{name}.weight' to {dtype}.")

            # Add additional layer types if necessary

        # Handle special buffers with detailed debug prints
        if hasattr(self.encoder, '_positional_embedding'):
            self.encoder._positional_embedding = self.encoder._positional_embedding.astype(dtype)
            debug_print(f"encoder._positional_embedding dtype after cast: {self.encoder._positional_embedding.dtype}")

        if hasattr(self.decoder, 'positional_embedding'):
            self.decoder.positional_embedding = self.decoder.positional_embedding.astype(dtype)
            debug_print(f"decoder.positional_embedding dtype after cast: {self.decoder.positional_embedding.dtype}")

        if hasattr(self.decoder, '_mask'):
            self.decoder._mask = self.decoder._mask.astype(dtype)
            debug_print(f"decoder._mask dtype after cast: {self.decoder._mask.dtype}")

        return self


    def embed_audio(self, mel):
        debug_print(f"Input dtype: {mel.dtype}")
        audio_features = self.encoder(mel)
        debug_print(f"After encoder, dtype: {audio_features.dtype}")
        # Cast audio_features to the desired dtype
        audio_features = audio_features.astype(self._dtype)
        debug_print(f"After casting, dtype: {audio_features.dtype}, type: {type(audio_features)}")
        return audio_features


    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk_old(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))

    def decode(self, *args, **kwargs):
        return self.decoder.decode(*args, **kwargs)

    def detect_language(self, *args, **kwargs):
        return self.decoder.detect_language(*args, **kwargs)

    def update(self, params):
        # Handle specific parameter transformations if needed
        if 'decoder.token_embedding.weight' in params:
            weight = params['decoder.token_embedding.weight']
            debug_print("decoder.token_embedding.weight shape:", weight.shape)

            n_vocab, n_state = weight.shape
            if (n_vocab, n_state) != (self.dims.n_vocab, self.dims.n_text_state):
                debug_print(f"Transposing token embedding weights from shape {(n_vocab, n_state)} to {(self.dims.n_vocab, self.dims.n_text_state)}")
                weight = weight.T
                params['decoder.token_embedding.weight'] = weight

        # Iterate over each parameter in the mapped state dictionary
        for param_name, param_value in tqdm(params.items(), desc="Assigning parameters", leave=False):
            
            if 'conv' in param_name and 'weight' in param_name:
                # Do NOT transpose the weights for convolution layers
                debug_print(f"Conv weight {param_name}: shape {param_value.shape}")

            parts = param_name.split('.')
            module = self  # Start from the root module (self)

            # Navigate through the module hierarchy based on parameter parts
            try:
                for part in parts[:-1]:
                    if part.isdigit():
                        index = int(part)
                        if isinstance(module, list):
                            if index < len(module):
                                module = module[index]  # Directly index the list
                            else:
                                raise IndexError(f"Index {index} out of range for module list.")
                        else:
                            # If the current module is not a list, check if it has a 'blocks' attribute that's a list
                            if hasattr(module, 'blocks') and isinstance(module.blocks, list):
                                if index < len(module.blocks):
                                    module = module.blocks[index]
                                else:
                                    raise IndexError(f"Index {index} out of range for module.blocks.")
                            else:
                                raise AttributeError(f"Module '{module}' is not a list, cannot index with '{part}'.")
                    else:
                        if hasattr(module, part):
                            module = getattr(module, part)
                        else:
                            raise AttributeError(f"Module '{part}' not found in '{module}'.")

                # Assign the parameter to the final attribute
                setattr(module, parts[-1], param_value)
                debug_print(f"Assigned parameter '{param_name}' with shape {param_value.shape} and dtype {param_value.dtype}")
            except AttributeError as e:
                debug_print(f"Error assigning parameter '{param_name}': {e}")
                raise e
            except IndexError as e:
                debug_print(f"Error assigning parameter '{param_name}': {e}")
                raise e

            # Free memory after assignment to prevent memory bloat
            del param_value
            gc.collect()

def debug_model_dtypes(model: CustomWhisper):
    debug_print("Debugging model dtypes and shapes:")
    debug_print(f"Encoder conv1.weight dtype: {model.encoder.conv1.weight.dtype}, shape: {model.encoder.conv1.weight.shape}")
    debug_print(f"Encoder blocks[0].attn.query.weight dtype: {model.encoder.blocks[0].attn.query.weight.dtype}, shape: {model.encoder.blocks[0].attn.query.weight.shape}")
    debug_print(f"Decoder blocks[0].attn.query.weight dtype: {model.decoder.blocks[0].attn.query.weight.dtype}, shape: {model.decoder.blocks[0].attn.query.weight.shape}")
    debug_print(f"Decoder token_embedding.weight dtype: {model.decoder.token_embedding.weight.dtype}, shape: {model.decoder.token_embedding.weight.shape}")
    # Add checks for weightss
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            debug_print(f"{name}: dtype={module.weight.dtype}")

def replace_layer_norms(module, dtype):
    for name, child in module.named_modules():
        if isinstance(child, nn.LayerNorm):
            # Split the module's full name to get parent path and attribute name
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent_module = get_submodule(module, parent_name.split('.'))
            else:
                parent_module = module
                attr_name = name

            # Use child.weight.shape as the normalized_shape
            normalized_shape = child.weight.shape

            # If normalized_shape is a single-element tuple, convert it to int
            if isinstance(normalized_shape, tuple) and len(normalized_shape) == 1:
                normalized_shape = normalized_shape[0]
                debug_print(f"Converted normalized_shape for '{name}': {normalized_shape}")
            else:
                # Keep as is for multi-dimensional LayerNorm
                debug_print(f"Keeping normalized_shape for '{name}': {normalized_shape}")

            # Create a FloatLayerNorm instance with the same parameters
            new_layer_norm = FloatLayerNorm(normalized_shape, child.eps, original_layer=child, dtype=dtype)

            # Replace the LayerNorm with FloatLayerNorm
            set_submodule(module, name.split('.'), new_layer_norm)
            debug_print(f"Replaced LayerNorm at '{name}' with FloatLayerNorm.")

def map_state_dict(state_dict: Dict[str, torch.Tensor], np_dtype: np.dtype) -> Dict[str, np.ndarray]:
    mapped_dict = {}
    specific_mappings = {
        'proj_out.weight': 'decoder.token_embedding.weight',
        'proj_out.bias': 'decoder.token_embedding.bias',
        'decoder.proj_out.weight': 'decoder.token_embedding.weight',
        'decoder.proj_out.bias': 'decoder.token_embedding.bias',
        # Add more specific mappings here if necessary
    }

    for k, v in state_dict.items():
        # Specific Mappings
        if k in specific_mappings:
            new_k = specific_mappings[k]
            mapped_dict[new_k] = v.detach().cpu().numpy().astype(np_dtype)
            debug_print(f"Mapped {k} to {new_k}: shape {mapped_dict[new_k].shape}, dtype {mapped_dict[new_k].dtype}")
            continue  # Proceed to next parameter

        # Dynamic Handling for MLP's proj_out
        if '.mlp.proj_out.' in k:
            new_k = k.replace('.mlp.proj_out.', '.mlp2.')
            mapped_dict[new_k] = v.detach().cpu().numpy().astype(np_dtype)
            debug_print(f"Mapped {k} to {new_k}: shape {mapped_dict[new_k].shape}, dtype {mapped_dict[new_k].dtype}")
            continue

        # General Replacement: Replace 'proj_out' with 'out' in all keys
        new_k = k.replace('proj_out.', 'out.')
        new_k = new_k.replace('.proj_out.', '.out.')
        new_k = new_k.replace('.proj_out', '.out')    # Handles cases where 'proj_out' is at the end
        new_k = new_k.replace('proj_out', 'out')      # Handles cases without preceding dot

        # Handle other naming conventions
        if new_k.startswith('model.encoder.'):
            new_k = new_k.replace('model.encoder.', 'encoder.')
            if 'layers' in new_k:
                new_k = new_k.replace('layers', 'blocks')
                new_k = new_k.replace('self_attn', 'attn')
                new_k = new_k.replace('encoder_attn', 'cross_attn')
                new_k = new_k.replace('k_proj', 'key')
                new_k = new_k.replace('v_proj', 'value')
                new_k = new_k.replace('q_proj', 'query')
                new_k = new_k.replace('out_proj', 'out')  # Already handled by general replacement
                new_k = new_k.replace('fc1', 'mlp1')
                new_k = new_k.replace('fc2', 'mlp2')
                new_k = new_k.replace('final_layer_norm', 'mlp_ln')
                # Handle layer norm naming
                if 'self_attn_layer_norm' in new_k:
                    new_k = new_k.replace('self_attn_layer_norm', 'attn_ln')
                elif 'attn_layer_norm' in new_k:
                    new_k = new_k.replace('attn_layer_norm', 'attn_ln')
                elif 'mlp_layer_norm' in new_k:
                    new_k = new_k.replace('mlp_layer_norm', 'mlp_ln')
            elif 'conv' in new_k:
                if 'weight' in new_k:
                    # Transpose Conv1d weights from (C_out, C_in, K) to (C_out, K, C_in)
                    mapped_weight = v.detach().cpu().numpy().astype(np_dtype).transpose(0, 2, 1)
                    mapped_dict[new_k] = mapped_weight
                    debug_print(f"Mapped {k} to {new_k} with transposition: shape {mapped_dict[new_k].shape}, dtype {mapped_dict[new_k].dtype}")
                else:
                    # Biases or other conv parameters
                    mapped_dict[new_k] = v.detach().cpu().numpy().astype(np_dtype)
                    debug_print(f"Mapped {k} to {new_k}: shape {mapped_dict[new_k].shape}, dtype {mapped_dict[new_k].dtype}")
                continue
            elif 'embed_positions' in new_k:
                new_k = new_k.replace('embed_positions.weight', '_positional_embedding')
            elif 'layer_norm' in new_k:
                if 'mlp_layer_norm' in new_k:
                    new_k = new_k.replace('mlp_layer_norm', 'mlp_ln')
                else:
                    new_k = new_k.replace('layer_norm', 'ln_post')

        elif new_k.startswith('model.decoder.'):
            new_k = new_k.replace('model.decoder.', 'decoder.')
            if 'layers' in new_k:
                new_k = new_k.replace('layers', 'blocks')
                new_k = new_k.replace('self_attn', 'attn')
                new_k = new_k.replace('encoder_attn', 'cross_attn')
                new_k = new_k.replace('k_proj', 'key')
                new_k = new_k.replace('v_proj', 'value')
                new_k = new_k.replace('q_proj', 'query')
                new_k = new_k.replace('out_proj', 'out')  # Already handled by general replacement
                new_k = new_k.replace('fc1', 'mlp1')
                new_k = new_k.replace('fc2', 'mlp2')
                new_k = new_k.replace('final_layer_norm', 'mlp_ln')
                # Handle layer norm naming
                if 'self_attn_layer_norm' in new_k:
                    new_k = new_k.replace('self_attn_layer_norm', 'attn_ln')
                elif 'encoder_attn_layer_norm' in new_k:
                    new_k = new_k.replace('encoder_attn_layer_norm', 'cross_attn_ln')
                elif 'attn_layer_norm' in new_k:
                    new_k = new_k.replace('attn_layer_norm', 'attn_ln')
                elif 'mlp_layer_norm' in new_k:
                    new_k = new_k.replace('mlp_layer_norm', 'mlp_ln')
            elif 'embed_positions' in new_k:
                new_k = new_k.replace('embed_positions.weight', 'positional_embedding')
            elif 'embed_tokens' in new_k:
                new_k = 'decoder.token_embedding.weight'
            elif 'layer_norm' in new_k:
                if 'mlp_layer_norm' in new_k:
                    new_k = new_k.replace('mlp_layer_norm', 'mlp_ln')
                else:
                    new_k = new_k.replace('layer_norm', 'ln')

        elif 'mlp.proj_out' in new_k:
            # Handle 'mlp.proj_out' parameters mapping to 'mlp2'
            new_k = new_k.replace('mlp.proj_out.weight', 'mlp2.weight')
            new_k = new_k.replace('mlp.proj_out.bias', 'mlp2.bias')
        else:
            # For any other keys not starting with 'model.encoder.' or 'model.decoder.', keep as is
            new_k = k

        # Assign other weights directly
        if new_k not in mapped_dict:
            mapped_dict[new_k] = v.detach().cpu().numpy().astype(np_dtype)
            debug_print(f"Mapped {k} to {new_k}: shape {mapped_dict[new_k].shape}, dtype {mapped_dict[new_k].dtype}")

    # Identify any unmapped 'proj_out' parameters for debugging
    unmapped_proj_out = [k for k in state_dict.keys() if 'proj_out' in k and k not in specific_mappings]
    if unmapped_proj_out:
        debug_print("Warning: The following 'proj_out' parameters were not specifically mapped and have been replaced with 'out':")
        for k in unmapped_proj_out:
            mapped_k = k.replace('proj_out.', 'out.').replace('.proj_out', '.out').replace('proj_out', 'out')
            debug_print(f"  Original key: {k} --> Mapped key: {mapped_k}")

    # Final Check: Ensure no 'proj_out' remains
    final_proj_out = [k for k in mapped_dict.keys() if 'proj_out' in k]
    if final_proj_out:
        raise ValueError(f"After mapping, the following parameters still contain 'proj_out': {final_proj_out}")

    return mapped_dict

def assert_dtypes(model, expected_dtype=mx.float16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.weight.dtype == expected_dtype, f"Linear layer '{name}.weight' is not {expected_dtype}."
            if hasattr(module, 'bias') and module.bias is not None:
                assert module.bias.dtype == expected_dtype, f"Linear layer '{name}.bias' is not {expected_dtype}."

        elif isinstance(module, QuantizedLinear):
            assert module.weight.dtype == mx.uint32, f"QuantizedLinear layer '{name}.weight' is not uint32."
            assert module.scale.dtype == expected_dtype, f"QuantizedLinear layer '{name}.scale' is not {expected_dtype}."
            if hasattr(module, 'bias') and module.bias is not None:
                assert module.bias.dtype == expected_dtype, f"QuantizedLinear layer '{name}.bias' is not {expected_dtype}."

        elif isinstance(module, FloatLayerNorm):
            assert module.weight.dtype == expected_dtype, f"FloatLayerNorm layer '{name}.weight' is not {expected_dtype}."
            assert module.bias.dtype == expected_dtype, f"FloatLayerNorm layer '{name}.bias' is not {expected_dtype}."

        elif isinstance(module, nn.Conv1d):
            assert module.weight.dtype == expected_dtype, f"Conv1d layer '{name}.weight' is not {expected_dtype}."
            if hasattr(module, 'bias') and module.bias is not None:
                assert module.bias.dtype == expected_dtype, f"Conv1d layer '{name}.bias' is not {expected_dtype}."

        elif isinstance(module, nn.Embedding):
            assert module.weight.dtype == expected_dtype, f"Embedding layer '{name}.weight' is not {expected_dtype}."

        # Add additional layer types if necessary

    # Check special buffers
    assert model.encoder._positional_embedding.dtype == expected_dtype, "encoder._positional_embedding is not float16."
    assert model.decoder.positional_embedding.dtype == expected_dtype, "decoder.positional_embedding is not float16."
    assert model.decoder._mask.dtype == expected_dtype, "decoder._mask is not float16."



def load_torch_model(name_or_path: str, download_root: str = None, np_dtype=np.float16, mx_dtype=mx.float16) -> CustomWhisper:
    def transpose_conv_weights(mapped_state_dict):
        # No transposition needed for Conv1d weights as it's handled in map_state_dict
        return mapped_state_dict

    if download_root is None:
        download_root = os.path.join(os.path.expanduser("~"), ".cache/whisper")

    os.makedirs(download_root, exist_ok=True)

    # Load the model using Transformers or safetensors
    if name_or_path.endswith(".safetensors"):
        debug_print(f"Loading safetensors model from {name_or_path}")
        state_dict = safetensors.torch.load_file(name_or_path, device='cpu')
        # After loading the state_dict
        proj_out_params = [k for k in state_dict.keys() if 'proj_out' in k]
        debug_print(f"Found {len(proj_out_params)} parameters containing 'proj_out':")
        for p in proj_out_params:
            debug_print(f"  {p}")

        hf_model = WhisperForConditionalGeneration.from_pretrained(name_or_path)
        config = hf_model.config
    else:
        debug_print(f"Loading PyTorch model from {name_or_path}")
        hf_model = WhisperForConditionalGeneration.from_pretrained(name_or_path, device_map='cpu')
        state_dict = hf_model.state_dict()
        config = hf_model.config

    # Verify the original Conv1D weight shape
    original_conv1_shape = state_dict.get('model.encoder.conv1.weight', None)
    if original_conv1_shape is not None:
        debug_print(f"Original 'model.encoder.conv1.weight' shape: {original_conv1_shape.shape}")
    else:
        raise KeyError("model.encoder.conv1.weight not found in state_dict")

    # Print configuration to verify correctness
    debug_print(f"Config num_mel_bins: {config.num_mel_bins}")
    debug_print(f"Config d_model: {config.d_model}")
    debug_print(f"Config encoder_attention_heads: {config.encoder_attention_heads}")
    debug_print(f"Config encoder_layers: {config.encoder_layers}")
    debug_print(f"Config decoder_attention_heads: {config.decoder_attention_heads}")
    debug_print(f"Config decoder_layers: {config.decoder_layers}")
    debug_print(f"Config vocab_size: {config.vocab_size}")

    # Extract model dimensions using the imported ModelDimensions
    dims = ModelDimensions(
        n_mels=config.num_mel_bins,
        n_audio_ctx=config.max_source_positions,
        n_audio_state=config.d_model,
        n_audio_head=config.encoder_attention_heads,
        n_audio_layer=config.encoder_layers,
        n_vocab=config.vocab_size,
        n_text_ctx=config.max_target_positions,
        n_text_state=config.d_model,
        n_text_head=config.decoder_attention_heads,
        n_text_layer=config.decoder_layers
    )

    debug_print(f"Model Dimensions: {dims}")

    # Create a CustomWhisper model instance
    debug_print(f"CustomWhisper initialized with MLX dtype: {mx_dtype}")
    model = CustomWhisper(dims, dtype=mx_dtype)
    model = model.astype(mx_dtype)  # Use the updated astype method

    verify_model_dtypes(model, expected_dtype=mx_dtype)
    debug_print(f"Number of mel bins (n_mels): {model.dims.n_mels}")

    # Map the state dict
    mapped_state_dict = map_state_dict(state_dict, np_dtype)

    # No transposition of convolutional weights (handled in map_state_dict)
    mapped_state_dict = transpose_conv_weights(mapped_state_dict)

    debug_print("Replacing layer norms...")
    replace_layer_norms(model, dtype=mx_dtype)

    # After mapping, check for any remaining 'proj_out' in the mapped keys
    remaining_proj_out = [k for k in mapped_state_dict.keys() if 'proj_out' in k]
    if remaining_proj_out:
        debug_print("Warning: The following 'proj_out' parameters still contain 'proj_out' and may not be correctly mapped:")
        for k in remaining_proj_out:
            debug_print(f"  {k}")

    # Convert numpy arrays to MLX arrays
    mlx_state_dict = {}
    for k, v in mapped_state_dict.items():
        mlx_state_dict[k] = mx.array(v, dtype=mx_dtype)
        debug_print(f"Converted {k} to MLX array: shape {v.shape}, dtype {v.dtype}")

    # After creating the model instance
    debug_print("Model created. Debugging shapes:")
    debug_print(f"encoder.conv1.weight shape: {model.encoder.conv1.weight.shape}")  # Expected: (1280, 128, 3)
    debug_print(f"encoder.conv2.weight shape: {model.encoder.conv2.weight.shape}")  # Expected: (1280, 1280, 3)
    debug_print(f"decoder.token_embedding.weight shape: {model.decoder.token_embedding.weight.shape}")  # Expected: (51866, 1280)

    # Update model parameters without transposing convolutional weights
    model.update(mlx_state_dict)

    # Detailed debugging
    debug_print("Model dtype after update:", model.dtype)
    debug_print("Encoder conv1.weight dtype:", model.encoder.conv1.weight.dtype)
    debug_print("Encoder conv1.weight shape:", model.encoder.conv1.weight.shape)
    debug_print(f"encoder.conv2.weight shape: {model.encoder.conv2.weight.shape}")
    debug_print(f"decoder.token_embedding.weight shape: {model.decoder.token_embedding.weight.shape}")

    debug_print("Decoder blocks[0].attn.query.weight dtype:", model.decoder.blocks[0].attn.query.weight.dtype)
    debug_print("Decoder blocks[0].attn.query.weight shape:", model.decoder.blocks[0].attn.query.weight.shape)

    # Create dummy input with shape (batch_size, sequence_length, channels)
    dummy_input_length = 3000  # Must be <= n_audio_ctx
    dummy_input = mx.random.normal((1, dummy_input_length, model.dims.n_mels)).astype(mx_dtype)  # Shape: (1, 3000, 128)
    debug_print(f"Dummy input shape: {dummy_input.shape}")  # Should be (1, 3000, 128)

    # Verify input channels
    expected_in_channels = model.encoder.conv1.weight.shape[2]  # C_in=128
    actual_in_channels = dummy_input.shape[2]
    debug_print(f"Expected input channels: {expected_in_channels}, Actual input channels: {actual_in_channels}")
    if actual_in_channels != expected_in_channels:
        raise ValueError(f"Input channels mismatch: expected {expected_in_channels}, got {actual_in_channels}")

    # Embed audio and check output
    try:
        audio_features = model.embed_audio(dummy_input)
        debug_print(f"Audio features dtype after loading: {audio_features.dtype}, shape: {audio_features.shape}")
    except Exception as e:
        debug_print(f"Error during embed_audio: {e}")
        raise e

    return model

def quantize_weights(weights, group_size, bits):
    """
    Quantize weights using group quantization and pack into uint32.
    """
    assert weights.ndim == 2
    orig_shape = weights.shape
    weights = weights.astype(np.float32)

    # Reshape to (out_features, num_groups, group_size)
    num_groups = weights.shape[1] // group_size
    remainder = weights.shape[1] % group_size
    if remainder != 0:
        # Pad weights to make the last group full
        padding = group_size - remainder
        weights = np.pad(weights, ((0, 0), (0, padding)), mode='constant')
    else:
        padding = 0

    weights = weights.reshape(orig_shape[0], -1, group_size)

    # Compute scales
    max_abs = np.max(np.abs(weights), axis=-1, keepdims=True)
    scales = max_abs / ((2 ** bits) - 1)

    # Quantize weights
    weights_q = np.round(weights / scales).astype(np.int32)
    weights_q = np.clip(weights_q, -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1)

    # Pack weights into uint32
    def pack_weights(weights_group):
        total_bits = weights_group.size * bits
        num_uint32 = (total_bits + 31) // 32
        packed = np.zeros(num_uint32, dtype=np.uint32)
        flat_weights = weights_group.flatten()
        for i in range(flat_weights.size):
            bit_position = (i * bits) % 32
            array_index = (i * bits) // 32
            if array_index >= packed.size:
                # Prevent out-of-bounds in case of overflow
                break
            packed[array_index] |= (flat_weights[i] & ((1 << bits) - 1)) << bit_position
        return packed

    packed_weights = []
    for w in tqdm(weights_q, desc="Packing weights", leave=False):
        packed_w = pack_weights(w)
        packed_weights.append(packed_w)
    packed_weights = np.stack(packed_weights, axis=0)

    # Remove padding from scales if any
    if padding != 0:
        scales = scales[:, :-1, :]
    scales = scales.reshape(orig_shape[0], -1)

    return packed_weights, scales

def get_submodule(parent_module, modules_list):
    """
    Recursively get the submodule specified by modules_list.
    """
    for m in modules_list:
        if isinstance(parent_module, (list, tuple)):
            index = int(m)
            parent_module = parent_module[index]
        else:
            parent_module = getattr(parent_module, m)
    return parent_module

def set_submodule(parent_module, modules_list, module_to_set):
    """
    Set the submodule in parent_module specified by modules_list to module_to_set.
    """
    for m in modules_list[:-1]:
        if isinstance(parent_module, (list, tuple)):
            index = int(m)
            parent_module = parent_module[index]
        elif isinstance(parent_module, nn.Module):
            parent_module = getattr(parent_module, m)
        else:
            raise TypeError(f"Unexpected type for parent_module: {type(parent_module)}")

    last_m = modules_list[-1]
    if isinstance(parent_module, (list, tuple)):
        index = int(last_m)
        parent_module[index] = module_to_set
    elif isinstance(parent_module, nn.Module):
        setattr(parent_module, last_m, module_to_set)
    else:
        raise TypeError(f"Unexpected type for parent_module: {type(parent_module)}")

def verify_quantized_linear_weights(model, quantized_param_names):
    for param_name in quantized_param_names:
        weight = getattr(model, param_name, None)
        if weight is None:
            raise ValueError(f"QuantizedLinear layer's weight '{param_name}' not found in the model.")
        if weight.dtype != mx.uint32:
            raise TypeError(f"QuantizedLinear layer '{param_name}' has weight dtype {weight.dtype}, expected uint32.")
        else:
            debug_print(f"QuantizedLinear layer '{param_name}' weight dtype correctly set to {weight.dtype}.")


def quantize_model(model, group_size, bits, main_dtype, np_dtype):
    """
    Quantize linear layers in the model using MLX's QuantizedLinear.from_linear(),
    except for those affecting audio_features.

    Parameters:
    - model: The CustomWhisper model instance to be quantized.
    - group_size (int): The group size for quantization.
    - bits (int): The bit width for quantization.
    - main_dtype (mx.Dtype): The target data type for non-quantized layers (e.g., mx.float16).
    - np_dtype (np.dtype): The NumPy data type for scales (e.g., np.float16).

    Returns:
    - model: The quantized model instance.
    """
    modules = list(model.named_modules())

    # Initial dtype verification before quantization
    debug_print("Before quantization:")
    verify_model_dtypes(model, expected_dtype=main_dtype)

    pbar = tqdm(total=len(modules), desc="Quantizing layers")

    for idx, (name, module) in enumerate(modules):
        # Define regex patterns for layers to skip quantization
        skip_patterns = [
            r'^encoder\.blocks\.\d+\.attn\.key$',    # e.g., encoder.blocks.0.attn.key
            r'^decoder\.blocks\.\d+\.attn\.key$',    # e.g., decoder.blocks.0.attn.key
            r'^encoder\.blocks\.\d+\.attn\.query$',  # e.g., encoder.blocks.0.attn.query
            r'^decoder\.blocks\.\d+\.attn\.query$',  # e.g., decoder.blocks.0.attn.query
            r'^decoder\.token_embedding$',           # Skip token embedding layers
            r'^encoder\.conv1$',                     # Example: encoder.conv1, adjust as needed
            # Add more patterns to skip as necessary
        ]

        # Check if the current layer matches any skip pattern
        skip = any(re.match(pattern, name) for pattern in skip_patterns)

        if isinstance(module, nn.Linear) and not skip:
            # Quantize this Linear layer using QuantizedLinear.from_linear()
            quantized_linear = QuantizedLinear.from_linear(
                linear_layer=module,
                group_size=group_size,
                bits=bits
            )
            debug_print(f"QuantizedLinear layer '{name}' created from Linear layer.")

            # Replace the original Linear layer with QuantizedLinear in the model
            modules_list = name.split('.')
            set_submodule(model, modules_list, quantized_linear)
            debug_print(f"Replaced Linear layer '{name}' with QuantizedLinear.")

        pbar.update(1)

        # Early validation after quantizing a certain number of layers
        if (idx + 1) % 50 == 0:  # Adjust the frequency as needed
            try:
                dummy_input = mx.random.normal((1, 3000, model.dims.n_mels)).astype(main_dtype)
                debug_print("Attempting audio features test with dummy input.")
                audio_features = model.embed_audio(dummy_input)
                if audio_features.dtype != main_dtype:
                    raise TypeError(f"After quantizing {idx + 1} layers, audio_features dtype is {audio_features.dtype}, expected {main_dtype}.")
                else:
                    debug_print(f"Early validation passed after quantizing {idx + 1} layers.")
            except TypeError as e:
                pbar.close()
                raise e

    pbar.close()

    debug_print("After quantization:")
    verify_model_dtypes(model, expected_dtype=main_dtype)

    # Reapply the astype method to ensure all modules are cast to main_dtype
    debug_print("Reapplying astype to ensure all modules are cast to main_dtype after quantization.")
    model = model.astype(main_dtype)

    # Final dtype check
    try:
        debug_print("Attempting final dtype check for audio features with dummy input.")
        dummy_input = mx.random.normal((1, 3000, model.dims.n_mels)).astype(main_dtype)
        audio_features = model.embed_audio(dummy_input)
        if audio_features.dtype != main_dtype:
            raise TypeError(f"After quantization, audio_features dtype is {audio_features.dtype}, expected {main_dtype}.")
        else:
            debug_print("Final dtype check passed.")
    except TypeError as e:
        raise e

    return model

def quantize(model, args, main_dtype, np_dtype):

    # Quantize the model:
    quantized_model = quantize_model(model, args.q_group_size, args.q_bits, main_dtype=main_dtype, np_dtype=np_dtype)

    # Update the config:
    quantized_config = vars(quantized_model.dims)
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }

    # Collect the quantized weights
    params_dict = quantized_model.parameters()
    weights = {}
    for key, value in tree_flatten(params_dict):
        weights[key] = value

    debug_print("After quantization:")
    verify_model_dtypes(quantized_model, expected_dtype=main_dtype)

    # Check if encoder output dtype is correct
    dummy_input = mx.random.normal((1, 3000, quantized_model.dims.n_mels)).astype(main_dtype)
    debug_print("Generating encoder output with dummy input.")
    encoder_output = quantized_model.encoder(dummy_input)
    debug_print(f"Encoder output dtype: {encoder_output.dtype}")

    # Check audio features dtype
    audio_features = quantized_model.embed_audio(dummy_input)
    debug_print(f"Audio features dtype after quantization: {audio_features.dtype}")

    return weights, quantized_config

def verify_model_dtypes(model, expected_dtype=mx.float16):
    debug_print("Verifying model dtypes...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.dtype != expected_dtype:
                debug_print(f"Linear layer '{name}.weight' has dtype {module.weight.dtype}, expected {expected_dtype}.")
            else:
                debug_print(f"Linear layer '{name}.weight' correctly set to {expected_dtype}.")
            
            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.dtype != expected_dtype:
                    debug_print(f"Linear layer '{name}.bias' has dtype {module.bias.dtype}, expected {expected_dtype}.")
                else:
                    debug_print(f"Linear layer '{name}.bias' correctly set to {expected_dtype}.")

        elif isinstance(module, QuantizedLinear):
            if module.weight.dtype != mx.uint32:
                debug_print(f"QuantizedLinear layer '{name}.weight' has dtype {module.weight.dtype}, expected uint32.")
            else:
                debug_print(f"QuantizedLinear layer '{name}.weight' correctly set to uint32.")
            
            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.dtype != expected_dtype:
                    debug_print(f"QuantizedLinear layer '{name}.bias' has dtype {module.bias.dtype}, expected {expected_dtype}.")
                else:
                    debug_print(f"QuantizedLinear layer '{name}.bias' correctly set to {expected_dtype}.")

        elif isinstance(module, FloatLayerNorm):
            if module.weight.dtype != expected_dtype:
                debug_print(f"FloatLayerNorm layer '{name}.weight' has dtype {module.weight.dtype}, expected {expected_dtype}.")
            else:
                debug_print(f"FloatLayerNorm layer '{name}.weight' correctly set to {expected_dtype}.")
            
            if module.bias.dtype != expected_dtype:
                debug_print(f"FloatLayerNorm layer '{name}.bias' has dtype {module.bias.dtype}, expected {expected_dtype}.")
            else:
                debug_print(f"FloatLayerNorm layer '{name}.bias' correctly set to {expected_dtype}.")

        elif isinstance(module, nn.Conv1d):
            if module.weight.dtype != expected_dtype:
                debug_print(f"Conv1d layer '{name}.weight' has dtype {module.weight.dtype}, expected {expected_dtype}.")
            else:
                debug_print(f"Conv1d layer '{name}.weight' correctly set to {expected_dtype}.")
            
            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.dtype != expected_dtype:
                    debug_print(f"Conv1d layer '{name}.bias' has dtype {module.bias.dtype}, expected {expected_dtype}.")
                else:
                    debug_print(f"Conv1d layer '{name}.bias' correctly set to {expected_dtype}.")

        elif isinstance(module, nn.Embedding):
            if module.weight.dtype != expected_dtype:
                debug_print(f"Embedding layer '{name}.weight' has dtype {module.weight.dtype}, expected {expected_dtype}.")
            else:
                debug_print(f"Embedding layer '{name}.weight' correctly set to {expected_dtype}.")

        # Add additional layer types if necessary

    # Check special buffers
    if hasattr(model.encoder, '_positional_embedding'):
        if model.encoder._positional_embedding.dtype != expected_dtype:
            debug_print(f"encoder._positional_embedding has dtype {model.encoder._positional_embedding.dtype}, expected {expected_dtype}.")
        else:
            debug_print(f"encoder._positional_embedding correctly set to {expected_dtype}.")

    if hasattr(model.decoder, 'positional_embedding'):
        if model.decoder.positional_embedding.dtype != expected_dtype:
            debug_print(f"decoder.positional_embedding has dtype {model.decoder.positional_embedding.dtype}, expected {expected_dtype}.")
        else:
            debug_print(f"decoder.positional_embedding correctly set to {expected_dtype}.")

    if hasattr(model.decoder, '_mask'):
        if model.decoder._mask.dtype != expected_dtype:
            debug_print(f"decoder._mask has dtype {model.decoder._mask.dtype}, expected {expected_dtype}.")
        else:
            debug_print(f"decoder._mask correctly set to {expected_dtype}.")

    debug_print("All dtypes verified.")


def upload_to_hub(path: str, name: str, torch_name_or_path: str):
    """
    Upload the MLX model to Hugging Face Hub.
    
    Parameters:
    - path: Path to the MLX model directory.
    - name: Name of the repository on Hugging Face Hub.
    - torch_name_or_path: The original Torch model name or path.
    """
    import os

    repo_id = f"mlx-community/{name}"
    text = f"""---
library_name: mlx
---

# {name}
This model was converted to MLX format from [{torch_name_or_path}](). 

## Use with MLX
bash
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/whisper/
pip install -r requirements.txt

>> import mlx_whisper
>> result = mlx_whisper.transcribe("test.mp3", path_or_hf_repo="mlx-community/{name}")
>> print(result)

"""
    # Create README.md
    readme_path = os.path.join(path, "README.md")
    with open(readme_path, "w") as f:
        f.write(text)

    # Set logging to info
    hf_logging.set_verbosity_info()

    api = HfApi()
    try:
        # Create repository
        api.create_repo(repo_id=repo_id, exist_ok=True)
        debug_print(f"Repository {repo_id} created or already exists.")

        # Upload folder
        api.upload_folder(
            folder_path=path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Model uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face Hub: {e}")


def test_encoder(model):
    debug_print("Testing encoder...")
    dummy_input_length = 3000
    dummy_input = mx.random.normal((1, dummy_input_length, model.dims.n_mels)).astype(model.dtype)
    debug_print(f"Original dummy input shape: {dummy_input.shape}")

    # Verify input channels
    expected_in_channels = model.encoder.conv1.weight.shape[2]
    actual_in_channels = dummy_input.shape[2]
    debug_print(f"Expected input channels: {expected_in_channels}, Actual input channels: {actual_in_channels}")
    if actual_in_channels != expected_in_channels:
        raise ValueError(f"Input channels mismatch: expected {expected_in_channels}, got {actual_in_channels}")

    try:
        encoder_output = model.embed_audio(dummy_input)
        debug_print(f"Encoder output shape: {encoder_output.shape}")
        debug_print(f"Encoder output dtype: {encoder_output.dtype}")
        assert encoder_output.dtype == model.dtype, f"Encoder output dtype mismatch: {encoder_output.dtype} != {model.dtype}"
        debug_print("Encoder test successful!")
    except Exception as e:
        print(f"Encoder test failed: {str(e)}")
        raise e


def main():

    parser = argparse.ArgumentParser(description="Convert Whisper weights to MLX.")
    parser.add_argument(
        "--torch-name-or-path",
        type=str,
        required=True,
        help="The path to the model or a model identifier from the HuggingFace Hub.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_models",
        help="The path to save the MLX model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="The dtype to save the MLX model.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q_group_size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q_bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        help="Enable verbose debug output.",
        action="store_true",
    )

    args = parser.parse_args()
    
    VERBOSE = args.verbose

    # Correct dtype mapping
    mx_dtype, np_dtype = get_dtype(args.dtype)
    debug_print(f"Using MLX dtype: {mx_dtype}, NumPy dtype: {np_dtype}")

    print("[INFO] Loading")
    model = load_torch_model(args.torch_name_or_path, np_dtype=np_dtype, mx_dtype=mx_dtype)
    debug_model_dtypes(model)
    verify_model_dtypes(model, expected_dtype=mx_dtype)
    test_encoder(model)

    # Adjust the dummy input length to 3000
    dummy_input_length = 3000  # Adjusted to match input length that results in output length matching n_audio_ctx
    dummy_input = mx.random.normal((1, dummy_input_length, model.dims.n_mels)).astype(mx_dtype)  # Shape: (1, 3000, 128)
    debug_print(f"Dummy input shape: {dummy_input.shape}")  # Should be (1, 3000, 128)

    # Verify input channels
    expected_in_channels = model.encoder.conv1.weight.shape[2]  # in_channels=128
    actual_in_channels = dummy_input.shape[2]
    debug_print(f"Expected input channels: {expected_in_channels}, Actual input channels: {actual_in_channels}")
    if actual_in_channels != expected_in_channels:
        raise ValueError(f"Input channels mismatch: expected {expected_in_channels}, got {actual_in_channels}")

    # Embed audio and check output
    try:
        audio_features = model.embed_audio(dummy_input)
        debug_print(f"Audio features dtype after loading: {audio_features.dtype}, shape: {audio_features.shape}")
    except Exception as e:
        debug_print(f"Error during embed_audio: {e}")
        raise e

    config = vars(model.dims)

    if args.quantize:
        print("[INFO] Quantizing")
        model = quantize_model(model, args.q_group_size, args.q_bits, main_dtype=mx_dtype, np_dtype=np_dtype)

        verify_model_dtypes(model, expected_dtype=mx_dtype)

        weights, config = quantize(model, args, main_dtype=mx_dtype, np_dtype=np_dtype)

    else:
        # Collect the weights
        params_dict = model.parameters()
        weights = {}
        for key, value in tree_flatten(params_dict):
            weights[key] = value

    mlx_path = os.path.join(args.mlx_path)
    os.makedirs(mlx_path, exist_ok=True)

    # Save weights
    print("[INFO] Saving")
    np.savez(os.path.join(mlx_path, "weights.npz"), **{k: np.array(v) for k, v in weights.items()})

    # Save config.json with model_type
    with open(os.path.join(mlx_path, "config.json"), "w") as f:
        config["model_type"] = "whisper"
        config["quantization"] = {
            "group_size": args.q_group_size,
            "bits": args.q_bits,
        }
        json.dump(config, f, indent=4)

    if args.upload_name is not None:
        print("[INFO] Uploading to Hugging Face Hub")
        upload_to_hub(mlx_path, args.upload_name, args.torch_name_or_path)

    print("[INFO] Conversion complete")

if __name__ == "__main__":
    main()
