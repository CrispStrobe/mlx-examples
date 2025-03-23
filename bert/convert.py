import argparse
import os
from pathlib import Path

import numpy
from transformers import AutoModel, AutoConfig


def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key


def convert(bert_model: str, mlx_model: str) -> None:
    # Load model and its configuration
    model = AutoModel.from_pretrained(bert_model)
    config = AutoConfig.from_pretrained(bert_model)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(mlx_model)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save config as well
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        f.write(config.to_json_string())
        
    print(f"Saved model config to {config_path}")
    
    # Save the tensors
    tensors = {
        replace_key(key): tensor.numpy() for key, tensor in model.state_dict().items()
    }
    numpy.savez(mlx_model, **tensors)
    print(f"Saved model weights to {mlx_model}")
    print(f"Model vocab size: {config.vocab_size}, hidden size: {config.hidden_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BERT weights to MLX.")
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="The huggingface name of the BERT model to save. Any BERT-like model can be specified.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/bert-base-uncased.npz",
        help="The output path for the MLX BERT weights.",
    )
    args = parser.parse_args()

    convert(args.bert_model, args.mlx_model)
