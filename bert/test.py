import argparse
import time
from typing import List

import model
import numpy as np
import mlx.core as mx
from transformers import AutoModel, AutoTokenizer


def run_torch(bert_model: str, batch: List[str]):
    print(f"\n[PyTorch] Loading model and tokenizer: {bert_model}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    torch_model = AutoModel.from_pretrained(bert_model)
    load_time = time.time() - start_time
    print(f"[PyTorch] Model loaded in {load_time:.2f} seconds")
    
    print(f"[PyTorch] Tokenizing batch of {len(batch)} sentences")
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    
    print(f"[PyTorch] Running model inference")
    inference_start = time.time()
    torch_forward = torch_model(**torch_tokens)
    inference_time = time.time() - inference_start
    print(f"[PyTorch] Inference completed in {inference_time:.4f} seconds")
    
    torch_output = torch_forward.last_hidden_state.detach().numpy()
    torch_pooled = torch_forward.pooler_output.detach().numpy()
    
    print(f"[PyTorch] Output shape: {torch_output.shape}")
    print(f"[PyTorch] Pooled output shape: {torch_pooled.shape}")
    
    # Print a small sample of the output to verify sensible values
    print(f"[PyTorch] Sample of output (first token, first 5 values): {torch_output[0, 0, :5]}")
    print(f"[PyTorch] Sample of pooled output (first 5 values): {torch_pooled[0, :5]}")
    
    return torch_output, torch_pooled


def run_mlx(bert_model: str, mlx_model: str, batch: List[str]):
    print(f"\n[MLX] Loading model and tokenizer with weights from: {mlx_model}")
    start_time = time.time()
    mlx_output, mlx_pooled = model.run(bert_model, mlx_model, batch)
    load_and_run_time = time.time() - start_time
    print(f"[MLX] Model loaded and run in {load_and_run_time:.2f} seconds")
    
    # Convert from MLX arrays to numpy for comparison
    # The correct way to convert MLX arrays to numpy
    mlx_output_np = np.array(mlx_output)
    mlx_pooled_np = np.array(mlx_pooled)
    
    print(f"[MLX] Output shape: {mlx_output_np.shape}")
    print(f"[MLX] Pooled output shape: {mlx_pooled_np.shape}")
    
    # Print a small sample of the output to verify sensible values
    print(f"[MLX] Sample of output (first token, first 5 values): {mlx_output_np[0, 0, :5]}")
    print(f"[MLX] Sample of pooled output (first 5 values): {mlx_pooled_np[0, :5]}")
    
    return mlx_output_np, mlx_pooled_np


def compare_outputs(torch_output, torch_pooled, mlx_output, mlx_pooled):
    print("\n[Comparison] Comparing PyTorch and MLX outputs")
    
    # Check shapes
    print(f"[Comparison] Shape match - Output: {torch_output.shape == mlx_output.shape}")
    print(f"[Comparison] Shape match - Pooled: {torch_pooled.shape == mlx_pooled.shape}")
    
    # Calculate differences
    output_max_diff = np.max(np.abs(torch_output - mlx_output))
    output_mean_diff = np.mean(np.abs(torch_output - mlx_output))
    pooled_max_diff = np.max(np.abs(torch_pooled - mlx_pooled))
    pooled_mean_diff = np.mean(np.abs(torch_pooled - mlx_pooled))
    
    print(f"[Comparison] Output - Max absolute difference: {output_max_diff:.6f}")
    print(f"[Comparison] Output - Mean absolute difference: {output_mean_diff:.6f}")
    print(f"[Comparison] Pooled - Max absolute difference: {pooled_max_diff:.6f}")
    print(f"[Comparison] Pooled - Mean absolute difference: {pooled_mean_diff:.6f}")
    
    # Detailed comparison of first few values from first sentence
    print("\n[Comparison] Detailed comparison of first 5 values from first output token:")
    for i in range(5):
        torch_val = torch_output[0, 0, i]
        mlx_val = mlx_output[0, 0, i]
        diff = abs(torch_val - mlx_val)
        print(f"Index {i}: PyTorch={torch_val:.6f}, MLX={mlx_val:.6f}, Diff={diff:.6f}")
    
    # Check if outputs are close
    outputs_close = np.allclose(torch_output, mlx_output, rtol=1e-4, atol=1e-4)
    pooled_close = np.allclose(torch_pooled, mlx_pooled, rtol=1e-4, atol=1e-4)
    
    print(f"\n[Comparison] Outputs match within tolerance: {outputs_close}")
    print(f"[Comparison] Pooled outputs match within tolerance: {pooled_close}")
    
    return outputs_close and pooled_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a BERT-like model for a batch of text and compare PyTorch and MLX outputs."
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="The model identifier for a BERT-like model from Hugging Face Transformers.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/bert-base-uncased.npz",
        help="The path of the stored MLX BERT weights (npz file).",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        default=["This is an example of BERT working in MLX."],
        help="A batch of texts to process. Multiple texts should be separated by spaces.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about the model execution.",
    )

    args = parser.parse_args()
    
    print(f"Testing BERT model: {args.bert_model}")
    print(f"MLX weights: {args.mlx_model}")
    print(f"Input text: {args.text}")
    
    # Run both implementations
    torch_output, torch_pooled = run_torch(args.bert_model, args.text)
    mlx_output, mlx_pooled = run_mlx(args.bert_model, args.mlx_model, args.text)
    
    # Compare outputs
    all_match = compare_outputs(torch_output, torch_pooled, mlx_output, mlx_pooled)
    
    if all_match:
        print("\n✅ TEST PASSED: PyTorch and MLX implementations produce equivalent results!")
    else:
        print("\n❌ TEST FAILED: PyTorch and MLX implementations produce different results.")
