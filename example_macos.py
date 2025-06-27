#!/usr/bin/env python3
"""
Example script for running nano-vLLM on macOS.
Optimized for Apple Silicon with Metal Performance Shaders (MPS).
"""
import os
from nanovllm import LLM, SamplingParams
from nanovllm.utils.device_utils import print_device_info, get_device_info
from transformers import AutoTokenizer


def main():
    # Print macOS device information
    print_device_info()
    
    # Get device info for configuration
    device_info = get_device_info()
    
    # Model path for Qwen3-0.6B
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download the model using:")
        print("huggingface-cli download --resume-download Qwen/Qwen3-0.6B \\")
        print("  --local-dir ~/huggingface/Qwen3-0.6B/ \\")
        print("  --local-dir-use-symlinks False")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # macOS optimized configuration
        config_kwargs = {
            'enforce_eager': True,  # Required for MPS compatibility
            'tensor_parallel_size': 1,  # Single device only on macOS
            'max_model_len': 2048,  # Conservative for memory
            'gpu_memory_utilization': 0.8,  # Safe for MPS
        }
        
        print(f"\nInitializing LLM on {device_info['recommended_device']}...")
        llm = LLM(model_path, **config_kwargs)

        # System prompt for chat models
        system_prompt = "Your name is NANO."
        
        # Test prompts
        sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
        prompts = [
            "Hello, what's your name?."
        ]
    
        # Apply chat template if available
        try:
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for prompt in prompts
            ]
        except Exception as e:
            print(f"Chat template not available, using prompts as-is: {e}")
            # Fallback: prepend system prompt manually
            prompts = [f"[SYSTEM] {system_prompt}\n[USER] {prompt}" for prompt in prompts]
        
        print("\nGenerating responses...")
        outputs = llm.generate(prompts, sampling_params)

        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {output['text']}")
            print(f"Tokens generated: {len(output['token_ids'])}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model is downloaded correctly")
        print("2. Check available memory (reduce max_model_len if needed)")
        print("3. Ensure you have the latest PyTorch with MPS support")
        print("4. Try restarting if MPS gives issues")


if __name__ == "__main__":
    main()
