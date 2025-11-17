#!/usr/bin/env python3
"""
Evaluation script matching the paper's methodology
Paper: "Scalable MatMul-free Language Modeling" (arXiv:2406.02528)

Evaluates models on:
- WikiText-2 (main benchmark)
- WikiText-103
- Penn TreeBank (PTB)
- LAMBADA (zero-shot)
"""

import argparse
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError("Checkpoint does not contain config")

    # Create model
    model = HGRNBitForCausalLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def evaluate_perplexity(model, dataset, tokenizer, seq_length=1024, batch_size=8, device='cuda'):
    """
    Evaluate perplexity on a dataset
    Following paper's evaluation methodology
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Tokenize dataset
    print("Tokenizing dataset...")
    if 'text' in dataset.column_names:
        text_column = 'text'
    elif 'sentence' in dataset.column_names:
        text_column = 'sentence'
    else:
        text_column = dataset.column_names[0]

    all_text = " ".join(dataset[text_column])
    tokens = tokenizer(all_text, return_tensors='pt', max_length=None, truncation=False)
    input_ids = tokens['input_ids'][0]

    print(f"Total tokens: {len(input_ids)}")

    # Evaluate in chunks
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids) - seq_length, seq_length), desc="Evaluating"):
            # Get chunk
            chunk = input_ids[i:i + seq_length + 1].unsqueeze(0).to(device)

            if chunk.size(1) <= 1:
                continue

            # Forward pass
            outputs = model(chunk[:, :-1], labels=chunk[:, 1:])
            loss = outputs.loss

            # Accumulate
            total_loss += loss.item() * (chunk.size(1) - 1)
            total_tokens += chunk.size(1) - 1

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Evaluate MatMul-Free LM following paper methodology')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--datasets', nargs='+', default=['wikitext-2', 'wikitext-103'],
                        choices=['wikitext-2', 'wikitext-103', 'ptb', 'lambada'],
                        help='Datasets to evaluate on')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length for evaluation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    print("="*80)
    print("MatMul-Free LM Evaluation - Paper Methodology")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {args.datasets}")
    print(f"Sequence Length: {args.seq_length}")
    print("="*80)
    print()

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print()

    # Evaluation results
    results = {
        'checkpoint': args.checkpoint,
        'model_config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_heads,
        },
        'datasets': {}
    }

    # Evaluate on each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*80}")

        # Load dataset
        if dataset_name == 'wikitext-2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        elif dataset_name == 'wikitext-103':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
        elif dataset_name == 'ptb':
            dataset = load_dataset('ptb_text_only', split='test')
        elif dataset_name == 'lambada':
            dataset = load_dataset('lambada', split='test')
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue

        # Evaluate
        perplexity, loss = evaluate_perplexity(
            model, dataset, tokenizer,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            device=args.device
        )

        print(f"\nResults for {dataset_name}:")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Loss: {loss:.4f}")

        results['datasets'][dataset_name] = {
            'perplexity': perplexity,
            'loss': loss
        }

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to: {args.output}")
    print(f"{'='*80}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Summary:")
    print("-" * 80)
    for dataset_name, metrics in results['datasets'].items():
        print(f"{dataset_name:20s}: Perplexity = {metrics['perplexity']:8.2f}, Loss = {metrics['loss']:.4f}")
    print("-" * 80)


if __name__ == '__main__':
    main()
