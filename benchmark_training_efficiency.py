#!/usr/bin/env python3
"""
Training Efficiency Benchmark Script
Replicates the paper's Figure 3(a-b) experiments:
- Fused vs Vanilla BitLinear training performance
- Memory consumption vs batch size
- Training time per iteration vs batch size
- Comparison with Pythia baseline models

Based on "Scalable MatMul-free Language Modeling" Section 5.1
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from src.ops.fusedbitnet import FusedBitLinear
from src.ops.bitnet import BitLinear


class TrainingEfficiencyBenchmark:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.results = {
            'fused_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []},
            'vanilla_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
                'pytorch_version': torch.__version__,
                'model_path': str(model_path) if model_path else None
            }
        }
    
    def create_test_model(self, use_fused=True, hidden_size=2048, num_layers=24):
        """Create 1.3B model matching paper specs"""
        config = HGRNBitConfig(
            vocab_size=50000,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_heads=16,
            max_position_embeddings=2048,
            expand_ratio=1,
            hidden_ratio=4,
        )
        
        model = HGRNBitForCausalLM(config).to(self.device)
        
        # Replace BitLinear layers if testing vanilla
        if not use_fused:
            self._replace_fused_with_vanilla(model)
        
        return model, config
    
    def _replace_fused_with_vanilla(self, model):
        """Replace FusedBitLinear with vanilla BitLinear for comparison"""
        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, FusedBitLinear):
                    # Create vanilla BitLinear with same dimensions
                    vanilla_layer = BitLinear(
                        child.in_features, 
                        child.out_features, 
                        bias=child.bias is not None
                    ).to(self.device)
                    setattr(module, name, vanilla_layer)
                else:
                    replace_layers(child)
        
        replace_layers(model)
    
    def benchmark_single_iteration(self, model, batch_size, seq_length=1024, warmup_steps=5, measure_steps=10):
        """Benchmark single training iteration - replicates paper methodology"""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch matching paper setup
        input_ids = torch.randint(0, 50000, (batch_size, seq_length)).to(self.device)
        
        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        
        # Clear cache and measure
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure training time
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(measure_steps):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        avg_time_per_iteration = (end_time - start_time) / measure_steps
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return avg_time_per_iteration, peak_memory_gb
    
    def run_batch_size_sweep(self, batch_sizes=[1, 2, 4, 8, 16, 28, 32], seq_length=1024):
        """Run the main benchmark sweep - replicates Figure 3(a-b)"""
        print("🚀 Starting Training Efficiency Benchmark")
        print(f"📊 Testing batch sizes: {batch_sizes}")
        print(f"📏 Sequence length: {seq_length}")
        print(f"🔧 Device: {self.results['metadata']['device']}")
        print("=" * 80)
        
        # Test Fused BitLinear (your implementation)
        print("\n🔥 Testing Fused BitLinear (Your Implementation)")
        fused_model, config = self.create_test_model(use_fused=True)
        print(f"📈 Model: {sum(p.numel() for p in fused_model.parameters())/1e9:.1f}B parameters")
        
        for batch_size in batch_sizes:
            try:
                print(f"  📦 Batch size {batch_size}...", end=" ")
                time_per_iter, memory_gb = self.benchmark_single_iteration(
                    fused_model, batch_size, seq_length
                )
                
                self.results['fused_bitlinear']['batch_sizes'].append(batch_size)
                self.results['fused_bitlinear']['times'].append(time_per_iter)
                self.results['fused_bitlinear']['memory'].append(memory_gb)
                
                print(f"⏱️  {time_per_iter:.3f}s/iter, 💾 {memory_gb:.1f}GB")
                
            except torch.cuda.OutOfMemoryError:
                print("❌ OOM")
                break
        
        # Clear memory
        del fused_model
        torch.cuda.empty_cache()
        
        # Test Vanilla BitLinear (baseline)
        print("\n📊 Testing Vanilla BitLinear (Baseline)")
        vanilla_model, _ = self.create_test_model(use_fused=False)
        
        for batch_size in batch_sizes:
            try:
                print(f"  📦 Batch size {batch_size}...", end=" ")
                time_per_iter, memory_gb = self.benchmark_single_iteration(
                    vanilla_model, batch_size, seq_length
                )
                
                self.results['vanilla_bitlinear']['batch_sizes'].append(batch_size)
                self.results['vanilla_bitlinear']['times'].append(time_per_iter)
                self.results['vanilla_bitlinear']['memory'].append(memory_gb)
                
                print(f"⏱️  {time_per_iter:.3f}s/iter, 💾 {memory_gb:.1f}GB")
                
            except torch.cuda.OutOfMemoryError:
                print("❌ OOM")
                break
        
        del vanilla_model
        torch.cuda.empty_cache()
    
    def calculate_improvements(self):
        """Calculate speedup and memory reduction"""
        improvements = {}
        
        # Find common batch sizes
        fused_batches = set(self.results['fused_bitlinear']['batch_sizes'])
        vanilla_batches = set(self.results['vanilla_bitlinear']['batch_sizes'])
        common_batches = fused_batches.intersection(vanilla_batches)
        
        for batch_size in common_batches:
            fused_idx = self.results['fused_bitlinear']['batch_sizes'].index(batch_size)
            vanilla_idx = self.results['vanilla_bitlinear']['batch_sizes'].index(batch_size)
            
            fused_time = self.results['fused_bitlinear']['times'][fused_idx]
            vanilla_time = self.results['vanilla_bitlinear']['times'][vanilla_idx]
            
            fused_memory = self.results['fused_bitlinear']['memory'][fused_idx]
            vanilla_memory = self.results['vanilla_bitlinear']['memory'][vanilla_idx]
            
            speedup = ((vanilla_time - fused_time) / vanilla_time) * 100
            memory_reduction = ((vanilla_memory - fused_memory) / vanilla_memory) * 100
            
            improvements[batch_size] = {
                'speedup_percent': speedup,
                'memory_reduction_percent': memory_reduction,
                'fused_time': fused_time,
                'vanilla_time': vanilla_time,
                'fused_memory': fused_memory,
                'vanilla_memory': vanilla_memory
            }
        
        return improvements
