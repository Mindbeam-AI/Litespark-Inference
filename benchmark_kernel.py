#!/usr/bin/env python3
"""Compatibility shim.

The benchmark now ships inside the installed package so it works from a
plain `pip install litespark-inference` (no clone required). Prefer:

    litespark-benchmark --inference --pytorch --no-matrix
    python -m litespark_inference.benchmark_kernel --inference --pytorch

This shim keeps `python benchmark_kernel.py ...` working from a checkout.
"""
from litespark_inference.benchmark_kernel import main

if __name__ == "__main__":
    main()
