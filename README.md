# MatMul-Free Language Model for Multi-GPU Training

This project presents an implementation of the MatMul-Free Language Model (MatMul-Free LM), adapted for efficient multi-GPU training and deployment on AWS SageMaker. The codebase is designed to leverage modern deep learning infrastructure.

## Background

The implementation is based on the paper "Scalable MatMul-free Language Modeling" (arXiv:2406.02528). This research introduces a novel language model architecture that eliminates traditional Matrix Multiplication (MatMul) operations, which are typically a major source of computational and memory overhead in large language models (LLMs). By replacing MatMul with more efficient additive operations and element-wise products, and utilizing ternary weights, the model aims for significant memory savings and improved throughput, especially on specialized hardware.

## Features

*   **Multi-GPU Distributed Training:** Configured for distributed training using PyTorch's `DistributedDataParallel` (DDP).
*   **AWS SageMaker Compatibility:** Project structure and `Dockerfile` are designed for seamless integration with AWS SageMaker training jobs.
*   **Hugging Face Transformers Integration:** The model architecture (`HGRNBitForCausalLM`, `HGRNBitConfig`) is compatible with the Hugging Face Transformers library.
*   **Triton-Based Custom Operations:** Core MatMul-free linear layers (`FusedBitLinear`) and recurrent attention mechanisms (fused HGRN operations) are implemented using Triton for optimized performance on CUDA-enabled GPUs.
*   **Data Handling:** Includes a basic data loading and preprocessing pipeline using the `wikitext` dataset.

## Directory Structure

```
matmulMM/
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── layers/
│   │   ├── __init__.py
│   │   └── hgrn_bit.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── hgrn_bit/
│   │       ├── __init__.py
│   │       ├── configuration_hgrn_bit.py
│   │       ├── modeling_hgrn_bit.py
│   │       └── modeling_hgrn_bit_nonorm.py
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── bitnet.py
│   │   ├── fusedbitnet.py
│   │   ├── hgrn/
│   │   │   ├── __init__.py
│   │   │   ├── chunk.py
│   │   │   ├── naive.py
│   │   │   └── recurrent_fuse.py
│   │   └── utils.py
│   └── utils.py
└── train.py
```

## Setup and Installation

### Prerequisites

Ensure the following are installed on your system or target environment (e.g., AWS EC2 instance, SageMaker instance):

*   Python 3.8+
*   PyTorch 2.0+ (with CUDA support)
*   NVIDIA GPU with CUDA 11.7+
*   Docker (for containerized deployment)
*   Git

### Cloning the Repository

```bash
git clone https://github.com/tonymindbeam/matmulMM.git
cd matmulMM
```

### Installing Dependencies

The project dependencies are listed in `requirements.txt`. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```
Note: Some dependencies, particularly Triton, may require specific CUDA versions or nightly builds. Refer to their official documentation for detailed installation instructions if issues arise.

## Usage

### Building the Docker Image

The provided `Dockerfile` sets up the environment for training. Build the Docker image using:

```bash
docker build -t matmulfreelm-training .
```

### Running the Training Script

#### Local Execution (with Docker)

To run the training script locally within the Docker container (assuming you have GPUs and Docker configured):

```bash
docker run --gpus all matmulfreelm-training
```

#### AWS SageMaker

For training on AWS SageMaker, the following general steps apply:

1.  **Upload Code to S3:** Package your `matmulMM` directory (e.g., as a `.tar.gz` file) and upload it to an Amazon S3 bucket.
2.  **Push Docker Image to ECR (Optional but Recommended):** If you built a custom Docker image, push it to Amazon Elastic Container Registry (ECR). Otherwise, you can use a pre-built SageMaker PyTorch image.
3.  **Create a SageMaker Training Job:** Configure a SageMaker training job, specifying:
    *   The S3 URI of your packaged code.
    *   The Docker image URI (from ECR or a pre-built SageMaker image).
    *   The desired GPU instance type (e.g., `ml.p3.2xlarge`, `ml.g4dn.xlarge`) and instance count for distributed training.
    *   The entry point script (`train.py`).

SageMaker will handle provisioning the instances, setting up the environment, and executing the `train.py` script for distributed training.

## Custom Operations

The `src/ops` directory contains custom implementations of key components, including `FusedBitLinear` and fused HGRN operations. These are written using Triton, a Python-based DSL for writing highly efficient custom CUDA kernels. This approach allows for significant performance gains by fusing multiple operations and optimizing memory access patterns, crucial for the MatMul-free architecture.

## Citation

If this work is used, please cite the original preprint:

```bib
@article{zhu2024scalable,
title={Scalable MatMul-free Language Modeling},
author={Zhu, Rui-Jie and Zhang, Yu and Sifferman, Ethan and Sheaves, Tyler and Wang, Yiqiao and Richmond, Dustin and Zhou, Peng and Eshraghian, Jason K},
journal={arXiv preprint arXiv:2406.02528},
year={2024}
}
```