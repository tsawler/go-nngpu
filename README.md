# GoMetal

**GoMetal** is a Go module designed to leverage the power of Apple's GPUs for machines equipped with Apple silicon, for numerical computations. It is primarily focused on accelerating operations crucial for Machine Learning and integrating seamlessly with [gonum](https://www.gonum.org/). This module aims to replace performance-critical gonum functions and methods with GPU-enhanced counterparts, providing significant speedups for data-intensive tasks. GPU acceleration is achieved using a CGO bridge to Objective C code that uses Apple's [Metal API](https://developer.apple.com/metal/).

**This project is under heavy development.**
## Table of Contents

- [Features](#features)
- [Current Implementation Status](#current-implementation-status)
- [Installation](#installation)
- [License](#license)

## Features

go-nngpu provides GPU-accelerated implementations for various matrix and tensor operations, offering both direct gonum replacements and native GPU-backed types.

Key features include:

- **gonum Compatibility**: Introduces GPUDense and GPUSparse types that implement gonum's mat.Matrix interface, allowing existing gonum code to be easily migrated to use GPU acceleration.
- **Core Matrix Operations**: GPU-accelerated matrix multiplication, element-wise addition, subtraction, multiplication (Hadamard product), division, and scalar operations.
- **Advanced Matrix Decompositions**: GPU-accelerated functions for matrix inverse, determinant, LU decomposition, QR decomposition, Cholesky decomposition, eigenvalue decomposition (for symmetric matrices), and Singular Value Decomposition (SVD).
- **Sparse Matrix Support**: Efficient handling and GPU-accelerated operations for sparse matrices, including sparse-dense, dense-sparse, and sparse-sparse multiplication, addition, scalar multiplication, matrix-vector multiplication, and conversions between dense and sparse formats.
- **Batch Operations**: Functions to perform multiple GPU operations in batches, minimizing data transfer overhead and maximizing GPU utilization.
- **Metal Performance Shaders (MPS) Integration**: Leverages Apple's Metal Performance Shaders for highly optimized GPU computations on macOS and iOS devices.
- **Accelerate Framework Integration**: Utilizes Apple's Accelerate framework for certain numerical linear algebra routines where efficient CPU or specialized hardware (e.g., Apple Neural Engine) implementations are available.
- **Tensor Management**: Underlying tensor package for efficient GPU memory management and data transfer between CPU and GPU.

## Current Implementation Status

The development of go-nngpu follows an incremental strategy. The current status is:

- [x] Phase 1: MatMul + Transpose (working now)
- [x] Phase 2: Add element-wise operations (Add, Sub, Mul, Div)
- [x] Phase 3: Add matrix inverse using Accelerate framework fallback
- [x] Phase 4: Add decompositions: QR, Cholesky, Eigenvalue, SVD, and LU
- [x] Phase 5: Add sparse matrix support
- [x] Phase 6A: Activation functions (ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU, ELU, Swish, GELU)
- [x] Phase 6B: Loss functions (CrossEntropy, MSE + gradients)
- [x] Phase 6C: Convolution operations (Conv2D, MaxPool, padding)
- [x] Phase 6D: Batch normalization (mean, variance, normalize)
- [x] Phase 7A: Gradient computation framework
- [x] Phase 7B: Optimizers (SGD, Adam, RMSprop with GPU state)
- [x] Phase 7C: Memory-efficient training loop
- [x] Phase 7D: Automatic differentiation helpers
- [x] Phase 8A: Fused operations (activation + gradient, conv + activation)
- [x] Phase 8B: Custom Metal kernels for specific operations
- [x] Phase 8C: Memory bandwidth optimization
- [x] Phase 9A: Mixed precision training (float16/float32)
- [ ] Phase 9B: Distributed training support
- [ ] Phase 9C: Advanced sparse training techniques

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.