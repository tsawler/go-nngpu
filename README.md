# go-nngpu

`github.com/tsawler/go-nngpu` is a Go module designed to leverage the power of Apple's GPUs for machines equipped with Apple silicon, for numerical computations. It is primarily focused on accelerating operations crucial for Machine Learning and integrating seamlessly with [gonum](https://www.gonum.org/). This module aims to replace performance-critical gonum functions and methods with GPU-enhanced counterparts, providing significant speedups for data-intensive tasks. GPU acceleration is achieved using a CGO bridge to Objective C code that uses Apple's [Metal API](https://developer.apple.com/metal/).

## Table of Contents

- [Features](#features)
- [Current Implementation Status](#current-implementation-status)
- [Installation](#installation)
- [Usage](#usage)
  - [GPU-Accelerated gonum Matrix Operations](#gpu-accelerated-gonum-matrix-operations)
  - [Working with GPUDense Matrices](#working-with-gpudense-matrices)
  - [Advanced Dense Matrix Decompositions](#advanced-dense-matrix-decompositions)
  - [Sparse Matrix Operations](#sparse-matrix-operations)
  - [Batch Operations](#batch-operations)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
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
- [ ] Phase 6A: Activation functions (ReLU, Sigmoid, Tanh, Softmax + derivatives)
- [ ] Phase 6B: Loss functions (CrossEntropy, MSE + gradients)
- [ ] Phase 6C: Convolution operations (Conv2D, MaxPool, padding)
- [ ] Phase 6D: Batch normalization (mean, variance, normalize)
- [ ] Phase 7A: Gradient computation framework
- [ ] Phase 7B: Optimizers (SGD, Adam, RMSprop with GPU state)
- [ ] Phase 7C: Memory-efficient training loop
- [ ] Phase 7D: Automatic differentiation helpers
- [ ] Phase 8A: Fused operations (activation + gradient, conv + activation)
- [ ] Phase 8B: Custom Metal kernels for specific operations
- [ ] Phase 8C: Memory bandwidth optimization
- [ ] Phase 9A: Mixed precision training (float16/float32)
- [ ] Phase 9B: Distributed training support
- [ ] Phase 9C: Advanced sparse training techniques

## Installation

To install go-nngpu, you need Go (version 1.18 or higher recommended) and a macOS environment with Metal support.

```bash
go get github.com/tsawler/go-nngpu
```

Since go-nngpu uses CGO to interface with Metal and Accelerate frameworks, you need to ensure your CGO environment is set up correctly. No additional external dependencies (like CUDA or OpenCL SDKs) are required beyond what's available on macOS.

## Usage

go-nngpu offers two primary ways to utilize GPU acceleration:

1. **Drop-in Replacements for gonum/mat functions**: Functions like GPUMatMul, GPUAdd, GPUSub, etc., directly accept gonum/mat.Matrix types and return gonum/mat.Dense matrices, handling the GPU transfer internally. This is the simplest way to get started.
2. **GPUDense and GPUSparse Types**: These types wrap go-nngpu's internal tensor.Tensor and tensor.SparseTensor types, providing more direct control over GPU memory and allowing for chained GPU operations without unnecessary CPU transfers. They also implement the gonum/mat.Matrix interface.

All internal GPU computations in go-nngpu operate on float32 data for performance reasons. Data is automatically converted from float64 to float32 when moved to the GPU and back to float64 when retrieved to the CPU for gonum compatibility.

### GPU-Accelerated gonum Matrix Operations

You can use the GPU* functions as direct replacements for gonum/mat operations.

```go
package main

import (
	"fmt"
	"github.com/tsawler/go-nngpu/gpu/matrix"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Create two gonum dense matrices
	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	b := mat.NewDense(3, 2, []float64{
		7, 8,
		9, 10,
		11, 12,
	})

	fmt.Println("Original Matrix A:\n", mat.Formatted(a))
	fmt.Println("Original Matrix B:\n", mat.Formatted(b))

	// Perform GPU-accelerated matrix multiplication
	resultMul := matrix.GPUMatMul(a, b)
	fmt.Println("\nGPU-accelerated A * B:\n", mat.Formatted(resultMul))

	// Perform GPU-accelerated element-wise addition
	c := mat.NewDense(2, 2, []float64{
		1, 1,
		1, 1,
	})
	d := mat.NewDense(2, 2, []float64{
		2, 2,
		2, 2,
	})
	resultAdd := matrix.GPUAdd(c, d)
	fmt.Println("\nGPU-accelerated C + D:\n", mat.Formatted(resultAdd))

	// Perform GPU-accelerated matrix inverse
	invMat := mat.NewDense(2, 2, []float64{
		4, 7,
		2, 6,
	})
	resultInv := matrix.GPUInverse(invMat)
	fmt.Println("\nGPU-accelerated Inverse of invMat:\n", mat.Formatted(resultInv))

	// Perform GPU-accelerated matrix determinant
	det := matrix.GPUDeterminant(invMat)
	fmt.Printf("\nGPU-accelerated Determinant of invMat: %.2f\n", det)
}
```

### Working with GPUDense Matrices

For more complex workflows or chained operations, it's more efficient to create GPUDense matrices directly, as they manage their data on the GPU, avoiding repeated transfers.

```go
package main

import (
	"fmt"
	"github.com/tsawler/go-nngpu/gpu/matrix"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Create a GPUDense matrix directly
	gpuA := matrix.NewGPUDense(2, 2, []float64{
		1.0, 2.0,
		3.0, 4.0,
	})
	defer gpuA.ReleaseGPU() // Important: Release GPU resources when done

	gpuB := matrix.NewGPUDense(2, 2, []float64{
		5.0, 6.0,
		7.0, 8.0,
	})
	defer gpuB.ReleaseGPU() // Important: Release GPU resources when done

	// Perform chained GPU operations
	// gpuC = gpuA * gpuB (in-place on gpuC's underlying tensor)
	gpuC := matrix.NewGPUDense(2, 2, make([]float64, 4))

	gpuC.Mul(gpuA, gpuB)
	fmt.Println("GPU-backed A * B:\n", mat.Formatted(gpuC))

	// gpuD = gpuC + gpuA (in-place on gpuD's underlying tensor)
	gpuD := matrix.NewGPUDense(2, 2, make([]float64, 4))
	gpuD.Add(gpuC, gpuA)
	fmt.Println("\nGPU-backed (A*B) + A:\n", mat.Formatted(gpuD))

	// Transpose operation
	gpuAT := gpuA.T()
	defer gpuAT.(*matrix.GPUDense).ReleaseGPU()
	fmt.Println("\nGPU-backed A Transposed:\n", mat.Formatted(gpuAT))

	// Convert back to gonum.Dense for interoperability (retrieves data to CPU)
	gonumD := gpuD.ToGonum()
	fmt.Println("\nConverted back to gonum.Dense:\n", mat.Formatted(gonumD))
}
```

### Advanced Dense Matrix Decompositions

go-nngpu provides GPU-accelerated versions of common matrix decompositions.

```go
package main

import (
	"fmt"
	"github.com/tsawler/go-nngpu/gpu/matrix"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Example for LU Decomposition
	a := mat.NewDense(3, 3, []float64{
		2, 1, 1,
		4, -2, 1,
		-2, 2, 1,
	})
	fmt.Println("Original Matrix for LU:\n", mat.Formatted(a))

	luDecomp := matrix.GPULU(a)

	fmt.Println("\nLU Decomposition - L:\n", mat.Formatted(luDecomp.L))
	fmt.Println("LU Decomposition - U:\n", mat.Formatted(luDecomp.U))
	fmt.Println("LU Decomposition - Pivot Indices:\n", luDecomp.PivotIndices)

	// Example for QR Decomposition
	b := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	fmt.Println("\nOriginal Matrix for QR:\n", mat.Formatted(b))

	qrDecomp := matrix.GPUQR(b)

	fmt.Println("\nQR Decomposition - Q:\n", mat.Formatted(qrDecomp.Q))
	fmt.Println("QR Decomposition - R:\n", mat.Formatted(qrDecomp.R))

	// Example for Cholesky Decomposition (must be symmetric positive definite)
	c := mat.NewDense(2, 2, []float64{
		25, 15,
		15, 18,
	})
	fmt.Println("\nOriginal Matrix for Cholesky:\n", mat.Formatted(c))

	choleskyL := matrix.GPUCholesky(c)

	fmt.Println("\nCholesky Decomposition - L:\n", mat.Formatted(choleskyL))

	// Example for Eigenvalue Decomposition (must be symmetric)
	e := mat.NewDense(3, 3, []float64{
		4, 1, 2,
		1, 3, 0,
		2, 0, 1,
	})
	fmt.Println("\nOriginal Matrix for Eigen:\n", mat.Formatted(e))

	eigenDecomp := matrix.GPUEigen(e)

	fmt.Println("\nEigenvalue Decomposition - Eigenvalues:\n", mat.Formatted(eigenDecomp.Eigenvalues))
	fmt.Println("Eigenvalue Decomposition - Eigenvectors:\n", mat.Formatted(eigenDecomp.Eigenvectors))

	// Example for Singular Value Decomposition (SVD)
	s := mat.NewDense(2, 3, []float64{
		1, 1, 0,
		0, 1, 1,
	})
	fmt.Println("\nOriginal Matrix for SVD:\n", mat.Formatted(s))

	svdDecomp := matrix.GPUSVD(s)

	fmt.Println("\nSVD - U:\n", mat.Formatted(svdDecomp.U))
	fmt.Println("SVD - S (Singular Values):\n", mat.Formatted(svdDecomp.S))
	fmt.Println("SVD - VT (V Transposed):\n", mat.Formatted(svdDecomp.VT))
}
```

### Sparse Matrix Operations

go-nngpu provides robust support for sparse matrices, crucial for efficiency in many ML applications.

```go
package main

import (
	"fmt"

	"github.com/tsawler/go-nngpu/gpu/matrix"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Original dense matrices
	denseA := mat.NewDense(4, 4, []float64{
		0, 0, 0, 5,
		0, 2, 0, 0,
		0, 0, 3, 0,
		4, 0, 0, 0,
	})
	fmt.Println("Original Dense Matrix A:\n", mat.Formatted(denseA))

	denseB := mat.NewDense(4, 4, []float64{
		10, 0, 0, 0,
		0, 20, 0, 0,
		0, 0, 30, 0,
		0, 0, 0, 40,
	})
	fmt.Println("\nOriginal Dense Matrix B:\n", mat.Formatted(denseB))

	// Convert to GPU sparse matrices
	sparseA := matrix.GPUDenseToSparse(denseA, 1e-9)
	defer sparseA.ReleaseGPU()
	fmt.Println("\nConverted to GPUSparse Matrix A info:", matrix.GetSparseMatrixInfo(sparseA))

	sparseB := matrix.GPUDenseToSparse(denseB, 1e-9)
	defer sparseB.ReleaseGPU()
	fmt.Println("\nConverted to GPUSparse Matrix B info:", matrix.GetSparseMatrixInfo(sparseB))

	// GPU sparse * sparse
	resultSparseMul := matrix.GPUSparseMatMul(sparseA, sparseB)
	fmt.Println("\nGPU Sparse A * Sparse B (Resulting Dense):\n", mat.Formatted(resultSparseMul))

	// ✅ CPU-side fallback: add denseA + denseB
	sumDense := mat.NewDense(4, 4, nil)
	sumDense.Add(denseA, denseB)
	fmt.Println("\nWorkaround: Dense A + Dense B (on CPU):\n", mat.Formatted(sumDense))

	// Sparse scalar multiplication
	resultScalarMul := sparseA.SparseScalarMul(2.0)
	defer resultScalarMul.ReleaseGPU()
	fmt.Println("\nGPU Sparse A * 2.0 (Resulting Sparse):\n", resultScalarMul)

	// Sparse Matrix-Vector Multiplication
	vec := []float64{1, 1, 1, 1}
	resultMatVec := sparseA.SparseMatVec(vec)
	fmt.Println("\nGPU Sparse A * Vector [1,1,1,1]:\n", resultMatVec)

	// Sparse matrix usefulness
	rows, cols := 1000, 1000
	nnz := 50000
	nnzDense := 800000
	fmt.Printf("\nIs sparse worthwhile for %dx%d matrix with %d non-zeros? %v\n", rows, cols, nnz, matrix.IsSparseWorthwhile(rows, cols, nnz))
	fmt.Printf("Is sparse worthwhile for %dx%d matrix with %d non-zeros? %v\n", rows, cols, nnzDense, matrix.IsSparseWorthwhile(rows, cols, nnzDense))
}
```

### Batch Operations

For scenarios involving many independent matrix operations, go-nngpu provides batch functions to optimize GPU utilization by reducing CPU-GPU data transfer overhead.

```go
package main

import (
	"fmt"
	"github.com/tsawler/go-nngpu/gpu/matrix"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Batch Matrix Multiplication
	m1 := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	m2 := mat.NewDense(2, 2, []float64{5, 6, 7, 8})
	m3 := mat.NewDense(2, 2, []float64{9, 10, 11, 12})

	batchMulOps := []struct{ A, B mat.Matrix }{
		{A: m1, B: m2},
		{A: m2, B: m3},
	}

	resultsMul := matrix.BatchGPUMatMul(batchMulOps)
	fmt.Println("Batch Matrix Multiplication Results:")
	for i, res := range resultsMul {
		fmt.Printf("Operation %d:\n%v\n", i+1, mat.Formatted(res))
	}

	// Batch Sparse Matrix-Vector Multiplication
	sparseM1 := matrix.NewGPUSparse(2, 3, []int32{0, 1}, []int32{1, 0}, []float32{10, 20})
	sparseM2 := matrix.NewGPUSparse(3, 2, []int32{0, 2}, []int32{1, 0}, []float32{5, 15})

	vec1 := []float64{1, 2, 3}
	vec2 := []float64{4, 5}

	batchMatVecOps := []*matrix.GPUSparse{sparseM1, sparseM2}
	batchVectors := [][]float64{vec1, vec2}

	resultsMatVec := matrix.BatchGPUSparseMatVec(batchMatVecOps, batchVectors)
	fmt.Println("\nBatch Sparse Matrix-Vector Multiplication Results:")
	for i, res := range resultsMatVec {
		fmt.Printf("Operation %d: %v\n", i+1, res)
	}
}
```

## Performance Considerations

- **float32 vs float64**: All GPU computations within go-nngpu utilize float32 for optimal performance on GPUs. When using gonum/mat.Matrix (which typically uses float64), data is converted to float32 before being sent to the GPU and back to float64 upon retrieval. For performance-critical applications, consider managing data in float32 format directly where possible.
- **CPU-GPU Data Transfer**: The primary bottleneck in GPU acceleration is often the transfer of data between the CPU and GPU.
  - **Minimize Transfers**: Use GPUDense or GPUSparse types for chained operations to keep data on the GPU for as long as possible.
  - **Batch Operations**: For multiple independent operations, use the provided BatchGPU* functions to amortize transfer costs.
- **Sparse vs. Dense**: While sparse matrices can offer significant memory and computation savings for sparse data, there is overhead associated with managing their sparse structure. The IsSparseWorthwhile utility function can help determine if a matrix's sparsity justifies using sparse storage.
- **runtime.LockOSThread()**: The CGO calls to Metal and Accelerate often require the current goroutine to be locked to an OS thread. This is handled internally by go-nngpu functions for correctness.

## Contributing

Contributions are welcome! If you're interested in contributing to go-nngpu, please feel free to open an issue or pull request on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.