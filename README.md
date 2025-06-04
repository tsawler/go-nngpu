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

## Installation

To install go-nngpu, you need Go (version 1.23 or higher recommended) and a macOS environment with Metal support.

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
	"math/rand"
	"time"

	"github.com/tsawler/go-nngpu/gpu/matrix"

	// Standard Gonum imports
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=======================================================")
	fmt.Println("=== Sparse Matrix Library vs Gonum Demonstration ===")
	fmt.Println("=======================================================")

	// Demo 1: Basic matrix operations comparison
	fmt.Println("\n--- Demo 1: Basic Matrix Operations ---")
	demoBasicOperations()

	// Demo 2: Performance comparison on large sparse matrices
	fmt.Println("\n--- Demo 2: Performance Comparison ---")
	demoPerformanceComparison()

	// Demo 3: Mixed operations (sparse + dense)
	fmt.Println("\n--- Demo 3: Mixed Sparse/Dense Operations ---")
	demoMixedOperations()

	// Demo 4: Real-world scenario - solving a sparse system
	fmt.Println("\n--- Demo 4: Real-world Sparse System ---")
	demoRealWorldScenario()
}

func demoBasicOperations() {
	// Create a sparse connectivity matrix (like a graph adjacency matrix)
	// This represents connections between 6 nodes
	fmt.Println("Creating a 6x6 sparse connectivity matrix...")

	// === USING THIS SPARSE LIBRARY ===
	// Create sparse matrix directly with COO format
	rowIndices := []int32{0, 0, 1, 1, 2, 3, 4, 4, 5}
	colIndices := []int32{1, 3, 2, 4, 5, 0, 1, 5, 2}
	values := []float32{1.5, 2.0, 3.2, 1.8, 2.5, 1.0, 2.2, 3.0, 1.7}

	sparseMat := matrix.NewGPUSparse(6, 6, rowIndices, colIndices, values)
	defer sparseMat.ReleaseGPU()

	sparseRows, sparseCols := sparseMat.Dims()
	fmt.Printf("Sparse matrix: %dx%d, NNZ=%d, Density=%.3f\n",
		sparseRows, sparseCols, sparseMat.GetNNZ(), sparseMat.GetDensity())

	// Convert to dense for display
	denseMat := sparseMat.ToDense()
	fmt.Println("Sparse matrix as dense:")
	printMatrix(denseMat)

	// === USING SPARSE LIBRARY FOR OPERATIONS ===
	// Scalar multiplication
	fmt.Println("\nScalar multiplication (×2.0) using sparse library:")
	start := time.Now()
	scaledSparse := sparseMat.SparseScalarMul(2.0) // <-- THIS SPARSE LIBRARY
	sparseScalarTime := time.Since(start)
	defer scaledSparse.ReleaseGPU()

	scaledDense := scaledSparse.ToDense()
	printMatrix(scaledDense)
	fmt.Printf("Sparse scalar multiplication took: %v\n", sparseScalarTime)

	// Compare with Gonum dense operation
	fmt.Println("\nSame operation using Gonum dense:")
	start = time.Now()
	gonumResult := mat.NewDense(6, 6, nil)
	gonumResult.Scale(2.0, denseMat) // <-- STANDARD GONUM
	gonumScalarTime := time.Since(start)
	fmt.Printf("Gonum dense scalar multiplication took: %v\n", gonumScalarTime)

	// Matrix-vector multiplication
	fmt.Println("\nMatrix-vector multiplication:")
	vector := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	// === USING THIS SPARSE LIBRARY ===
	start = time.Now()
	sparseResult := sparseMat.SparseMatVec(vector) // <-- THIS SPARSE LIBRARY
	sparseMatVecTime := time.Since(start)

	fmt.Printf("Sparse result: %v\n", sparseResult)
	fmt.Printf("Sparse matvec took: %v\n", sparseMatVecTime)

	// Compare with Gonum
	start = time.Now()
	gonumVec := mat.NewVecDense(6, vector)
	gonumResult2 := mat.NewVecDense(6, nil)
	gonumResult2.MulVec(denseMat, gonumVec) // <-- STANDARD GONUM
	gonumMatVecTime := time.Since(start)

	fmt.Printf("Gonum result: %v\n", gonumResult2.RawVector().Data)
	fmt.Printf("Gonum matvec took: %v\n", gonumMatVecTime)
}

func demoPerformanceComparison() {
	// Create larger sparse matrices for performance testing
	size := 1000
	sparsity := 0.02 // 2% non-zero elements

	fmt.Printf("Creating %dx%d matrices with %.1f%% sparsity...\n", size, size, sparsity*100)

	// Generate random sparse data
	nnz := int(float64(size*size) * sparsity)
	rowIndices, colIndices, values := generateRandomSparseData(size, size, nnz)

	// === CREATE USING THIS SPARSE LIBRARY ===
	fmt.Println("Creating sparse matrix using this library...")
	start := time.Now()
	sparseMat := matrix.NewGPUSparse(size, size, rowIndices, colIndices, values) // <-- THIS SPARSE LIBRARY
	defer sparseMat.ReleaseGPU()
	sparseCreateTime := time.Since(start)

	// Create equivalent dense matrix using Gonum
	fmt.Println("Creating equivalent dense matrix using Gonum...")
	start = time.Now()
	denseData := make([]float64, size*size)
	for i := 0; i < nnz; i++ {
		row, col := int(rowIndices[i]), int(colIndices[i])
		denseData[row*size+col] = float64(values[i])
	}
	denseMat := mat.NewDense(size, size, denseData) // <-- STANDARD GONUM
	denseCreateTime := time.Since(start)

	fmt.Printf("Sparse matrix creation: %v\n", sparseCreateTime)
	fmt.Printf("Dense matrix creation: %v\n", denseCreateTime)

	// Test matrix-vector multiplication performance
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rand.Float64()
	}

	// === USING THIS SPARSE LIBRARY ===
	fmt.Println("\nTesting matrix-vector multiplication performance...")
	start = time.Now()
	sparseResult := sparseMat.SparseMatVec(vector) // <-- THIS SPARSE LIBRARY
	sparseTime := time.Since(start)

	// Using Gonum dense
	start = time.Now()
	vecGonum := mat.NewVecDense(size, vector)
	resultGonum := mat.NewVecDense(size, nil)
	resultGonum.MulVec(denseMat, vecGonum) // <-- STANDARD GONUM
	denseTime := time.Since(start)

	fmt.Printf("Sparse matvec time: %v\n", sparseTime)
	fmt.Printf("Dense matvec time: %v\n", denseTime)

	// Verify results match
	gonumData := resultGonum.RawVector().Data
	match := true
	maxDiff := 0.0
	tolerance := 1e-4 // More relaxed tolerance for float32 vs float64 comparison

	for i := range sparseResult {
		diff := abs(sparseResult[i] - gonumData[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tolerance {
			match = false
			// Don't break immediately - let's see the max difference
		}
	}

	fmt.Printf("Maximum difference between results: %.2e\n", maxDiff)

	if match {
		speedup := float64(denseTime) / float64(sparseTime)
		fmt.Printf("✓ Results match! Sparse was %.2fx the speed of dense\n", speedup)
	} else {
		speedup := float64(denseTime) / float64(sparseTime)
		fmt.Printf("⚠ Results differ by %.2e (likely float32 vs float64 precision)\n", maxDiff)
		fmt.Printf("  Sparse was %.2fx the speed of dense\n", speedup)

		// Show a few sample comparisons for debugging
		fmt.Println("  Sample comparisons:")
		for i := 0; i < 5 && i < len(sparseResult); i++ {
			fmt.Printf("    [%d] Sparse: %.6f, Dense: %.6f, Diff: %.2e\n",
				i, sparseResult[i], gonumData[i], abs(sparseResult[i]-gonumData[i]))
		}
	}

	// Memory usage comparison
	sparseMemory := estimateMemoryUsage(sparseMat)
	denseMemory := int64(size * size * 8) // float64 = 8 bytes

	fmt.Printf("Sparse memory: ~%d bytes\n", sparseMemory)
	fmt.Printf("Dense memory: %d bytes\n", denseMemory)
	fmt.Printf("Memory savings: %.2fx\n", float64(denseMemory)/float64(sparseMemory))
}

func demoMixedOperations() {
	// Demonstrate mixing sparse and dense operations
	fmt.Println("Creating sparse matrix A and dense matrix B...")

	// === CREATE SPARSE MATRIX USING THIS LIBRARY ===
	// Sparse matrix A (4x3)
	aRowIndices := []int32{0, 0, 1, 2, 3, 3}
	aColIndices := []int32{0, 2, 1, 0, 1, 2}
	aValues := []float32{2.0, 3.0, 4.0, 1.0, 5.0, 2.0}

	sparseA := matrix.NewGPUSparse(4, 3, aRowIndices, aColIndices, aValues) // <-- THIS SPARSE LIBRARY
	defer sparseA.ReleaseGPU()

	// Dense matrix B (3x2) using Gonum
	bData := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	denseB := mat.NewDense(3, 2, bData) // <-- STANDARD GONUM

	fmt.Println("Sparse matrix A:")
	printMatrix(sparseA.ToDense())

	fmt.Println("Dense matrix B:")
	printMatrix(denseB)

	// === MIXED MULTIPLICATION USING THIS LIBRARY ===
	fmt.Println("Performing sparse × dense multiplication using this library...")
	start := time.Now()
	result := matrix.GPUSparseMatMul(sparseA, denseB) // <-- THIS SPARSE LIBRARY (mixed operation)
	mixedTime := time.Since(start)

	fmt.Println("Result (Sparse × Dense):")
	printMatrix(result)
	fmt.Printf("Mixed operation took: %v\n", mixedTime)

	// Compare with pure Gonum dense operations
	fmt.Println("\nCompare with pure Gonum dense operations...")
	denseA := sparseA.ToDense()
	start = time.Now()
	gonumResult := mat.NewDense(4, 2, nil)
	gonumResult.Mul(denseA, denseB) // <-- STANDARD GONUM
	gonumTime := time.Since(start)

	fmt.Println("Gonum result:")
	printMatrix(gonumResult)
	fmt.Printf("Gonum operation took: %v\n", gonumTime)
}

func demoRealWorldScenario() {
	// Simulate a real-world scenario: web page ranking (simplified PageRank)
	fmt.Println("Simulating web page ranking with sparse link matrix...")

	numPages := 100
	linksPerPage := 3 // Average links per page (sparse!)

	// === CREATE SPARSE LINK MATRIX USING THIS LIBRARY ===
	// Generate random web link structure
	rand.Seed(42) // For reproducible results
	var rowIndices, colIndices []int32
	var values []float32

	for page := 0; page < numPages; page++ {
		// Each page links to a few other pages
		numLinks := rand.Intn(linksPerPage*2) + 1
		for link := 0; link < numLinks; link++ {
			targetPage := rand.Intn(numPages)
			if targetPage != page { // No self-links
				rowIndices = append(rowIndices, int32(page))
				colIndices = append(colIndices, int32(targetPage))
				values = append(values, 1.0) // Equal weight for all links
			}
		}
	}

	linkMatrix := matrix.NewGPUSparse(numPages, numPages, rowIndices, colIndices, values) // <-- THIS SPARSE LIBRARY
	defer linkMatrix.ReleaseGPU()

	linkRows, linkCols := linkMatrix.Dims()
	fmt.Printf("Created link matrix: %dx%d with %d links (%.2f%% sparse)\n",
		linkRows, linkCols, linkMatrix.GetNNZ(), linkMatrix.GetDensity()*100)

	// Normalize the matrix (each row sums to 1)
	fmt.Println("Normalizing link matrix using sparse operations...")

	// === USING THIS SPARSE LIBRARY FOR NORMALIZATION ===
	// This would typically require custom sparse operations, but we'll simulate
	// by working with the dense version for this demo
	denseLinkMatrix := linkMatrix.ToDense()

	// Normalize rows
	rows, cols := denseLinkMatrix.Dims()
	for i := 0; i < rows; i++ {
		rowSum := 0.0
		for j := 0; j < cols; j++ {
			rowSum += denseLinkMatrix.At(i, j)
		}
		if rowSum > 0 {
			for j := 0; j < cols; j++ {
				if denseLinkMatrix.At(i, j) > 0 {
					denseLinkMatrix.Set(i, j, denseLinkMatrix.At(i, j)/rowSum)
				}
			}
		}
	}

	// Convert back to sparse for efficient operations
	normalizedSparse := matrix.NewGPUSparseFromDense(denseLinkMatrix, 1e-10) // <-- THIS SPARSE LIBRARY
	defer normalizedSparse.ReleaseGPU()

	// Simulate PageRank iteration using sparse matrix-vector multiplication
	fmt.Println("Running simplified PageRank iterations...")

	// Initial page rank vector (uniform distribution)
	pageRank := make([]float64, numPages)
	for i := range pageRank {
		pageRank[i] = 1.0 / float64(numPages)
	}

	// Run a few PageRank iterations
	dampingFactor := 0.85
	iterations := 10

	start := time.Now()
	for iter := 0; iter < iterations; iter++ {
		// === USING THIS SPARSE LIBRARY FOR PAGERANK ===
		newRank := normalizedSparse.SparseMatVec(pageRank) // <-- THIS SPARSE LIBRARY

		// Apply damping factor
		for i := range newRank {
			newRank[i] = (1.0-dampingFactor)/float64(numPages) + dampingFactor*newRank[i]
		}

		pageRank = newRank
	}
	pageRankTime := time.Since(start)

	fmt.Printf("PageRank completed in %v (%d iterations)\n", pageRankTime, iterations)

	// Find top-ranked pages
	type pageScore struct {
		page  int
		score float64
	}

	var scores []pageScore
	for i, score := range pageRank {
		scores = append(scores, pageScore{i, score})
	}

	// Sort to find top pages (simple bubble sort for demo)
	for i := 0; i < len(scores)-1; i++ {
		for j := 0; j < len(scores)-i-1; j++ {
			if scores[j].score < scores[j+1].score {
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}

	fmt.Println("Top 5 ranked pages:")
	for i := 0; i < 5 && i < len(scores); i++ {
		fmt.Printf("  Page %d: %.6f\n", scores[i].page, scores[i].score)
	}

	// Memory usage analysis
	info := matrix.GetSparseMatrixInfo(normalizedSparse)
	fmt.Printf("\nSparse matrix analysis:\n%s\n", info.String())

	denseMemory := int64(numPages * numPages * 8) // float64
	fmt.Printf("Memory savings vs dense: %.1fx\n",
		float64(denseMemory)/float64(info.MemoryUsage))
}

// Helper functions

func generateRandomSparseData(rows, cols, nnz int) ([]int32, []int32, []float32) {
	var rowIndices, colIndices []int32
	var values []float32

	used := make(map[int]bool)
	for len(rowIndices) < nnz {
		pos := rand.Intn(rows * cols)
		if !used[pos] {
			used[pos] = true
			row := pos / cols
			col := pos % cols
			val := rand.Float32()*10 - 5 // Random value between -5 and 5

			rowIndices = append(rowIndices, int32(row))
			colIndices = append(colIndices, int32(col))
			values = append(values, val)
		}
	}

	return rowIndices, colIndices, values
}

func estimateMemoryUsage(sparse *matrix.GPUSparse) int64 {
	info := matrix.GetSparseMatrixInfo(sparse)
	return info.MemoryUsage
}

func printMatrix(m mat.Matrix) {
	rows, cols := m.Dims()
	if rows > 8 || cols > 8 {
		fmt.Printf("Matrix too large to display (%dx%d)\n", rows, cols)
		return
	}

	for i := 0; i < rows; i++ {
		fmt.Print("[")
		for j := 0; j < cols; j++ {
			fmt.Printf("%6.2f", m.At(i, j))
			if j < cols-1 {
				fmt.Print(" ")
			}
		}
		fmt.Println("]")
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
```

### Activation functions

The program below demonstrates how ReLU activation function transforms the input by setting all negative values to zero while keeping positive values unchanged. It's a minimal example showing the GPU-accelerated activation function in action.

```go
package main

import (
	"fmt"
	"github.com/tsawler/go-nngpu/matrix"
	"github.com/tsawler/go-nngpu/tensor"
)

func main() {
	// Create input data
	data := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	input, _ := tensor.NewTensor([]int{1, 5}, data)
	defer input.ReleaseGPU()

	// Apply ReLU activation
	output, err := matrix.ActivationForward(input, matrix.ReLU, 0.0)
	if err != nil {
		panic(err)
	}
	defer output.ReleaseGPU()

	// Get results
	output.RetrieveCPU()

	// Print results
	fmt.Printf("Input:  %v\n", data)
	fmt.Printf("ReLU:   %v\n", output.Data)
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