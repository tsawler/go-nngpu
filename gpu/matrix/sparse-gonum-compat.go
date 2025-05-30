package matrix

import (
	"fmt"
	"math"

	"github.com/tsawler/go-nngpu/tensor"
	"gonum.org/v1/gonum/mat"
)

// GPUSparse wraps a sparse tensor to implement gonum's mat.Matrix interface
type GPUSparse struct {
	sparse *tensor.SparseTensor
}

// NewGPUSparse creates a new GPU-backed sparse matrix compatible with Gonum
func NewGPUSparse(rows, cols int, rowIndices, colIndices []int32, values []float32) *GPUSparse {
	sparseTensor, err := tensor.NewSparseTensorCOO([]int{rows, cols}, rowIndices, colIndices, values)
	if err != nil {
		panic(err) // In production, handle this more gracefully
	}

	return &GPUSparse{sparse: sparseTensor}
}

// NewGPUSparseFromDense creates a sparse matrix from a dense gonum matrix
func NewGPUSparseFromDense(m mat.Matrix, threshold float64) *GPUSparse {
	rows, cols := m.Dims()
	var rowIndices, colIndices []int32
	var values []float32

	// Extract non-zero elements
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := m.At(i, j)
			if math.Abs(val) > threshold {
				rowIndices = append(rowIndices, int32(i))
				colIndices = append(colIndices, int32(j))
				values = append(values, float32(val))
			}
		}
	}

	return NewGPUSparse(rows, cols, rowIndices, colIndices, values)
}

// FromGonumSparse converts a gonum sparse matrix to GPUSparse
// Note: This is a placeholder - in a full implementation, you'd support
// various gonum sparse matrix types when they become available
func FromGonumSparse(rows, cols int, data []float64, threshold float64) *GPUSparse {
	var rowIndices, colIndices []int32
	var values []float32

	// Extract non-zero elements from the data
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := data[i*cols+j]
			if math.Abs(val) > threshold {
				rowIndices = append(rowIndices, int32(i))
				colIndices = append(colIndices, int32(j))
				values = append(values, float32(val))
			}
		}
	}

	return NewGPUSparse(rows, cols, rowIndices, colIndices, values)
}

// Implement mat.Matrix interface
func (gs *GPUSparse) Dims() (r, c int) {
	return gs.sparse.Shape[0], gs.sparse.Shape[1]
}

func (gs *GPUSparse) At(i, j int) float64 {
	// This is inefficient for sparse matrices, but required for interface compliance
	// In practice, users should access sparse data through specialized methods
	
	rows, cols := gs.Dims()
	if i < 0 || i >= rows || j < 0 || j >= cols {
		panic("matrix index out of range")
	}
	
	switch gs.sparse.Format {
	case tensor.COO:
		for k := 0; k < gs.sparse.NNZ; k++ {
			if int(gs.sparse.RowIndices[k]) == i && int(gs.sparse.ColIndices[k]) == j {
				return float64(gs.sparse.Values[k])
			}
		}
	case tensor.CSR:
		start, end := gs.sparse.RowPtr[i], gs.sparse.RowPtr[i+1]
		for k := start; k < end; k++ {
			if int(gs.sparse.Indices[k]) == j {
				return float64(gs.sparse.Data[k])
			}
		}
	case tensor.CSC:
		start, end := gs.sparse.ColPtr[j], gs.sparse.ColPtr[j+1]
		for k := start; k < end; k++ {
			if int(gs.sparse.Indices[k]) == i {
				return float64(gs.sparse.Data[k])
			}
		}
	}
	
	return 0.0 // Element not found, return zero
}

func (gs *GPUSparse) T() mat.Matrix {
	transposed, err := SparseTranspose(gs.sparse)
	if err != nil {
		panic(err)
	}
	return &GPUSparse{sparse: transposed}
}

// Sparse-specific methods

// GetNNZ returns the number of non-zero elements
func (gs *GPUSparse) GetNNZ() int {
	return gs.sparse.GetNNZ()
}

// GetDensity returns the density (sparsity ratio) of the matrix
func (gs *GPUSparse) GetDensity() float64 {
	return gs.sparse.GetDensity()
}

// GetFormat returns the current storage format
func (gs *GPUSparse) GetFormat() tensor.SparseFormat {
	return gs.sparse.GetFormat()
}

// ConvertToCSR converts the sparse matrix to CSR format
func (gs *GPUSparse) ConvertToCSR() error {
	return gs.sparse.ConvertToCSR()
}

// ConvertToCSC converts the sparse matrix to CSC format
func (gs *GPUSparse) ConvertToCSC() error {
	return gs.sparse.ConvertToCSC()
}

// GPU-accelerated sparse matrix operations

// SparseMul performs sparse matrix multiplication
func (gs *GPUSparse) SparseMul(a, b interface{}) *mat.Dense {
	result, err := SparseMatMul(a, b)
	if err != nil {
		panic(err)
	}
	defer result.ReleaseGPU()

	// Convert back to gonum format
	if err := result.RetrieveCPU(); err != nil {
		panic(err)
	}

	rows, cols := result.Shape[0], result.Shape[1]
	data := make([]float64, len(result.Data))
	for i, v := range result.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

// SparseAdd performs sparse matrix addition
func (gs *GPUSparse) SparseAdd(other *GPUSparse) *GPUSparse {
	result, err := SparseAdd(gs.sparse, other.sparse)
	if err != nil {
		panic(err)
	}
	return &GPUSparse{sparse: result}
}

// SparseScalarMul performs scalar multiplication
func (gs *GPUSparse) SparseScalarMul(scalar float64) *GPUSparse {
	result, err := SparseScalarMul(gs.sparse, float32(scalar))
	if err != nil {
		panic(err)
	}
	return &GPUSparse{sparse: result}
}

// SparseMatVec performs sparse matrix-vector multiplication
func (gs *GPUSparse) SparseMatVec(x []float64) []float64 {
	// Convert input vector to tensor
	xData := make([]float32, len(x))
	for i, v := range x {
		xData[i] = float32(v)
	}
	
	xTensor, err := tensor.NewTensor([]int{len(x)}, xData)
	if err != nil {
		panic(err)
	}
	defer xTensor.ReleaseGPU()

	result, err := SparseMatVec(gs.sparse, xTensor)
	if err != nil {
		panic(err)
	}
	defer result.ReleaseGPU()

	if err := result.RetrieveCPU(); err != nil {
		panic(err)
	}

	// Convert back to float64 slice
	output := make([]float64, len(result.Data))
	for i, v := range result.Data {
		output[i] = float64(v)
	}

	return output
}

// ToDense converts the sparse matrix to a dense gonum matrix
func (gs *GPUSparse) ToDense() *mat.Dense {
	denseTensor, err := SparseToDense(gs.sparse)
	if err != nil {
		panic(err)
	}
	defer denseTensor.ReleaseGPU()

	if err := denseTensor.RetrieveCPU(); err != nil {
		panic(err)
	}

	rows, cols := denseTensor.Shape[0], denseTensor.Shape[1]
	data := make([]float64, len(denseTensor.Data))
	for i, v := range denseTensor.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

// ReleaseGPU releases GPU resources
func (gs *GPUSparse) ReleaseGPU() {
	gs.sparse.ReleaseGPU()
}

// GetSparseTensor returns the underlying sparse tensor (for advanced users)
func (gs *GPUSparse) GetSparseTensor() *tensor.SparseTensor {
	return gs.sparse
}

// Drop-in replacement functions for gonum operations with sparse matrices

// GPUSparseMatMul is a drop-in replacement for matrix multiplication with sparse matrices
func GPUSparseMatMul(a, b interface{}) *mat.Dense {
	// Handle different input types
	var aInterface, bInterface interface{}
	
	switch at := a.(type) {
	case *GPUSparse:
		aInterface = at.sparse
	case *mat.Dense:
		// Convert dense to our tensor format
		rows, cols := at.Dims()
		data := make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				data[i*cols+j] = at.At(i, j)
			}
		}
		float32Data := make([]float32, len(data))
		for i, v := range data {
			float32Data[i] = float32(v)
		}
		denseTensor, err := tensor.NewTensor([]int{rows, cols}, float32Data)
		if err != nil {
			panic(err)
		}
		defer denseTensor.ReleaseGPU()
		aInterface = denseTensor
	case mat.Matrix:
		// Convert general matrix to dense tensor
		gpuDense := FromGonum(at)
		defer gpuDense.ReleaseGPU()
		aInterface = gpuDense.tensor
	default:
		panic("unsupported matrix type for first operand")
	}
	
	switch bt := b.(type) {
	case *GPUSparse:
		bInterface = bt.sparse
	case *mat.Dense:
		// Convert dense to our tensor format
		rows, cols := bt.Dims()
		data := make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				data[i*cols+j] = bt.At(i, j)
			}
		}
		float32Data := make([]float32, len(data))
		for i, v := range data {
			float32Data[i] = float32(v)
		}
		denseTensor, err := tensor.NewTensor([]int{rows, cols}, float32Data)
		if err != nil {
			panic(err)
		}
		defer denseTensor.ReleaseGPU()
		bInterface = denseTensor
	case mat.Matrix:
		// Convert general matrix to dense tensor
		gpuDense := FromGonum(bt)
		defer gpuDense.ReleaseGPU()
		bInterface = gpuDense.tensor
	default:
		panic("unsupported matrix type for second operand")
	}

	result, err := SparseMatMul(aInterface, bInterface)
	if err != nil {
		panic(err)
	}
	defer result.ReleaseGPU()

	// Convert back to gonum format
	if err := result.RetrieveCPU(); err != nil {
		panic(err)
	}

	rows, cols := result.Shape[0], result.Shape[1]
	data := make([]float64, len(result.Data))
	for i, v := range result.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

// GPUDenseToSparse converts a dense matrix to sparse format
func GPUDenseToSparse(m mat.Matrix, threshold float64) *GPUSparse {
	gpuDense := FromGonum(m)
	defer gpuDense.ReleaseGPU()

	sparseTensor, err := DenseToSparse(gpuDense.tensor, float32(threshold))
	if err != nil {
		panic(err)
	}

	return &GPUSparse{sparse: sparseTensor}
}

// GPUSparseToDense converts a sparse matrix to dense format
func GPUSparseToDense(gs *GPUSparse) *mat.Dense {
	return gs.ToDense()
}

// Batch operations for sparse matrices

// BatchGPUSparseMatMul performs multiple sparse matrix multiplications efficiently
func BatchGPUSparseMatMul(operations []struct{ A, B interface{} }) []*mat.Dense {
	results := make([]*mat.Dense, len(operations))

	for i, op := range operations {
		results[i] = GPUSparseMatMul(op.A, op.B)
	}

	return results
}

// BatchGPUSparseMatVec performs multiple sparse matrix-vector multiplications
func BatchGPUSparseMatVec(matrices []*GPUSparse, vectors [][]float64) [][]float64 {
	if len(matrices) != len(vectors) {
		panic("number of matrices must equal number of vectors")
	}

	results := make([][]float64, len(matrices))
	for i := range matrices {
		results[i] = matrices[i].SparseMatVec(vectors[i])
	}

	return results
}

// BatchGPUSparseAdd performs multiple sparse matrix additions efficiently
func BatchGPUSparseAdd(operations []struct{ A, B *GPUSparse }) []*GPUSparse {
	results := make([]*GPUSparse, len(operations))

	for i, op := range operations {
		results[i] = op.A.SparseAdd(op.B)
	}

	return results
}

// BatchGPUSparseScalarMul performs multiple sparse scalar multiplications efficiently
func BatchGPUSparseScalarMul(matrices []*GPUSparse, scalars []float64) []*GPUSparse {
	if len(matrices) != len(scalars) {
		panic("number of matrices must equal number of scalars")
	}

	results := make([]*GPUSparse, len(matrices))
	for i := range matrices {
		results[i] = matrices[i].SparseScalarMul(scalars[i])
	}

	return results
}

// Utility functions for sparse matrix analysis

// SparseMatrixInfo provides information about a sparse matrix
type SparseMatrixInfo struct {
	Rows         int
	Cols         int
	NNZ          int
	Density      float64
	Format       tensor.SparseFormat
	MemoryUsage  int64 // Estimated memory usage in bytes
}

// GetSparseMatrixInfo returns detailed information about a sparse matrix
func GetSparseMatrixInfo(gs *GPUSparse) SparseMatrixInfo {
	rows, cols := gs.Dims()
	nnz := gs.GetNNZ()
	density := gs.GetDensity()
	format := gs.GetFormat()
	
	// Estimate memory usage
	var memoryUsage int64
	switch format {
	case tensor.COO:
		memoryUsage = int64(nnz) * (4 + 4 + 4) // int32 + int32 + float32
	case tensor.CSR:
		memoryUsage = int64(rows+1)*4 + int64(nnz)*4 + int64(nnz)*4 // rowPtr + indices + data
	case tensor.CSC:
		memoryUsage = int64(cols+1)*4 + int64(nnz)*4 + int64(nnz)*4 // colPtr + indices + data
	}
	
	return SparseMatrixInfo{
		Rows:        rows,
		Cols:        cols,
		NNZ:         nnz,
		Density:     density,
		Format:      format,
		MemoryUsage: memoryUsage,
	}
}

// IsSparseWorthwhile determines if using sparse format is beneficial
func IsSparseWorthwhile(rows, cols, nnz int) bool {
	totalElements := int64(rows) * int64(cols)
	denseMemory := totalElements * 4 // float32
	
	// Estimate sparse memory (using CSR as baseline)
	sparseMemory := int64(rows+1)*4 + int64(nnz)*8 // rowPtr + (indices + data)
	
	// Consider sparse worthwhile if it uses less than 70% of dense memory
	// and density is less than 30%
	density := float64(nnz) / float64(totalElements)
	
	return sparseMemory < (denseMemory*7/10) && density < 0.3
}

// String returns a string representation of sparse matrix info
func (info SparseMatrixInfo) String() string {
	return fmt.Sprintf("SparseMatrix{%dx%d, NNZ=%d, Density=%.4f, Format=%s, Memory=%d bytes}",
		info.Rows, info.Cols, info.NNZ, info.Density, info.Format.String(), info.MemoryUsage)
}