package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	_ "github.com/tsawler/go-nngpu/internal/cgo"
	"github.com/tsawler/go-nngpu/tensor"
)

// SparseMatMul performs sparse matrix multiplication C = A * B
// Supports sparse-dense, dense-sparse, and sparse-sparse multiplication
func SparseMatMul(A, B interface{}) (*tensor.Tensor, error) {
	// Determine the types of A and B
	sparseA, isASparse := A.(*tensor.SparseTensor)
	sparseB, isBSparse := B.(*tensor.SparseTensor)
	denseA, isADense := A.(*tensor.Tensor)
	denseB, isBDense := B.(*tensor.Tensor)

	if !isASparse && !isADense {
		return nil, fmt.Errorf("first operand must be either SparseTensor or Tensor")
	}
	if !isBSparse && !isBDense {
		return nil, fmt.Errorf("second operand must be either SparseTensor or Tensor")
	}

	var aShape, bShape []int
	if isASparse {
		aShape = sparseA.Shape
	} else {
		aShape = denseA.Shape
	}
	if isBSparse {
		bShape = sparseB.Shape
	} else {
		bShape = denseB.Shape
	}

	// Validate dimensions
	if len(aShape) != 2 || len(bShape) != 2 {
		return nil, fmt.Errorf("sparse matrix multiplication requires 2D tensors")
	}
	if aShape[1] != bShape[0] {
		return nil, fmt.Errorf("incompatible matrix dimensions for multiplication: A columns (%d) != B rows (%d)", aShape[1], bShape[0])
	}

	// Route to appropriate implementation based on operand types
	switch {
	case isASparse && isBSparse:
		return sparseSparseMatMul(sparseA, sparseB)
	case isASparse && isBDense:
		return sparseDenseMatMul(sparseA, denseB)
	case isADense && isBSparse:
		return denseSparseMatMul(denseA, sparseB)
	default:
		// Both dense - use regular dense multiplication
		return MatMul(denseA, denseB)
	}
}

// sparseSparseMatMul performs sparse-sparse matrix multiplication
func sparseSparseMatMul(A, B *tensor.SparseTensor) (*tensor.Tensor, error) {
	// For sparse-sparse multiplication, we'll convert to CSR format for efficiency
	if err := A.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert A to CSR: %w", err)
	}
	if err := B.ConvertToCSC(); err != nil {
		return nil, fmt.Errorf("failed to convert B to CSC: %w", err)
	}

	resultRows := A.Shape[0]
	resultCols := B.Shape[1]

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor B to GPU: %w", err)
	}

	// Create result tensor
	resultSize := resultRows * resultCols
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{resultRows, resultCols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_sparse_sparse_matmul(
		C.GPUPtr(A.GPUPtr()),
		(*C.int)(unsafe.Pointer(&A.RowPtr[0])), C.long(len(A.RowPtr)),
		(*C.int)(unsafe.Pointer(&A.Indices[0])), C.long(len(A.Indices)),
		C.long(A.Shape[0]), C.long(A.Shape[1]), C.long(A.NNZ),

		C.GPUPtr(B.GPUPtr()),
		(*C.int)(unsafe.Pointer(&B.ColPtr[0])), C.long(len(B.ColPtr)),
		(*C.int)(unsafe.Pointer(&B.Indices[0])), C.long(len(B.Indices)),
		C.long(B.Shape[0]), C.long(B.Shape[1]), C.long(B.NNZ),

		C.GPUPtr(resultTensor.GPUPtr()), C.long(resultTensor.Shape[0]), C.long(resultTensor.Shape[1]),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse-sparse matrix multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// sparseDenseMatMul performs sparse-dense matrix multiplication
func sparseDenseMatMul(A *tensor.SparseTensor, B *tensor.Tensor) (*tensor.Tensor, error) {
	// Convert sparse matrix to CSR for efficient row-wise access
	if err := A.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert A to CSR: %w", err)
	}

	resultRows := A.Shape[0]
	resultCols := B.Shape[1]

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}

	// Create result tensor
	resultSize := resultRows * resultCols
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{resultRows, resultCols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_sparse_dense_matmul(
		C.GPUPtr(A.GPUPtr()),
		(*C.int)(unsafe.Pointer(&A.RowPtr[0])), C.long(len(A.RowPtr)),
		(*C.int)(unsafe.Pointer(&A.Indices[0])), C.long(len(A.Indices)),
		C.long(A.Shape[0]), C.long(A.Shape[1]), C.long(A.NNZ),

		C.GPUPtr(B.GPUPtr()), C.long(B.Shape[0]), C.long(B.Shape[1]),

		C.GPUPtr(resultTensor.GPUPtr()), C.long(resultTensor.Shape[0]), C.long(resultTensor.Shape[1]),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse-dense matrix multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// denseSparseMatMul performs dense-sparse matrix multiplication
func denseSparseMatMul(A *tensor.Tensor, B *tensor.SparseTensor) (*tensor.Tensor, error) {
	// Convert sparse matrix to CSC for efficient column-wise access
	if err := B.ConvertToCSC(); err != nil {
		return nil, fmt.Errorf("failed to convert B to CSC: %w", err)
	}

	resultRows := A.Shape[0]
	resultCols := B.Shape[1]

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor B to GPU: %w", err)
	}

	// Create result tensor
	resultSize := resultRows * resultCols
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{resultRows, resultCols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_dense_sparse_matmul(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),

		C.GPUPtr(B.GPUPtr()),
		(*C.int)(unsafe.Pointer(&B.ColPtr[0])), C.long(len(B.ColPtr)),
		(*C.int)(unsafe.Pointer(&B.Indices[0])), C.long(len(B.Indices)),
		C.long(B.Shape[0]), C.long(B.Shape[1]), C.long(B.NNZ),

		C.GPUPtr(resultTensor.GPUPtr()), C.long(resultTensor.Shape[0]), C.long(resultTensor.Shape[1]),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU dense-sparse matrix multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// SparseAdd performs element-wise addition of sparse matrices
func SparseAdd(A, B *tensor.SparseTensor) (*tensor.SparseTensor, error) {
	if err := A.IsCompatibleWith(B); err != nil {
		return nil, fmt.Errorf("incompatible tensors for addition: %w", err)
	}

	// Convert both to COO format for easier addition
	if err := A.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert A to CSR: %w", err)
	}
	if err := B.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert B to CSR: %w", err)
	}

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor B to GPU: %w", err)
	}

	// Estimate result size (upper bound)
	maxResultNNZ := A.NNZ + B.NNZ
	resultRowIndices := make([]int32, maxResultNNZ)
	resultColIndices := make([]int32, maxResultNNZ)
	resultValues := make([]float32, maxResultNNZ)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var actualNNZ C.long
	var cErr C.CError
	retCode := C.perform_sparse_add(
		C.GPUPtr(A.GPUPtr()),
		(*C.int)(unsafe.Pointer(&A.RowPtr[0])), C.long(len(A.RowPtr)),
		(*C.int)(unsafe.Pointer(&A.Indices[0])), C.long(len(A.Indices)),
		C.long(A.Shape[0]), C.long(A.Shape[1]), C.long(A.NNZ),

		C.GPUPtr(B.GPUPtr()),
		(*C.int)(unsafe.Pointer(&B.RowPtr[0])), C.long(len(B.RowPtr)),
		(*C.int)(unsafe.Pointer(&B.Indices[0])), C.long(len(B.Indices)),
		C.long(B.Shape[0]), C.long(B.Shape[1]), C.long(B.NNZ),

		(*C.int)(unsafe.Pointer(&resultRowIndices[0])),
		(*C.int)(unsafe.Pointer(&resultColIndices[0])),
		(*C.float)(unsafe.Pointer(&resultValues[0])),
		&actualNNZ, C.long(maxResultNNZ),

		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse addition failed (code %d): %s", retCode, errMsg)
	}

	// Trim result arrays to actual size
	finalRowIndices := resultRowIndices[:actualNNZ]
	finalColIndices := resultColIndices[:actualNNZ]
	finalValues := resultValues[:actualNNZ]

	return tensor.NewSparseTensorCOO(A.Shape, finalRowIndices, finalColIndices, finalValues)
}

// SparseScalarMul performs scalar multiplication on a sparse matrix
func SparseScalarMul(A *tensor.SparseTensor, scalar float32) (*tensor.SparseTensor, error) {
	// Convert to CSR format for GPU operations
	if err := A.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert A to CSR: %w", err)
	}

	// Clone the sparse tensor
	result, err := A.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone sparse tensor: %w", err)
	}

	// Ensure tensor is on GPU
	if err := result.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_sparse_scalar_multiply(
		C.GPUPtr(result.GPUPtr()),
		(*C.int)(unsafe.Pointer(&result.RowPtr[0])), C.long(len(result.RowPtr)),
		(*C.int)(unsafe.Pointer(&result.Indices[0])), C.long(len(result.Indices)),
		C.long(result.Shape[0]), C.long(result.Shape[1]), C.long(result.NNZ),
		C.float(scalar),
		C.DevicePtr(result.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse scalar multiplication failed (code %d): %s", retCode, errMsg)
	}

	return result, nil
}

// SparseTranspose computes the transpose of a sparse matrix
func SparseTranspose(A *tensor.SparseTensor) (*tensor.SparseTensor, error) {
	switch A.Format {
	case tensor.COO:
		// For COO, just swap row and column indices
		rowIndices := make([]int32, len(A.ColIndices))
		colIndices := make([]int32, len(A.RowIndices))
		values := make([]float32, len(A.Values))

		copy(rowIndices, A.ColIndices)
		copy(colIndices, A.RowIndices)
		copy(values, A.Values)

		// Swap dimensions
		newShape := []int{A.Shape[1], A.Shape[0]}

		return tensor.NewSparseTensorCOO(newShape, rowIndices, colIndices, values)

	case tensor.CSR:
		// CSR transpose becomes CSC with swapped dimensions
		result := &tensor.SparseTensor{
			Shape:   []int{A.Shape[1], A.Shape[0]},
			Format:  tensor.CSC,
			NNZ:     A.NNZ,
			ColPtr:  make([]int32, len(A.RowPtr)),
			Indices: make([]int32, len(A.Indices)),
			Data:    make([]float32, len(A.Data)),
			IsOwner: true,
		}

		copy(result.ColPtr, A.RowPtr)
		copy(result.Indices, A.Indices)
		copy(result.Data, A.Data)

		return result, nil

	case tensor.CSC:
		// CSC transpose becomes CSR with swapped dimensions
		result := &tensor.SparseTensor{
			Shape:   []int{A.Shape[1], A.Shape[0]},
			Format:  tensor.CSR,
			NNZ:     A.NNZ,
			RowPtr:  make([]int32, len(A.ColPtr)),
			Indices: make([]int32, len(A.Indices)),
			Data:    make([]float32, len(A.Data)),
			IsOwner: true,
		}

		copy(result.RowPtr, A.ColPtr)
		copy(result.Indices, A.Indices)
		copy(result.Data, A.Data)

		return result, nil

	default:
		return nil, fmt.Errorf("unsupported sparse format for transpose: %d", A.Format)
	}
}

// SparseToDense converts a sparse matrix to dense format using GPU acceleration
func SparseToDense(A *tensor.SparseTensor) (*tensor.Tensor, error) {
	// Convert to CSR format for efficient GPU conversion
	if err := A.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert to CSR: %w", err)
	}

	rows, cols := A.Shape[0], A.Shape[1]
	resultSize := rows * cols
	resultData := make([]float32, resultSize)

	result, err := tensor.NewTensor(A.Shape, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create dense result tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse tensor to GPU: %w", err)
	}
	if err := result.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_sparse_to_dense(
		C.GPUPtr(A.GPUPtr()),
		(*C.int)(unsafe.Pointer(&A.RowPtr[0])), C.long(len(A.RowPtr)),
		(*C.int)(unsafe.Pointer(&A.Indices[0])), C.long(len(A.Indices)),
		C.long(A.Shape[0]), C.long(A.Shape[1]), C.long(A.NNZ),
		C.GPUPtr(result.GPUPtr()), C.long(result.Shape[0]), C.long(result.Shape[1]),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse to dense conversion failed (code %d): %s", retCode, errMsg)
	}

	return result, nil
}

// DenseToSparse converts a dense matrix to sparse format using GPU acceleration
func DenseToSparse(A *tensor.Tensor, threshold float32) (*tensor.SparseTensor, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("dense to sparse conversion requires 2D tensor")
	}

	rows, cols := A.Shape[0], A.Shape[1]
	maxNNZ := rows * cols

	// Allocate maximum possible space for result
	rowIndices := make([]int32, maxNNZ)
	colIndices := make([]int32, maxNNZ)
	values := make([]float32, maxNNZ)

	// Ensure tensor is on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var actualNNZ C.long
	var cErr C.CError
	retCode := C.perform_dense_to_sparse(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
		C.float(threshold),
		(*C.int)(unsafe.Pointer(&rowIndices[0])),
		(*C.int)(unsafe.Pointer(&colIndices[0])),
		(*C.float)(unsafe.Pointer(&values[0])),
		&actualNNZ, C.long(maxNNZ),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU dense to sparse conversion failed (code %d): %s", retCode, errMsg)
	}

	// Trim arrays to actual size
	finalRowIndices := rowIndices[:actualNNZ]
	finalColIndices := colIndices[:actualNNZ]
	finalValues := values[:actualNNZ]

	return tensor.NewSparseTensorCOO(A.Shape, finalRowIndices, finalColIndices, finalValues)
}

// SparseMatVec performs sparse matrix-vector multiplication
func SparseMatVec(A *tensor.SparseTensor, x *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(x.Shape) != 1 {
		return nil, fmt.Errorf("sparse matrix-vector multiplication requires 2D matrix and 1D vector")
	}
	if A.Shape[1] != x.Shape[0] {
		return nil, fmt.Errorf("incompatible dimensions: matrix cols (%d) != vector size (%d)", A.Shape[1], x.Shape[0])
	}

	// Convert to CSR format for efficient row-wise access
	if err := A.ConvertToCSR(); err != nil {
		return nil, fmt.Errorf("failed to convert to CSR: %w", err)
	}

	// Create result vector
	resultData := make([]float32, A.Shape[0])
	result, err := tensor.NewTensor([]int{A.Shape[0]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result vector: %w", err)
	}

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move sparse matrix to GPU: %w", err)
	}
	if err := x.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move vector to GPU: %w", err)
	}
	if err := result.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result vector to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_sparse_matvec(
		C.GPUPtr(A.GPUPtr()),
		(*C.int)(unsafe.Pointer(&A.RowPtr[0])), C.long(len(A.RowPtr)),
		(*C.int)(unsafe.Pointer(&A.Indices[0])), C.long(len(A.Indices)),
		C.long(A.Shape[0]), C.long(A.Shape[1]), C.long(A.NNZ),
		C.GPUPtr(x.GPUPtr()), C.long(x.Shape[0]),
		C.GPUPtr(result.GPUPtr()), C.long(result.Shape[0]),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse matrix-vector multiplication failed (code %d): %s", retCode, errMsg)
	}

	return result, nil
}
