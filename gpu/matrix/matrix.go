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

	"github.com/tsawler/go-nngpu/tensor"
	_ "github.com/tsawler/go-nngpu/internal/cgo"
)

// Global device pointer for unified memory system
var globalMatrixDevice unsafe.Pointer

// SetGlobalMatrixDevice sets the global device for matrix operations
func SetGlobalMatrixDevice(device unsafe.Pointer) {
	globalMatrixDevice = device
}

// ensureUnifiedGPU ensures tensor is on GPU using unified memory system when available
func ensureUnifiedGPU(t *tensor.Tensor) error {
	if globalMatrixDevice != nil {
		// Use unified memory adapter
		adapter := GetGlobalTensorAdapter(globalMatrixDevice)
		// fmt.Printf("[DEBUG] Using unified memory for tensor %p\n", t)
		return adapter.EnsureUnifiedGPU(t)
	}
	// Fall back to standard GPU allocation
	// fmt.Printf("[DEBUG] Using standard GPU allocation for tensor %p\n", t)
	return t.EnsureGPU()
}

// getUnifiedGPUPtr gets GPU pointer using unified memory system when available
func getUnifiedGPUPtr(t *tensor.Tensor) unsafe.Pointer {
	if globalMatrixDevice != nil {
		// Try unified memory first
		adapter := GetGlobalTensorAdapter(globalMatrixDevice)
		if ptr := adapter.GetUnifiedGPUBuffer(t); ptr != nil {
			return ptr
		}
	}
	// Fall back to standard GPU pointer
	return t.GPUPtr()
}

// MatMul performs matrix multiplication C = A * B on the GPU
func MatMul(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors (matrices)")
	}
	if A.Shape[1] != B.Shape[0] {
		return nil, fmt.Errorf("incompatible matrix dimensions for multiplication: A columns (%d) != B rows (%d)", A.Shape[1], B.Shape[0])
	}

	rowsA := A.Shape[0]
	colsB := B.Shape[1]
	resultRows := rowsA
	resultCols := colsB
	resultSize := resultRows * resultCols

	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{resultRows, resultCols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := ensureUnifiedGPU(A); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := ensureUnifiedGPU(B); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := ensureUnifiedGPU(resultTensor); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_multiplication(
		C.GPUPtr(getUnifiedGPUPtr(A)), C.long(A.Shape[0]), C.long(A.Shape[1]),
		C.GPUPtr(getUnifiedGPUPtr(B)), C.long(B.Shape[0]), C.long(B.Shape[1]),
		C.GPUPtr(getUnifiedGPUPtr(resultTensor)), C.long(resultTensor.Shape[0]), C.long(resultTensor.Shape[1]),
		C.DevicePtr(globalMatrixDevice),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU matrix multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Cholesky performs Cholesky decomposition using the Accelerate framework
// Returns the lower triangular matrix L such that A = L * L^T
func Cholesky(A *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("Cholesky decomposition requires 2D tensor (matrix)")
	}
	if A.Shape[0] != A.Shape[1] {
		return nil, fmt.Errorf("Cholesky decomposition requires square matrix, got %dx%d", A.Shape[0], A.Shape[1])
	}

	rows := A.Shape[0]
	cols := A.Shape[1]
	resultSize := rows * cols
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{rows, cols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_matrix_cholesky_decomposition(
		C.GPUPtr(A.GPUPtr()), C.long(rows), C.long(cols),
		C.GPUPtr(resultTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("Cholesky decomposition failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// EigenDecomposition represents the result of eigenvalue decomposition
type EigenDecomposition struct {
	Eigenvalues  *tensor.Tensor // Vector of eigenvalues
	Eigenvectors *tensor.Tensor // Matrix of eigenvectors (column-wise)
}

// ReleaseGPU releases GPU resources for the eigenvalue decomposition
func (eigen *EigenDecomposition) ReleaseGPU() {
	if eigen.Eigenvalues != nil {
		eigen.Eigenvalues.ReleaseGPU()
	}
	if eigen.Eigenvectors != nil {
		eigen.Eigenvectors.ReleaseGPU()
	}
}

// Eigen performs eigenvalue decomposition for symmetric matrices using the Accelerate framework
func Eigen(A *tensor.Tensor) (*EigenDecomposition, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("Eigenvalue decomposition requires 2D tensor (matrix)")
	}
	if A.Shape[0] != A.Shape[1] {
		return nil, fmt.Errorf("Eigenvalue decomposition requires square matrix, got %dx%d", A.Shape[0], A.Shape[1])
	}

	n := A.Shape[0]

	// Create eigenvalues vector (n) and eigenvectors matrix (n x n)
	eigenvaluesData := make([]float32, n)
	eigenvectorsData := make([]float32, n*n)

	eigenvaluesTensor, err := tensor.NewTensor([]int{n}, eigenvaluesData)
	if err != nil {
		return nil, fmt.Errorf("failed to create eigenvalues tensor: %w", err)
	}

	eigenvectorsTensor, err := tensor.NewTensor([]int{n, n}, eigenvectorsData)
	if err != nil {
		eigenvaluesTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to create eigenvectors tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		eigenvaluesTensor.ReleaseGPU()
		eigenvectorsTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := eigenvaluesTensor.EnsureGPU(); err != nil {
		eigenvaluesTensor.ReleaseGPU()
		eigenvectorsTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move eigenvalues tensor to GPU: %w", err)
	}
	if err := eigenvectorsTensor.EnsureGPU(); err != nil {
		eigenvaluesTensor.ReleaseGPU()
		eigenvectorsTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move eigenvectors tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_matrix_eigenvalue_decomposition(
		C.GPUPtr(A.GPUPtr()), C.long(n), C.long(n),
		C.GPUPtr(eigenvaluesTensor.GPUPtr()),
		C.GPUPtr(eigenvectorsTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		eigenvaluesTensor.ReleaseGPU()
		eigenvectorsTensor.ReleaseGPU()
		return nil, fmt.Errorf("eigenvalue decomposition failed (code %d): %s", retCode, errMsg)
	}

	return &EigenDecomposition{
		Eigenvalues:  eigenvaluesTensor,
		Eigenvectors: eigenvectorsTensor,
	}, nil
}

// SVDDecomposition represents the result of Singular Value Decomposition
type SVDDecomposition struct {
	U  *tensor.Tensor // Left singular vectors (m x m)
	S  *tensor.Tensor // Singular values (min(m,n))
	VT *tensor.Tensor // Right singular vectors transposed (n x n)
}

// ReleaseGPU releases GPU resources for the SVD decomposition
func (svd *SVDDecomposition) ReleaseGPU() {
	if svd.U != nil {
		svd.U.ReleaseGPU()
	}
	if svd.S != nil {
		svd.S.ReleaseGPU()
	}
	if svd.VT != nil {
		svd.VT.ReleaseGPU()
	}
}

// SVD performs Singular Value Decomposition using the Accelerate framework
func SVD(A *tensor.Tensor) (*SVDDecomposition, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("SVD requires 2D tensor (matrix)")
	}

	m := A.Shape[0] // rows
	n := A.Shape[1] // cols
	minDim := m
	if n < m {
		minDim = n
	}

	// Create U matrix (m x m), S vector (min(m,n)), and VT matrix (n x n)
	uData := make([]float32, m*m)
	sData := make([]float32, minDim)
	vtData := make([]float32, n*n)

	uTensor, err := tensor.NewTensor([]int{m, m}, uData)
	if err != nil {
		return nil, fmt.Errorf("failed to create U tensor: %w", err)
	}

	sTensor, err := tensor.NewTensor([]int{minDim}, sData)
	if err != nil {
		uTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to create S tensor: %w", err)
	}

	vtTensor, err := tensor.NewTensor([]int{n, n}, vtData)
	if err != nil {
		uTensor.ReleaseGPU()
		sTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to create VT tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		uTensor.ReleaseGPU()
		sTensor.ReleaseGPU()
		vtTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := uTensor.EnsureGPU(); err != nil {
		uTensor.ReleaseGPU()
		sTensor.ReleaseGPU()
		vtTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move U tensor to GPU: %w", err)
	}
	if err := sTensor.EnsureGPU(); err != nil {
		uTensor.ReleaseGPU()
		sTensor.ReleaseGPU()
		vtTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move S tensor to GPU: %w", err)
	}
	if err := vtTensor.EnsureGPU(); err != nil {
		uTensor.ReleaseGPU()
		sTensor.ReleaseGPU()
		vtTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move VT tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_matrix_svd_decomposition(
		C.GPUPtr(A.GPUPtr()), C.long(m), C.long(n),
		C.GPUPtr(uTensor.GPUPtr()),
		C.GPUPtr(sTensor.GPUPtr()),
		C.GPUPtr(vtTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		uTensor.ReleaseGPU()
		sTensor.ReleaseGPU()
		vtTensor.ReleaseGPU()
		return nil, fmt.Errorf("SVD failed (code %d): %s", retCode, errMsg)
	}

	return &SVDDecomposition{
		U:  uTensor,
		S:  sTensor,
		VT: vtTensor,
	}, nil
}

// Transpose performs matrix transpose on the GPU
func Transpose(A *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("Transpose requires 2D tensor (matrix)")
	}

	// Transposed dimensions
	outputRows := A.Shape[1]
	outputCols := A.Shape[0]
	resultSize := outputRows * outputCols

	// Create result tensor
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{outputRows, outputCols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor for transpose: %w", err)
	}

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	// Perform GPU computation
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_transpose(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
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
		return nil, fmt.Errorf("GPU matrix transpose failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Add performs element-wise matrix addition C = A + B on the GPU
func Add(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("Add requires 2D tensors (matrices)")
	}
	
	// Check for broadcasting compatibility
	canBroadcast, resultShape := canBroadcast2D(A.Shape, B.Shape)
	if !canBroadcast {
		return nil, fmt.Errorf("incompatible matrix dimensions for addition: A (%dx%d) vs B (%dx%d)", 
			A.Shape[0], A.Shape[1], B.Shape[0], B.Shape[1])
	}
	
	// If shapes are identical, use the original fast path
	if A.Shape[0] == B.Shape[0] && A.Shape[1] == B.Shape[1] {
		return addSameShape(A, B)
	}
	
	// Handle broadcasting case
	return addWithBroadcasting(A, B, resultShape)
}

// canBroadcast2D checks if two 2D tensors can be broadcast together
func canBroadcast2D(shapeA, shapeB []int) (bool, []int) {
	// For 2D broadcasting: [M, N] + [1, N] = [M, N] or [M, N] + [M, 1] = [M, N]
	// Also handle exact matches: [M, N] + [M, N] = [M, N]
	
	if len(shapeA) != 2 || len(shapeB) != 2 {
		return false, nil
	}
	
	// Case 1: Exact match
	if shapeA[0] == shapeB[0] && shapeA[1] == shapeB[1] {
		return true, []int{shapeA[0], shapeA[1]}
	}
	
	// Case 2: Broadcasting along dimension 0 (bias case: [M, N] + [1, N])
	if (shapeA[0] != shapeB[0] && (shapeA[0] == 1 || shapeB[0] == 1)) && shapeA[1] == shapeB[1] {
		resultRows := shapeA[0]
		if shapeB[0] > resultRows {
			resultRows = shapeB[0]
		}
		return true, []int{resultRows, shapeA[1]}
	}
	
	// Case 3: Broadcasting along dimension 1 ([M, N] + [M, 1])
	if shapeA[0] == shapeB[0] && (shapeA[1] != shapeB[1] && (shapeA[1] == 1 || shapeB[1] == 1)) {
		resultCols := shapeA[1]
		if shapeB[1] > resultCols {
			resultCols = shapeB[1]
		}
		return true, []int{shapeA[0], resultCols}
	}
	
	return false, nil
}

// addSameShape performs addition when both tensors have the same shape (fast path)
func addSameShape(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	resultSize := A.Shape[0] * A.Shape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{A.Shape[0], A.Shape[1]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_add(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
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
		return nil, fmt.Errorf("GPU matrix addition failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// addWithBroadcasting performs addition with broadcasting support
func addWithBroadcasting(A, B *tensor.Tensor, resultShape []int) (*tensor.Tensor, error) {
	// Create result tensor
	resultSize := resultShape[0] * resultShape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor(resultShape, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_add_broadcast(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
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
		return nil, fmt.Errorf("GPU matrix addition with broadcasting failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Sub performs element-wise matrix subtraction C = A - B on the GPU
func Sub(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("Sub requires 2D tensors (matrices)")
	}
	if A.Shape[0] != B.Shape[0] || A.Shape[1] != B.Shape[1] {
		return nil, fmt.Errorf("incompatible matrix dimensions for subtraction: A (%dx%d) != B (%dx%d)", 
			A.Shape[0], A.Shape[1], B.Shape[0], B.Shape[1])
	}

	resultSize := A.Shape[0] * A.Shape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{A.Shape[0], A.Shape[1]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_subtract(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
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
		return nil, fmt.Errorf("GPU matrix subtraction failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Mul performs element-wise matrix multiplication (Hadamard product) C = A ⊙ B on the GPU
func Mul(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("Mul requires 2D tensors (matrices)")
	}
	if A.Shape[0] != B.Shape[0] || A.Shape[1] != B.Shape[1] {
		return nil, fmt.Errorf("incompatible matrix dimensions for element-wise multiplication: A (%dx%d) != B (%dx%d)", 
			A.Shape[0], A.Shape[1], B.Shape[0], B.Shape[1])
	}

	resultSize := A.Shape[0] * A.Shape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{A.Shape[0], A.Shape[1]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_element_multiply(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
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
		return nil, fmt.Errorf("GPU matrix element-wise multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Div performs element-wise matrix division C = A ⊘ B on the GPU
func Div(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("Div requires 2D tensors (matrices)")
	}
	if A.Shape[0] != B.Shape[0] || A.Shape[1] != B.Shape[1] {
		return nil, fmt.Errorf("incompatible matrix dimensions for element-wise division: A (%dx%d) != B (%dx%d)", 
			A.Shape[0], A.Shape[1], B.Shape[0], B.Shape[1])
	}

	resultSize := A.Shape[0] * A.Shape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{A.Shape[0], A.Shape[1]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_element_divide(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
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
		return nil, fmt.Errorf("GPU matrix element-wise division failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// ScalarAdd performs scalar addition C = A + scalar on the GPU
func ScalarAdd(A *tensor.Tensor, scalar float32) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("ScalarAdd requires 2D tensor (matrix)")
	}

	resultSize := A.Shape[0] * A.Shape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{A.Shape[0], A.Shape[1]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_scalar_add(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
		C.float(scalar),
		C.GPUPtr(resultTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU scalar addition failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// ScalarMul performs scalar multiplication C = A * scalar on the GPU
func ScalarMul(A *tensor.Tensor, scalar float32) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("ScalarMul requires 2D tensor (matrix)")
	}

	resultSize := A.Shape[0] * A.Shape[1]
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{A.Shape[0], A.Shape[1]}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_scalar_multiply(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
		C.float(scalar),
		C.GPUPtr(resultTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU scalar multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Phase 3: Advanced Matrix Operations using Accelerate Framework

// Inverse computes the matrix inverse using the Accelerate framework
func Inverse(A *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("Inverse requires 2D tensor (matrix)")
	}
	if A.Shape[0] != A.Shape[1] {
		return nil, fmt.Errorf("Inverse requires square matrix, got %dx%d", A.Shape[0], A.Shape[1])
	}

	rows := A.Shape[0]
	cols := A.Shape[1]
	resultSize := rows * cols
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{rows, cols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_matrix_inverse(
		C.GPUPtr(A.GPUPtr()), C.long(rows), C.long(cols),
		C.GPUPtr(resultTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("matrix inverse failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Determinant computes the matrix determinant using the Accelerate framework
func Determinant(A *tensor.Tensor) (float32, error) {
	if len(A.Shape) != 2 {
		return 0, fmt.Errorf("Determinant requires 2D tensor (matrix)")
	}
	if A.Shape[0] != A.Shape[1] {
		return 0, fmt.Errorf("Determinant requires square matrix, got %dx%d", A.Shape[0], A.Shape[1])
	}

	if err := A.EnsureGPU(); err != nil {
		return 0, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var determinant C.float
	var cErr C.CError
	retCode := C.perform_matrix_determinant(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]),
		&determinant,
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return 0, fmt.Errorf("matrix determinant failed (code %d): %s", retCode, errMsg)
	}

	return float32(determinant), nil
}

// LUDecomposition represents the result of LU decomposition
type LUDecomposition struct {
	L            *tensor.Tensor // Lower triangular matrix
	U            *tensor.Tensor // Upper triangular matrix
	PivotIndices []int          // Pivot indices for row swaps
}

// ReleaseGPU releases GPU resources for the LU decomposition
func (lu *LUDecomposition) ReleaseGPU() {
	if lu.L != nil {
		lu.L.ReleaseGPU()
	}
	if lu.U != nil {
		lu.U.ReleaseGPU()
	}
}

// LU performs LU decomposition using the Accelerate framework
func LU(A *tensor.Tensor) (*LUDecomposition, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("LU decomposition requires 2D tensor (matrix)")
	}

	rows := A.Shape[0]
	cols := A.Shape[1]
	
	// Create L matrix (rows x rows) and U matrix (rows x cols)
	lData := make([]float32, rows*rows)
	uData := make([]float32, rows*cols)
	
	lTensor, err := tensor.NewTensor([]int{rows, rows}, lData)
	if err != nil {
		return nil, fmt.Errorf("failed to create L tensor: %w", err)
	}
	
	uTensor, err := tensor.NewTensor([]int{rows, cols}, uData)
	if err != nil {
		lTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to create U tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		lTensor.ReleaseGPU()
		uTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := lTensor.EnsureGPU(); err != nil {
		lTensor.ReleaseGPU()
		uTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move L tensor to GPU: %w", err)
	}
	if err := uTensor.EnsureGPU(); err != nil {
		lTensor.ReleaseGPU()
		uTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move U tensor to GPU: %w", err)
	}

	// Allocate pivot indices array
	minDim := rows
	if cols < rows {
		minDim = cols
	}
	pivotIndices := make([]int, minDim)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Convert Go slice to C array
	cPivotIndices := (*C.int)(unsafe.Pointer(&pivotIndices[0]))

	var cErr C.CError
	retCode := C.perform_matrix_lu_decomposition(
		C.GPUPtr(A.GPUPtr()), C.long(rows), C.long(cols),
		C.GPUPtr(lTensor.GPUPtr()), C.GPUPtr(uTensor.GPUPtr()),
		cPivotIndices,
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		lTensor.ReleaseGPU()
		uTensor.ReleaseGPU()
		return nil, fmt.Errorf("LU decomposition failed (code %d): %s", retCode, errMsg)
	}

	return &LUDecomposition{
		L:            lTensor,
		U:            uTensor,
		PivotIndices: pivotIndices,
	}, nil
}

// Phase 4: Advanced Decompositions

// QRDecomposition represents the result of QR decomposition
type QRDecomposition struct {
	Q *tensor.Tensor // Orthogonal matrix
	R *tensor.Tensor // Upper triangular matrix
}

// ReleaseGPU releases GPU resources for the QR decomposition
func (qr *QRDecomposition) ReleaseGPU() {
	if qr.Q != nil {
		qr.Q.ReleaseGPU()
	}
	if qr.R != nil {
		qr.R.ReleaseGPU()
	}
}

// QR performs QR decomposition using the Accelerate framework
func QR(A *tensor.Tensor) (*QRDecomposition, error) {
	if len(A.Shape) != 2 {
		return nil, fmt.Errorf("QR decomposition requires 2D tensor (matrix)")
	}

	rows := A.Shape[0]
	cols := A.Shape[1]

	// Create Q matrix (rows x cols) and R matrix (cols x cols)
	qData := make([]float32, rows*cols)
	rData := make([]float32, cols*cols)

	qTensor, err := tensor.NewTensor([]int{rows, cols}, qData)
	if err != nil {
		return nil, fmt.Errorf("failed to create Q tensor: %w", err)
	}

	rTensor, err := tensor.NewTensor([]int{cols, cols}, rData)
	if err != nil {
		qTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to create R tensor: %w", err)
	}

	if err := A.EnsureGPU(); err != nil {
		qTensor.ReleaseGPU()
		rTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := qTensor.EnsureGPU(); err != nil {
		qTensor.ReleaseGPU()
		rTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move Q tensor to GPU: %w", err)
	}
	if err := rTensor.EnsureGPU(); err != nil {
		qTensor.ReleaseGPU()
		rTensor.ReleaseGPU()
		return nil, fmt.Errorf("failed to move R tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_matrix_qr_decomposition(
		C.GPUPtr(A.GPUPtr()), C.long(rows), C.long(cols),
		C.GPUPtr(qTensor.GPUPtr()), C.GPUPtr(rTensor.GPUPtr()),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		qTensor.ReleaseGPU()
		rTensor.ReleaseGPU()
		return nil, fmt.Errorf("QR decomposition failed (code %d): %s", retCode, errMsg)
	}

	return &QRDecomposition{
		Q: qTensor,
		R: rTensor,
	}, nil
}
