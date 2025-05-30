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
	retCode := C.perform_mps_matrix_multiplication(
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
		return nil, fmt.Errorf("GPU matrix multiplication failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
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
	if A.Shape[0] != B.Shape[0] || A.Shape[1] != B.Shape[1] {
		return nil, fmt.Errorf("incompatible matrix dimensions for addition: A (%dx%d) != B (%dx%d)", 
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