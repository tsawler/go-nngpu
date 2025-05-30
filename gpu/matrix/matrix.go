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