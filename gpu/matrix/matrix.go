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

	"github.com/tsawler/go-nngpu/tensor" // Import the tensor package
	_ "github.com/tsawler/go-nngpu/internal/cgo" // Import the cgo package to ensure it gets compiled
)

// MatMul performs matrix multiplication C = A * B on the GPU.
// A, B, and C must be 2D Tensors (matrices).
// Dimensions must be compatible: A (m x k), B (k x n), C (m x n).
func MatMul(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	// 1. Basic Dimension Checks
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors (matrices)")
	}
	if A.Shape[1] != B.Shape[0] {
		return nil, fmt.Errorf("incompatible matrix dimensions for multiplication: A columns (%d) != B rows (%d)", A.Shape[1], B.Shape[0])
	}

	rowsA := A.Shape[0]
	// colsA := A.Shape[1] // k
	// rowsB := B.Shape[0] // k
	colsB := B.Shape[1]

	resultRows := rowsA
	resultCols := colsB
	resultSize := resultRows * resultCols

	// 2. Create Result Tensor (CPU-backed initially)
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor([]int{resultRows, resultCols}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	// 3. Ensure all tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor B to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	// 4. Perform GPU computation via CGO
	// Lock OS thread during the CGO call
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_mps_matrix_multiplication(
		C.GPUPtr(A.GPUPtr()), C.long(A.Shape[0]), C.long(A.Shape[1]), // A matrix
		C.GPUPtr(B.GPUPtr()), C.long(B.Shape[0]), C.long(B.Shape[1]), // B matrix
		C.GPUPtr(resultTensor.GPUPtr()), C.long(resultTensor.Shape[0]), C.long(resultTensor.Shape[1]), // Result matrix
		C.DevicePtr(A.DevicePtr()), // Pass the MTLDevice pointer (from A, assuming all use same device)
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
