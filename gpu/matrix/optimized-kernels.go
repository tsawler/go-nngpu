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
)

// Phase 8B: Custom Metal Kernels for Specific Operations
// These functions provide optimized GPU implementations using custom Metal kernels

// OptimizedGEMM performs optimized General Matrix Multiplication with tiling and shared memory
func OptimizedGEMM(A, B *tensor.Tensor, alpha, beta float32) (*tensor.Tensor, error) {
	if len(A.Shape) != 2 || len(B.Shape) != 2 {
		return nil, fmt.Errorf("optimized GEMM requires 2D tensors")
	}

	M := A.Shape[0]
	K := A.Shape[1]
	K2 := B.Shape[0]
	N := B.Shape[1]

	if K != K2 {
		return nil, fmt.Errorf("matrix dimensions don't match: A(%dx%d) x B(%dx%d)", M, K, K2, N)
	}

	// Create output tensor
	outputData := make([]float32, M*N)
	output, err := tensor.NewTensor([]int{M, N}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move B to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_optimized_gemm(
		C.GPUPtr(A.GPUPtr()), C.GPUPtr(B.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.long(M), C.long(N), C.long(K),
		C.float(alpha), C.float(beta),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized GEMM failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedBatchMatMul performs optimized batch matrix multiplication
func OptimizedBatchMatMul(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if len(A.Shape) != 3 || len(B.Shape) != 3 {
		return nil, fmt.Errorf("batch matrix multiplication requires 3D tensors")
	}

	batchSize := A.Shape[0]
	M := A.Shape[1]
	K := A.Shape[2]
	K2 := B.Shape[1]
	N := B.Shape[2]

	if A.Shape[0] != B.Shape[0] {
		return nil, fmt.Errorf("batch sizes don't match: %d vs %d", A.Shape[0], B.Shape[0])
	}
	if K != K2 {
		return nil, fmt.Errorf("matrix dimensions don't match: K=%d vs K2=%d", K, K2)
	}

	// Create output tensor
	outputData := make([]float32, batchSize*M*N)
	output, err := tensor.NewTensor([]int{batchSize, M, N}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := A.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move A to GPU: %w", err)
	}
	if err := B.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move B to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_batch_matmul_optimized(
		C.GPUPtr(A.GPUPtr()), C.GPUPtr(B.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.long(batchSize), C.long(M), C.long(N), C.long(K),
		C.DevicePtr(A.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized batch matrix multiplication failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedConv1x1 performs optimized 1x1 convolution (pointwise convolution)
func OptimizedConv1x1(input, weight *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input must be 4D tensor [batch, height, width, channels]")
	}
	if len(weight.Shape) != 2 {
		return nil, fmt.Errorf("weight must be 2D tensor [in_channels, out_channels]")
	}

	batch := input.Shape[0]
	height := input.Shape[1]
	width := input.Shape[2]
	inChannels := input.Shape[3]
	outChannels := weight.Shape[1]

	if weight.Shape[0] != inChannels {
		return nil, fmt.Errorf("weight input channels (%d) must match input channels (%d)", weight.Shape[0], inChannels)
	}

	// Create output tensor
	outputData := make([]float32, batch*height*width*outChannels)
	output, err := tensor.NewTensor([]int{batch, height, width, outChannels}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input to GPU: %w", err)
	}
	if err := weight.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move weight to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_conv1x1_optimized(
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(weight.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.long(batch), C.long(height), C.long(width),
		C.long(inChannels), C.long(outChannels),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized 1x1 convolution failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedDepthwiseConv performs optimized depthwise convolution
func OptimizedDepthwiseConv(input, kernel *tensor.Tensor, strideH, strideW, padH, padW int) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input must be 4D tensor [batch, height, width, channels]")
	}
	if len(kernel.Shape) != 3 {
		return nil, fmt.Errorf("kernel must be 3D tensor [kernel_h, kernel_w, channels]")
	}

	batch := input.Shape[0]
	inHeight := input.Shape[1]
	inWidth := input.Shape[2]
	channels := input.Shape[3]
	kernelH := kernel.Shape[0]
	kernelW := kernel.Shape[1]

	if kernel.Shape[2] != channels {
		return nil, fmt.Errorf("kernel channels (%d) must match input channels (%d)", kernel.Shape[2], channels)
	}

	// Calculate output dimensions
	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1

	// Create output tensor
	outputData := make([]float32, batch*outHeight*outWidth*channels)
	output, err := tensor.NewTensor([]int{batch, outHeight, outWidth, channels}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input to GPU: %w", err)
	}
	if err := kernel.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move kernel to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_depthwise_conv_optimized(
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(kernel.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.long(batch), C.long(inHeight), C.long(inWidth), C.long(channels),
		C.long(kernelH), C.long(kernelW),
		C.long(strideH), C.long(strideW), C.long(padH), C.long(padW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized depthwise convolution failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedElementwiseBinaryOp performs optimized elementwise binary operations with broadcasting
func OptimizedElementwiseBinaryOp(a, b *tensor.Tensor, opType int) (*tensor.Tensor, error) {
	// Determine output shape (should be broadcastable)
	var outputShape []int
	var size int

	if len(a.Data) >= len(b.Data) {
		outputShape = a.Shape
		size = len(a.Data)
	} else {
		outputShape = b.Shape
		size = len(b.Data)
	}

	// Create output tensor
	outputData := make([]float32, size)
	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := a.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor a to GPU: %w", err)
	}
	if err := b.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move tensor b to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	// Calculate strides for broadcasting
	aStride := int64(1)
	bStride := int64(1)
	if len(a.Data) < len(b.Data) {
		aStride = 0 // Will broadcast a
	} else if len(b.Data) < len(a.Data) {
		bStride = 0 // Will broadcast b
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_elementwise_binary_op_optimized(
		C.GPUPtr(a.GPUPtr()), C.GPUPtr(b.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.int(opType), C.long(size), C.long(aStride), C.long(bStride),
		C.DevicePtr(a.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized elementwise binary operation failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedReduce performs optimized reduction operations
func OptimizedReduce(input *tensor.Tensor, opType int) (*tensor.Tensor, error) {
	size := len(input.Data)

	// Create output tensor (scalar result)
	outputData := make([]float32, 1)
	output, err := tensor.NewTensor([]int{1}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_reduce_optimized(
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.long(size), C.int(opType),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized reduction failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedSoftmax performs numerically stable optimized softmax
func OptimizedSoftmax(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("optimized softmax requires 2D tensor [batch_size, num_classes]")
	}

	batchSize := input.Shape[0]
	numClasses := input.Shape[1]

	// Create output tensor
	outputData := make([]float32, batchSize*numClasses)
	output, err := tensor.NewTensor([]int{batchSize, numClasses}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_softmax_optimized(
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(output.GPUPtr()),
		C.long(batchSize), C.long(numClasses),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("optimized softmax failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// OptimizedLayerNorm performs optimized layer normalization
func OptimizedLayerNorm(input, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, nil, nil, fmt.Errorf("optimized layer norm requires 2D tensor [batch_size, feature_size]")
	}

	batchSize := input.Shape[0]
	featureSize := input.Shape[1]

	if len(gamma.Shape) != 1 || gamma.Shape[0] != featureSize {
		return nil, nil, nil, fmt.Errorf("gamma must have shape [%d], got %v", featureSize, gamma.Shape)
	}
	if len(beta.Shape) != 1 || beta.Shape[0] != featureSize {
		return nil, nil, nil, fmt.Errorf("beta must have shape [%d], got %v", featureSize, beta.Shape)
	}

	// Create output tensors
	outputData := make([]float32, batchSize*featureSize)
	meanData := make([]float32, batchSize)
	varData := make([]float32, batchSize)

	output, err := tensor.NewTensor([]int{batchSize, featureSize}, outputData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	mean, err := tensor.NewTensor([]int{batchSize}, meanData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create mean tensor: %w", err)
	}

	variance, err := tensor.NewTensor([]int{batchSize}, varData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create variance tensor: %w", err)
	}

	// Ensure tensors are on GPU
	tensors := []*tensor.Tensor{input, gamma, beta, output, mean, variance}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_layer_norm_optimized(
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(gamma.GPUPtr()), C.GPUPtr(beta.GPUPtr()),
		C.GPUPtr(output.GPUPtr()), C.GPUPtr(mean.GPUPtr()), C.GPUPtr(variance.GPUPtr()),
		C.long(batchSize), C.long(featureSize), C.float(epsilon),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, nil, nil, fmt.Errorf("optimized layer normalization failed (code %d): %s", retCode, errMsg)
	}

	return output, mean, variance, nil
}

// Optimized elementwise operation types
const (
	OptimizedOpAdd = iota
	OptimizedOpSubtract
	OptimizedOpMultiply
	OptimizedOpDivide
	OptimizedOpMaximum
	OptimizedOpMinimum
	OptimizedOpPower
)

// Optimized reduction operation types
const (
	OptimizedOpSum = iota
	OptimizedOpMax
	OptimizedOpMin
)

// Helper functions for common optimized operations

// OptimizedAdd performs optimized element-wise addition with broadcasting
func OptimizedAdd(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedElementwiseBinaryOp(a, b, OptimizedOpAdd)
}

// OptimizedSubtract performs optimized element-wise subtraction with broadcasting
func OptimizedSubtract(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedElementwiseBinaryOp(a, b, OptimizedOpSubtract)
}

// OptimizedMultiply performs optimized element-wise multiplication with broadcasting
func OptimizedMultiply(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedElementwiseBinaryOp(a, b, OptimizedOpMultiply)
}

// OptimizedDivide performs optimized element-wise division with broadcasting
func OptimizedDivide(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedElementwiseBinaryOp(a, b, OptimizedOpDivide)
}

// OptimizedSum performs optimized sum reduction
func OptimizedSum(input *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedReduce(input, OptimizedOpSum)
}

// OptimizedMax performs optimized maximum reduction
func OptimizedMax(input *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedReduce(input, OptimizedOpMax)
}

// OptimizedMin performs optimized minimum reduction
func OptimizedMin(input *tensor.Tensor) (*tensor.Tensor, error) {
	return OptimizedReduce(input, OptimizedOpMin)
}