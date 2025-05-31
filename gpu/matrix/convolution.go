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

	_ "github.com/tsawler/go-nngpu/internal/cgo"
	"github.com/tsawler/go-nngpu/tensor"
)

// Conv2DParams represents parameters for 2D convolution
type Conv2DParams struct {
	StrideH int // Stride in height dimension
	StrideW int // Stride in width dimension
	PadH    int // Padding in height dimension
	PadW    int // Padding in width dimension
}

// Pool2DParams represents parameters for 2D pooling operations
type Pool2DParams struct {
	PoolH   int // Pool kernel height
	PoolW   int // Pool kernel width
	StrideH int // Stride in height dimension
	StrideW int // Stride in width dimension
	PadH    int // Padding in height dimension
	PadW    int // Padding in width dimension
}

// Conv2DResult contains the result of a convolution operation and any auxiliary data
type Conv2DResult struct {
	Output *tensor.Tensor
}

// ReleaseGPU releases GPU resources for the convolution result
func (cr *Conv2DResult) ReleaseGPU() {
	if cr.Output != nil {
		cr.Output.ReleaseGPU()
	}
}

// MaxPool2DResult contains the result of max pooling and indices for backward pass
type MaxPool2DResult struct {
	Output  *tensor.Tensor
	Indices *tensor.Tensor // Stores indices of max elements for backward pass
}

// ReleaseGPU releases GPU resources for the max pooling result
func (mpr *MaxPool2DResult) ReleaseGPU() {
	if mpr.Output != nil {
		mpr.Output.ReleaseGPU()
	}
	if mpr.Indices != nil {
		mpr.Indices.ReleaseGPU()
	}
}

// CalculateConv2DOutputSize calculates the output dimensions for 2D convolution
func CalculateConv2DOutputSize(inputH, inputW, kernelH, kernelW, strideH, strideW, padH, padW int) (int, int) {
	outputH := (inputH+2*padH-kernelH)/strideH + 1
	outputW := (inputW+2*padW-kernelW)/strideW + 1
	return outputH, outputW
}

// CalculatePool2DOutputSize calculates the output dimensions for 2D pooling
func CalculatePool2DOutputSize(inputH, inputW, poolH, poolW, strideH, strideW, padH, padW int) (int, int) {
	outputH := (inputH+2*padH-poolH)/strideH + 1
	outputW := (inputW+2*padW-poolW)/strideW + 1
	return outputH, outputW
}

// Conv2DForward performs 2D convolution forward pass
// Input tensor shape: [batch, height, width, channels]
// Kernel tensor shape: [kernel_height, kernel_width, input_channels, output_channels]
// Output tensor shape: [batch, output_height, output_width, output_channels]
func Conv2DForward(input, kernel *tensor.Tensor, params Conv2DParams) (*Conv2DResult, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (batch, height, width, channels), got %dD", len(input.Shape))
	}
	if len(kernel.Shape) != 4 {
		return nil, fmt.Errorf("kernel tensor must be 4D (kernel_height, kernel_width, input_channels, output_channels), got %dD", len(kernel.Shape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	inputChannels := input.Shape[3]

	kernelH := kernel.Shape[0]
	kernelW := kernel.Shape[1]
	kernelInputChannels := kernel.Shape[2]
	kernelOutputChannels := kernel.Shape[3]

	// Validate channel compatibility
	if inputChannels != kernelInputChannels {
		return nil, fmt.Errorf("input channels (%d) must match kernel input channels (%d)", inputChannels, kernelInputChannels)
	}

	// Calculate output dimensions
	outputH, outputW := CalculateConv2DOutputSize(inputH, inputW, kernelH, kernelW, params.StrideH, params.StrideW, params.PadH, params.PadW)

	if outputH <= 0 || outputW <= 0 {
		return nil, fmt.Errorf("invalid output dimensions: %dx%d (check stride and padding parameters)", outputH, outputW)
	}

	// Create output tensor
	outputShape := []int{batch, outputH, outputW, kernelOutputChannels}
	outputSize := batch * outputH * outputW * kernelOutputChannels
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := kernel.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move kernel tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_conv2d_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(inputChannels),
		C.GPUPtr(kernel.GPUPtr()), C.long(kernelH), C.long(kernelW), C.long(kernelInputChannels), C.long(kernelOutputChannels),
		C.GPUPtr(output.GPUPtr()), C.long(batch), C.long(outputH), C.long(outputW), C.long(kernelOutputChannels),
		C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D convolution forward failed (code %d): %s", retCode, errMsg)
	}

	return &Conv2DResult{Output: output}, nil
}

// Conv2DBackwardInput computes gradients with respect to input
func Conv2DBackwardInput(gradOutput, kernel *tensor.Tensor, inputShape []int, params Conv2DParams) (*tensor.Tensor, error) {
	if len(gradOutput.Shape) != 4 {
		return nil, fmt.Errorf("gradOutput tensor must be 4D, got %dD", len(gradOutput.Shape))
	}
	if len(kernel.Shape) != 4 {
		return nil, fmt.Errorf("kernel tensor must be 4D, got %dD", len(kernel.Shape))
	}
	if len(inputShape) != 4 {
		return nil, fmt.Errorf("inputShape must be 4D, got %dD", len(inputShape))
	}

	// Extract dimensions
	batch := inputShape[0]
	inputH := inputShape[1]
	inputW := inputShape[2]
	inputChannels := inputShape[3]

	kernelH := kernel.Shape[0]
	kernelW := kernel.Shape[1]
	kernelInputChannels := kernel.Shape[2]
	kernelOutputChannels := kernel.Shape[3]

	// Validate dimensions
	if inputChannels != kernelInputChannels {
		return nil, fmt.Errorf("input channels (%d) must match kernel input channels (%d)", inputChannels, kernelInputChannels)
	}

	// Create gradient input tensor
	gradInputSize := batch * inputH * inputW * inputChannels
	gradInputData := make([]float32, gradInputSize)

	gradInput, err := tensor.NewTensor(inputShape, gradInputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gradient input tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := gradOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradOutput tensor to GPU: %w", err)
	}
	if err := kernel.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move kernel tensor to GPU: %w", err)
	}
	if err := gradInput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradInput tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_conv2d_backward_input(
		C.GPUPtr(gradOutput.GPUPtr()), C.long(gradOutput.Shape[0]), C.long(gradOutput.Shape[1]), C.long(gradOutput.Shape[2]), C.long(gradOutput.Shape[3]),
		C.GPUPtr(kernel.GPUPtr()), C.long(kernelH), C.long(kernelW), C.long(kernelInputChannels), C.long(kernelOutputChannels),
		C.GPUPtr(gradInput.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(inputChannels),
		C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(gradOutput.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D convolution backward input failed (code %d): %s", retCode, errMsg)
	}

	return gradInput, nil
}

// Conv2DBackwardKernel computes gradients with respect to kernel
func Conv2DBackwardKernel(input, gradOutput *tensor.Tensor, kernelShape []int, params Conv2DParams) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D, got %dD", len(input.Shape))
	}
	if len(gradOutput.Shape) != 4 {
		return nil, fmt.Errorf("gradOutput tensor must be 4D, got %dD", len(gradOutput.Shape))
	}
	if len(kernelShape) != 4 {
		return nil, fmt.Errorf("kernelShape must be 4D, got %dD", len(kernelShape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	inputChannels := input.Shape[3]

	kernelH := kernelShape[0]
	kernelW := kernelShape[1]
	kernelInputChannels := kernelShape[2]
	kernelOutputChannels := kernelShape[3]

	// Validate dimensions
	if inputChannels != kernelInputChannels {
		return nil, fmt.Errorf("input channels (%d) must match kernel input channels (%d)", inputChannels, kernelInputChannels)
	}

	// Create gradient kernel tensor
	gradKernelSize := kernelH * kernelW * kernelInputChannels * kernelOutputChannels
	gradKernelData := make([]float32, gradKernelSize)

	gradKernel, err := tensor.NewTensor(kernelShape, gradKernelData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gradient kernel tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := gradOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradOutput tensor to GPU: %w", err)
	}
	if err := gradKernel.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradKernel tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_conv2d_backward_kernel(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(inputChannels),
		C.GPUPtr(gradOutput.GPUPtr()), C.long(gradOutput.Shape[0]), C.long(gradOutput.Shape[1]), C.long(gradOutput.Shape[2]), C.long(gradOutput.Shape[3]),
		C.GPUPtr(gradKernel.GPUPtr()), C.long(kernelH), C.long(kernelW), C.long(kernelInputChannels), C.long(kernelOutputChannels),
		C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D convolution backward kernel failed (code %d): %s", retCode, errMsg)
	}

	return gradKernel, nil
}

// MaxPool2DForward performs 2D max pooling forward pass
func MaxPool2DForward(input *tensor.Tensor, params Pool2DParams) (*MaxPool2DResult, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (batch, height, width, channels), got %dD", len(input.Shape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	channels := input.Shape[3]

	// Calculate output dimensions
	outputH, outputW := CalculatePool2DOutputSize(inputH, inputW, params.PoolH, params.PoolW, params.StrideH, params.StrideW, params.PadH, params.PadW)

	if outputH <= 0 || outputW <= 0 {
		return nil, fmt.Errorf("invalid output dimensions: %dx%d (check stride and padding parameters)", outputH, outputW)
	}

	// Create output tensor
	outputShape := []int{batch, outputH, outputW, channels}
	outputSize := batch * outputH * outputW * channels
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Create indices tensor for backward pass (stores indices of max elements)
	indicesData := make([]float32, outputSize) // We'll store indices as float32 for GPU compatibility
	indices, err := tensor.NewTensor(outputShape, indicesData)
	if err != nil {
		return nil, fmt.Errorf("failed to create indices tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}
	if err := indices.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move indices tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_maxpool2d_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.GPUPtr(output.GPUPtr()), C.long(batch), C.long(outputH), C.long(outputW), C.long(channels),
		C.GPUPtr(indices.GPUPtr()),
		C.long(params.PoolH), C.long(params.PoolW), C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D max pooling forward failed (code %d): %s", retCode, errMsg)
	}

	return &MaxPool2DResult{Output: output, Indices: indices}, nil
}

// MaxPool2DBackward performs 2D max pooling backward pass
func MaxPool2DBackward(gradOutput, indices *tensor.Tensor, inputShape []int, params Pool2DParams) (*tensor.Tensor, error) {
	if len(gradOutput.Shape) != 4 {
		return nil, fmt.Errorf("gradOutput tensor must be 4D, got %dD", len(gradOutput.Shape))
	}
	if len(inputShape) != 4 {
		return nil, fmt.Errorf("inputShape must be 4D, got %dD", len(inputShape))
	}

	// Extract dimensions
	batch := inputShape[0]
	inputH := inputShape[1]
	inputW := inputShape[2]
	channels := inputShape[3]

	// Create gradient input tensor
	gradInputSize := batch * inputH * inputW * channels
	gradInputData := make([]float32, gradInputSize)

	gradInput, err := tensor.NewTensor(inputShape, gradInputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gradient input tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := gradOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradOutput tensor to GPU: %w", err)
	}
	if err := indices.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move indices tensor to GPU: %w", err)
	}
	if err := gradInput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradInput tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_maxpool2d_backward(
		C.GPUPtr(gradOutput.GPUPtr()), C.long(gradOutput.Shape[0]), C.long(gradOutput.Shape[1]), C.long(gradOutput.Shape[2]), C.long(gradOutput.Shape[3]),
		C.GPUPtr(indices.GPUPtr()),
		C.GPUPtr(gradInput.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.long(params.PoolH), C.long(params.PoolW), C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(gradOutput.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D max pooling backward failed (code %d): %s", retCode, errMsg)
	}

	return gradInput, nil
}

// AvgPool2DForward performs 2D average pooling forward pass
func AvgPool2DForward(input *tensor.Tensor, params Pool2DParams) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (batch, height, width, channels), got %dD", len(input.Shape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	channels := input.Shape[3]

	// Calculate output dimensions
	outputH, outputW := CalculatePool2DOutputSize(inputH, inputW, params.PoolH, params.PoolW, params.StrideH, params.StrideW, params.PadH, params.PadW)

	if outputH <= 0 || outputW <= 0 {
		return nil, fmt.Errorf("invalid output dimensions: %dx%d (check stride and padding parameters)", outputH, outputW)
	}

	// Create output tensor
	outputShape := []int{batch, outputH, outputW, channels}
	outputSize := batch * outputH * outputW * channels
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_avgpool2d_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.GPUPtr(output.GPUPtr()), C.long(batch), C.long(outputH), C.long(outputW), C.long(channels),
		C.long(params.PoolH), C.long(params.PoolW), C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D average pooling forward failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// AvgPool2DBackward performs 2D average pooling backward pass
func AvgPool2DBackward(gradOutput *tensor.Tensor, inputShape []int, params Pool2DParams) (*tensor.Tensor, error) {
	if len(gradOutput.Shape) != 4 {
		return nil, fmt.Errorf("gradOutput tensor must be 4D, got %dD", len(gradOutput.Shape))
	}
	if len(inputShape) != 4 {
		return nil, fmt.Errorf("inputShape must be 4D, got %dD", len(inputShape))
	}

	// Extract dimensions
	batch := inputShape[0]
	inputH := inputShape[1]
	inputW := inputShape[2]
	channels := inputShape[3]

	// Create gradient input tensor
	gradInputSize := batch * inputH * inputW * channels
	gradInputData := make([]float32, gradInputSize)

	gradInput, err := tensor.NewTensor(inputShape, gradInputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gradient input tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := gradOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradOutput tensor to GPU: %w", err)
	}
	if err := gradInput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradInput tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_avgpool2d_backward(
		C.GPUPtr(gradOutput.GPUPtr()), C.long(gradOutput.Shape[0]), C.long(gradOutput.Shape[1]), C.long(gradOutput.Shape[2]), C.long(gradOutput.Shape[3]),
		C.GPUPtr(gradInput.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.long(params.PoolH), C.long(params.PoolW), C.long(params.StrideH), C.long(params.StrideW), C.long(params.PadH), C.long(params.PadW),
		C.DevicePtr(gradOutput.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D average pooling backward failed (code %d): %s", retCode, errMsg)
	}

	return gradInput, nil
}

// Pad2D adds padding to a 2D tensor
func Pad2D(input *tensor.Tensor, padTop, padBottom, padLeft, padRight int, padValue float32) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (batch, height, width, channels), got %dD", len(input.Shape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	channels := input.Shape[3]

	// Calculate output dimensions
	outputH := inputH + padTop + padBottom
	outputW := inputW + padLeft + padRight

	if outputH <= 0 || outputW <= 0 {
		return nil, fmt.Errorf("invalid output dimensions: %dx%d (check padding parameters)", outputH, outputW)
	}

	// Create output tensor
	outputShape := []int{batch, outputH, outputW, channels}
	outputSize := batch * outputH * outputW * channels
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_pad2d(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.GPUPtr(output.GPUPtr()), C.long(batch), C.long(outputH), C.long(outputW), C.long(channels),
		C.long(padTop), C.long(padBottom), C.long(padLeft), C.long(padRight),
		C.float(padValue),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D padding failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// Unpad2D removes padding from a 2D tensor (crop operation)
func Unpad2D(input *tensor.Tensor, padTop, padBottom, padLeft, padRight int) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (batch, height, width, channels), got %dD", len(input.Shape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	channels := input.Shape[3]

	// Calculate output dimensions
	outputH := inputH - padTop - padBottom
	outputW := inputW - padLeft - padRight

	if outputH <= 0 || outputW <= 0 {
		return nil, fmt.Errorf("invalid output dimensions: %dx%d (check padding parameters)", outputH, outputW)
	}

	// Create output tensor
	outputShape := []int{batch, outputH, outputW, channels}
	outputSize := batch * outputH * outputW * channels
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_unpad2d(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.GPUPtr(output.GPUPtr()), C.long(batch), C.long(outputH), C.long(outputW), C.long(channels),
		C.long(padTop), C.long(padBottom), C.long(padLeft), C.long(padRight),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU 2D unpadding failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// Im2Col performs the im2col operation for efficient convolution implementation
func Im2Col(input *tensor.Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (batch, height, width, channels), got %dD", len(input.Shape))
	}

	// Extract dimensions
	batch := input.Shape[0]
	inputH := input.Shape[1]
	inputW := input.Shape[2]
	channels := input.Shape[3]

	// Calculate output dimensions
	outputH := (inputH+2*padH-kernelH)/strideH + 1
	outputW := (inputW+2*padW-kernelW)/strideW + 1

	if outputH <= 0 || outputW <= 0 {
		return nil, fmt.Errorf("invalid output dimensions: %dx%d (check stride and padding parameters)", outputH, outputW)
	}

	// Im2Col output shape: (batch * output_h * output_w) x (kernel_h * kernel_w * channels)
	col_height := batch * outputH * outputW
	col_width := kernelH * kernelW * channels
	outputShape := []int{col_height, col_width}
	outputSize := col_height * col_width
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_im2col(
		C.GPUPtr(input.GPUPtr()), C.long(batch), C.long(inputH), C.long(inputW), C.long(channels),
		C.GPUPtr(output.GPUPtr()),
		C.long(kernelH), C.long(kernelW), C.long(strideH), C.long(strideW), C.long(padH), C.long(padW),
		C.long(outputH), C.long(outputW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU im2col failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// Col2Im performs the col2im operation (inverse of im2col)
func Col2Im(input *tensor.Tensor, outputShape []int, kernelH, kernelW, strideH, strideW, padH, padW int) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("input tensor must be 2D (col format), got %dD", len(input.Shape))
	}
	if len(outputShape) != 4 {
		return nil, fmt.Errorf("outputShape must be 4D, got %dD", len(outputShape))
	}

	// Extract output dimensions
	batch := outputShape[0]
	outputH := outputShape[1]
	outputW := outputShape[2]
	channels := outputShape[3]

	// Calculate input dimensions that were used for im2col
	inputH := (outputH-1)*strideH + kernelH - 2*padH
	inputW := (outputW-1)*strideW + kernelW - 2*padW

	// Create output tensor
	outputSize := batch * outputH * outputW * channels
	outputData := make([]float32, outputSize)

	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_col2im(
		C.GPUPtr(input.GPUPtr()),
		C.GPUPtr(output.GPUPtr()), C.long(batch), C.long(outputH), C.long(outputW), C.long(channels),
		C.long(kernelH), C.long(kernelW), C.long(strideH), C.long(strideW), C.long(padH), C.long(padW),
		C.long(inputH), C.long(inputW),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU col2im failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// Convenience functions for common convolution operations

// Conv2D performs a simple 2D convolution with default parameters
func Conv2D(input, kernel *tensor.Tensor, stride, padding int) (*Conv2DResult, error) {
	params := Conv2DParams{
		StrideH: stride,
		StrideW: stride,
		PadH:    padding,
		PadW:    padding,
	}
	return Conv2DForward(input, kernel, params)
}

// MaxPool2D performs 2D max pooling with default parameters
func MaxPool2D(input *tensor.Tensor, poolSize, stride, padding int) (*MaxPool2DResult, error) {
	params := Pool2DParams{
		PoolH:   poolSize,
		PoolW:   poolSize,
		StrideH: stride,
		StrideW: stride,
		PadH:    padding,
		PadW:    padding,
	}
	return MaxPool2DForward(input, params)
}

// AvgPool2D performs 2D average pooling with default parameters
func AvgPool2D(input *tensor.Tensor, poolSize, stride, padding int) (*tensor.Tensor, error) {
	params := Pool2DParams{
		PoolH:   poolSize,
		PoolW:   poolSize,
		StrideH: stride,
		StrideW: stride,
		PadH:    padding,
		PadW:    padding,
	}
	return AvgPool2DForward(input, params)
}

// ZeroPad2D adds zero padding to a tensor
func ZeroPad2D(input *tensor.Tensor, padding int) (*tensor.Tensor, error) {
	return Pad2D(input, padding, padding, padding, padding, 0.0)
}

// ReflectionPad2D adds reflection padding to a tensor
func ReflectionPad2D(input *tensor.Tensor, padding int) (*tensor.Tensor, error) {
	// Note: This is a simplified version that uses zero padding
	// A full implementation would require reflection logic in the GPU kernel
	return Pad2D(input, padding, padding, padding, padding, 0.0)
}
