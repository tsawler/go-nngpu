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

	_ "github.com/tsawler/gometal/internal/cgo"
	"github.com/tsawler/gometal/tensor"
)

// BatchNormType represents different normalization types
type BatchNormType int

const (
	BatchNorm BatchNormType = iota
	LayerNorm
	InstanceNorm
	GroupNorm
)

// String returns string representation of batch norm type
func (bnt BatchNormType) String() string {
	switch bnt {
	case BatchNorm:
		return "BatchNorm"
	case LayerNorm:
		return "LayerNorm"
	case InstanceNorm:
		return "InstanceNorm"
	case GroupNorm:
		return "GroupNorm"
	default:
		return "Unknown"
	}
}

// BatchNormResult contains the result of batch normalization and auxiliary data
type BatchNormResult struct {
	Output   *tensor.Tensor // Normalized output
	Mean     *tensor.Tensor // Computed mean (for backward pass)
	Variance *tensor.Tensor // Computed variance (for backward pass)
}

// ReleaseGPU releases GPU resources for the batch norm result
func (bnr *BatchNormResult) ReleaseGPU() {
	if bnr.Output != nil {
		bnr.Output.ReleaseGPU()
	}
	if bnr.Mean != nil {
		bnr.Mean.ReleaseGPU()
	}
	if bnr.Variance != nil {
		bnr.Variance.ReleaseGPU()
	}
}

// BatchNormGradients contains gradients from batch normalization backward pass
type BatchNormGradients struct {
	GradInput *tensor.Tensor // Gradients w.r.t. input
	GradGamma *tensor.Tensor // Gradients w.r.t. gamma (scale parameter)
	GradBeta  *tensor.Tensor // Gradients w.r.t. beta (shift parameter)
}

// ReleaseGPU releases GPU resources for the batch norm gradients
func (bng *BatchNormGradients) ReleaseGPU() {
	if bng.GradInput != nil {
		bng.GradInput.ReleaseGPU()
	}
	if bng.GradGamma != nil {
		bng.GradGamma.ReleaseGPU()
	}
	if bng.GradBeta != nil {
		bng.GradBeta.ReleaseGPU()
	}
}

// BatchMean computes the mean across the batch dimension
func BatchMean(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("batch mean requires 2D tensor (batch_size, features), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	features := input.Shape[1]

	// Create mean tensor
	meanData := make([]float32, features)
	meanTensor, err := tensor.NewTensor([]int{features}, meanData)
	if err != nil {
		return nil, fmt.Errorf("failed to create mean tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := meanTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move mean tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_batch_mean(
		C.GPUPtr(input.GPUPtr()), C.long(batchSize), C.long(features),
		C.GPUPtr(meanTensor.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU batch mean failed (code %d): %s", retCode, errMsg)
	}

	return meanTensor, nil
}

// BatchVariance computes the variance across the batch dimension
func BatchVariance(input, mean *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("batch variance requires 2D tensor (batch_size, features), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	features := input.Shape[1]

	if len(mean.Shape) != 1 || mean.Shape[0] != features {
		return nil, fmt.Errorf("mean tensor must have shape [%d], got %v", features, mean.Shape)
	}

	// Create variance tensor
	varianceData := make([]float32, features)
	varianceTensor, err := tensor.NewTensor([]int{features}, varianceData)
	if err != nil {
		return nil, fmt.Errorf("failed to create variance tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := mean.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move mean tensor to GPU: %w", err)
	}
	if err := varianceTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move variance tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_batch_variance(
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(mean.GPUPtr()),
		C.long(batchSize), C.long(features),
		C.GPUPtr(varianceTensor.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU batch variance failed (code %d): %s", retCode, errMsg)
	}

	return varianceTensor, nil
}

// BatchNormForward performs batch normalization forward pass
func BatchNormForward(input, mean, variance, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("batch norm forward requires 2D tensor (batch_size, features), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	features := input.Shape[1]

	// Validate parameter shapes
	if len(mean.Shape) != 1 || mean.Shape[0] != features {
		return nil, fmt.Errorf("mean tensor must have shape [%d], got %v", features, mean.Shape)
	}
	if len(variance.Shape) != 1 || variance.Shape[0] != features {
		return nil, fmt.Errorf("variance tensor must have shape [%d], got %v", features, variance.Shape)
	}
	if len(gamma.Shape) != 1 || gamma.Shape[0] != features {
		return nil, fmt.Errorf("gamma tensor must have shape [%d], got %v", features, gamma.Shape)
	}
	if len(beta.Shape) != 1 || beta.Shape[0] != features {
		return nil, fmt.Errorf("beta tensor must have shape [%d], got %v", features, beta.Shape)
	}

	// Create output tensor
	outputData := make([]float32, batchSize*features)
	outputTensor, err := tensor.NewTensor([]int{batchSize, features}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := mean.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move mean tensor to GPU: %w", err)
	}
	if err := variance.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move variance tensor to GPU: %w", err)
	}
	if err := gamma.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gamma tensor to GPU: %w", err)
	}
	if err := beta.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move beta tensor to GPU: %w", err)
	}
	if err := outputTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_batch_norm_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batchSize), C.long(features),
		C.GPUPtr(mean.GPUPtr()), C.GPUPtr(variance.GPUPtr()),
		C.GPUPtr(gamma.GPUPtr()), C.GPUPtr(beta.GPUPtr()),
		C.float(epsilon),
		C.GPUPtr(outputTensor.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU batch norm forward failed (code %d): %s", retCode, errMsg)
	}

	return outputTensor, nil
}

// BatchNormBackward performs batch normalization backward pass
func BatchNormBackward(gradOutput, input, mean, variance, gamma *tensor.Tensor, epsilon float32) (*BatchNormGradients, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("batch norm backward requires 2D tensor (batch_size, features), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	features := input.Shape[1]

	// Create gradient tensors
	gradInputData := make([]float32, batchSize*features)
	gradGammaData := make([]float32, features)
	gradBetaData := make([]float32, features)

	gradInput, err := tensor.NewTensor([]int{batchSize, features}, gradInputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create grad input tensor: %w", err)
	}

	gradGamma, err := tensor.NewTensor([]int{features}, gradGammaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create grad gamma tensor: %w", err)
	}

	gradBeta, err := tensor.NewTensor([]int{features}, gradBetaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create grad beta tensor: %w", err)
	}

	// Ensure all tensors are on GPU
	tensors := []*tensor.Tensor{gradOutput, input, mean, variance, gamma, gradInput, gradGamma, gradBeta}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError

	// Compute gradients w.r.t. input
	retCode := C.perform_batch_norm_backward_input(
		C.GPUPtr(gradOutput.GPUPtr()), C.long(batchSize), C.long(features),
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(mean.GPUPtr()), C.GPUPtr(variance.GPUPtr()),
		C.GPUPtr(gamma.GPUPtr()), C.float(epsilon),
		C.GPUPtr(gradInput.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU batch norm backward input failed (code %d): %s", retCode, errMsg)
	}

	// Compute gradients w.r.t. parameters
	retCode = C.perform_batch_norm_backward_params(
		C.GPUPtr(gradOutput.GPUPtr()), C.long(batchSize), C.long(features),
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(mean.GPUPtr()), C.GPUPtr(variance.GPUPtr()),
		C.float(epsilon),
		C.GPUPtr(gradGamma.GPUPtr()), C.GPUPtr(gradBeta.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU batch norm backward params failed (code %d): %s", retCode, errMsg)
	}

	return &BatchNormGradients{
		GradInput: gradInput,
		GradGamma: gradGamma,
		GradBeta:  gradBeta,
	}, nil
}

// LayerNormForward performs layer normalization forward pass
func LayerNormForward(input, gamma, beta *tensor.Tensor, epsilon float32) (*BatchNormResult, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("layer norm forward requires 2D tensor (batch_size, features), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	features := input.Shape[1]

	// Create output and statistics tensors
	outputData := make([]float32, batchSize*features)
	meanData := make([]float32, batchSize)     // Per-sample means
	varianceData := make([]float32, batchSize) // Per-sample variances

	outputTensor, err := tensor.NewTensor([]int{batchSize, features}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	meanTensor, err := tensor.NewTensor([]int{batchSize}, meanData)
	if err != nil {
		return nil, fmt.Errorf("failed to create mean tensor: %w", err)
	}

	varianceTensor, err := tensor.NewTensor([]int{batchSize}, varianceData)
	if err != nil {
		return nil, fmt.Errorf("failed to create variance tensor: %w", err)
	}

	// Ensure tensors are on GPU
	tensors := []*tensor.Tensor{input, gamma, beta, outputTensor, meanTensor, varianceTensor}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_layer_norm_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batchSize), C.long(features),
		C.GPUPtr(gamma.GPUPtr()), C.GPUPtr(beta.GPUPtr()),
		C.float(epsilon),
		C.GPUPtr(outputTensor.GPUPtr()), C.GPUPtr(meanTensor.GPUPtr()), C.GPUPtr(varianceTensor.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU layer norm forward failed (code %d): %s", retCode, errMsg)
	}

	return &BatchNormResult{
		Output:   outputTensor,
		Mean:     meanTensor,
		Variance: varianceTensor,
	}, nil
}

// LayerNormBackward performs layer normalization backward pass
func LayerNormBackward(gradOutput, input, mean, variance, gamma *tensor.Tensor, epsilon float32) (*BatchNormGradients, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("layer norm backward requires 2D tensor (batch_size, features), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	features := input.Shape[1]

	// Create gradient tensors
	gradInputData := make([]float32, batchSize*features)
	gradGammaData := make([]float32, features)
	gradBetaData := make([]float32, features)

	gradInput, err := tensor.NewTensor([]int{batchSize, features}, gradInputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create grad input tensor: %w", err)
	}

	gradGamma, err := tensor.NewTensor([]int{features}, gradGammaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create grad gamma tensor: %w", err)
	}

	gradBeta, err := tensor.NewTensor([]int{features}, gradBetaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create grad beta tensor: %w", err)
	}

	// Ensure all tensors are on GPU
	tensors := []*tensor.Tensor{gradOutput, input, mean, variance, gamma, gradInput, gradGamma, gradBeta}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_layer_norm_backward(
		C.GPUPtr(gradOutput.GPUPtr()), C.long(batchSize), C.long(features),
		C.GPUPtr(input.GPUPtr()), C.GPUPtr(mean.GPUPtr()), C.GPUPtr(variance.GPUPtr()),
		C.GPUPtr(gamma.GPUPtr()), C.float(epsilon),
		C.GPUPtr(gradInput.GPUPtr()), C.GPUPtr(gradGamma.GPUPtr()), C.GPUPtr(gradBeta.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU layer norm backward failed (code %d): %s", retCode, errMsg)
	}

	return &BatchNormGradients{
		GradInput: gradInput,
		GradGamma: gradGamma,
		GradBeta:  gradBeta,
	}, nil
}

// UpdateRunningStats updates running mean and variance for batch normalization
func UpdateRunningStats(runningMean, runningVar, batchMean, batchVar *tensor.Tensor, momentum float32) error {
	if len(runningMean.Shape) != 1 || len(runningVar.Shape) != 1 {
		return fmt.Errorf("running statistics must be 1D tensors")
	}

	features := runningMean.Shape[0]
	if runningVar.Shape[0] != features || batchMean.Shape[0] != features || batchVar.Shape[0] != features {
		return fmt.Errorf("all statistics tensors must have the same size")
	}

	// Ensure tensors are on GPU
	tensors := []*tensor.Tensor{runningMean, runningVar, batchMean, batchVar}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_update_running_stats(
		C.GPUPtr(runningMean.GPUPtr()), C.GPUPtr(runningVar.GPUPtr()),
		C.GPUPtr(batchMean.GPUPtr()), C.GPUPtr(batchVar.GPUPtr()),
		C.float(momentum), C.long(features),
		C.DevicePtr(runningMean.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU running stats update failed (code %d): %s", retCode, errMsg)
	}

	return nil
}

// InstanceNormForward performs instance normalization forward pass
func InstanceNormForward(input, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("instance norm requires 4D tensor (batch, channels, height, width), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	channels := input.Shape[1]
	height := input.Shape[2]
	width := input.Shape[3]

	// Create output tensor
	outputData := make([]float32, batchSize*channels*height*width)
	outputTensor, err := tensor.NewTensor(input.Shape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	tensors := []*tensor.Tensor{input, gamma, beta, outputTensor}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_instance_norm_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batchSize), C.long(channels), C.long(height), C.long(width),
		C.GPUPtr(gamma.GPUPtr()), C.GPUPtr(beta.GPUPtr()),
		C.float(epsilon),
		C.GPUPtr(outputTensor.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU instance norm forward failed (code %d): %s", retCode, errMsg)
	}

	return outputTensor, nil
}

// GroupNormForward performs group normalization forward pass
func GroupNormForward(input, gamma, beta *tensor.Tensor, numGroups int, epsilon float32) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("group norm requires 4D tensor (batch, channels, height, width), got %dD", len(input.Shape))
	}

	batchSize := input.Shape[0]
	channels := input.Shape[1]
	height := input.Shape[2]
	width := input.Shape[3]

	if channels%numGroups != 0 {
		return nil, fmt.Errorf("number of channels (%d) must be divisible by number of groups (%d)", channels, numGroups)
	}

	// Create output tensor
	outputData := make([]float32, batchSize*channels*height*width)
	outputTensor, err := tensor.NewTensor(input.Shape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	tensors := []*tensor.Tensor{input, gamma, beta, outputTensor}
	for _, t := range tensors {
		if err := t.EnsureGPU(); err != nil {
			return nil, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_group_norm_forward(
		C.GPUPtr(input.GPUPtr()), C.long(batchSize), C.long(channels), C.long(height), C.long(width),
		C.long(numGroups), C.GPUPtr(gamma.GPUPtr()), C.GPUPtr(beta.GPUPtr()),
		C.float(epsilon),
		C.GPUPtr(outputTensor.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU group norm forward failed (code %d): %s", retCode, errMsg)
	}

	return outputTensor, nil
}

// Convenience functions for common batch normalization operations

// BatchNormForwardBackward performs both forward and backward passes efficiently
func BatchNormForwardBackward(input, mean, variance, gamma, beta, gradOutput *tensor.Tensor, epsilon float32) (*tensor.Tensor, *BatchNormGradients, error) {
	output, err := BatchNormForward(input, mean, variance, gamma, beta, epsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("forward pass failed: %w", err)
	}

	gradients, err := BatchNormBackward(gradOutput, input, mean, variance, gamma, epsilon)
	if err != nil {
		output.ReleaseGPU()
		return nil, nil, fmt.Errorf("backward pass failed: %w", err)
	}

	return output, gradients, nil
}

// LayerNormForwardBackward performs both forward and backward passes for layer normalization
func LayerNormForwardBackward(input, gamma, beta, gradOutput *tensor.Tensor, epsilon float32) (*BatchNormResult, *BatchNormGradients, error) {
	result, err := LayerNormForward(input, gamma, beta, epsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("forward pass failed: %w", err)
	}

	gradients, err := LayerNormBackward(gradOutput, input, result.Mean, result.Variance, gamma, epsilon)
	if err != nil {
		result.ReleaseGPU()
		return nil, nil, fmt.Errorf("backward pass failed: %w", err)
	}

	return result, gradients, nil
}

// BatchNormInference performs batch normalization in inference mode using running statistics
func BatchNormInference(input, runningMean, runningVar, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	return BatchNormForward(input, runningMean, runningVar, gamma, beta, epsilon)
}

// BatchNormTraining performs batch normalization in training mode, computing batch statistics
func BatchNormTraining(input, gamma, beta *tensor.Tensor, epsilon float32) (*BatchNormResult, error) {
	// Compute batch statistics
	mean, err := BatchMean(input)
	if err != nil {
		return nil, fmt.Errorf("failed to compute batch mean: %w", err)
	}

	variance, err := BatchVariance(input, mean)
	if err != nil {
		mean.ReleaseGPU()
		return nil, fmt.Errorf("failed to compute batch variance: %w", err)
	}

	// Apply batch normalization
	output, err := BatchNormForward(input, mean, variance, gamma, beta, epsilon)
	if err != nil {
		mean.ReleaseGPU()
		variance.ReleaseGPU()
		return nil, fmt.Errorf("failed to apply batch normalization: %w", err)
	}

	return &BatchNormResult{
		Output:   output,
		Mean:     mean,
		Variance: variance,
	}, nil
}
