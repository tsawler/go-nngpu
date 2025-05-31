package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"

import (
	"fmt"

	"github.com/tsawler/go-nngpu/tensor"
)

// This file contains higher-level gradient-aware operations that build on the
// gradient computation framework. These functions automatically track gradients
// when called in gradient mode.

// Linear performs a linear transformation: output = input * weight + bias
func Linear(input, weight, bias *GradientTensor) (*GradientTensor, error) {
	// Forward pass: output = input * weight + bias
	matmulResult, err := GradMatMul(input, weight)
	if err != nil {
		return nil, fmt.Errorf("matrix multiplication failed in linear layer: %w", err)
	}

	if bias != nil {
		return GradAdd(matmulResult, bias)
	}

	return matmulResult, nil
}

// Conv2D performs gradient-aware 2D convolution
func GradConv2D(input, kernel *GradientTensor, params Conv2DParams) (*GradientTensor, error) {
	// Perform forward convolution
	result, err := Conv2DForward(input.Tensor, kernel.Tensor, params)
	if err != nil {
		return nil, fmt.Errorf("forward convolution failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result.Output,
		RequiresGrad: input.RequiresGrad || kernel.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		// Create backward function for convolution
		metadata := map[string]interface{}{
			"params": params,
		}
		resultGT.GradFn = CreateBackwardFunction(OpConv2D, []*GradientTensor{input, kernel},
			[]*tensor.Tensor{input.Tensor, kernel.Tensor}, metadata)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// MaxPool2D performs gradient-aware 2D max pooling
func GradMaxPool2D(input *GradientTensor, params Pool2DParams) (*GradientTensor, error) {
	// Perform forward max pooling
	result, err := MaxPool2DForward(input.Tensor, params)
	if err != nil {
		return nil, fmt.Errorf("forward max pooling failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result.Output,
		RequiresGrad: input.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		metadata := map[string]interface{}{
			"params":  params,
			"indices": result.Indices, // Save indices for backward pass
		}
		resultGT.GradFn = CreateBackwardFunction(OpMaxPool2D, []*GradientTensor{input},
			[]*tensor.Tensor{input.Tensor}, metadata)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// BatchNorm performs gradient-aware batch normalization
func GradBatchNorm(input, gamma, beta *GradientTensor, epsilon float32) (*GradientTensor, error) {
	// Compute batch statistics for training
	bnResult, err := BatchNormTraining(input.Tensor, gamma.Tensor, beta.Tensor, epsilon)
	if err != nil {
		return nil, fmt.Errorf("forward batch normalization failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       bnResult.Output,
		RequiresGrad: input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		metadata := map[string]interface{}{
			"epsilon":  epsilon,
			"mean":     bnResult.Mean,
			"variance": bnResult.Variance,
		}
		resultGT.GradFn = CreateBackwardFunction(OpBatchNorm, []*GradientTensor{input, gamma, beta},
			[]*tensor.Tensor{input.Tensor, gamma.Tensor, beta.Tensor}, metadata)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// Dropout performs gradient-aware dropout for regularization
func GradDropout(input *GradientTensor, probability float32, training bool, seed uint32) (*GradientTensor, error) {
	if !training {
		// During inference, dropout is a no-op
		return input, nil
	}

	// Create mask tensor for dropout
	maskData := make([]float32, len(input.Tensor.Data))
	mask, err := tensor.NewTensor(input.Tensor.Shape, maskData)
	if err != nil {
		return nil, fmt.Errorf("failed to create dropout mask: %w", err)
	}

	// Create output tensor
	outputData := make([]float32, len(input.Tensor.Data))
	output, err := tensor.NewTensor(input.Tensor.Shape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create dropout output: %w", err)
	}

	// Perform dropout forward pass
	err = DropoutForward(input.Tensor, output, mask, probability, seed)
	if err != nil {
		return nil, fmt.Errorf("dropout forward failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       output,
		RequiresGrad: input.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		metadata := map[string]interface{}{
			"probability": probability,
			"mask":        mask,
		}
		resultGT.GradFn = &GradientFunction{
			OpType:   OpReshape, // Reuse as placeholder
			Inputs:   []*GradientTensor{input},
			Metadata: metadata,
			BackwardFn: func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
				// Apply dropout mask to gradients
				gradInput, err := DropoutBackward(gradOutput, mask, probability)
				if err != nil {
					return nil, fmt.Errorf("dropout backward failed: %w", err)
				}
				return []*tensor.Tensor{gradInput}, nil
			},
		}
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// Helper functions for dropout operations

// DropoutForward applies dropout during forward pass
func DropoutForward(input, output, mask *tensor.Tensor, probability float32, seed uint32) error {
	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move input to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move output to GPU: %w", err)
	}
	if err := mask.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move mask to GPU: %w", err)
	}

	var cErr C.CError
	retCode := C.perform_dropout_forward(
		C.GPUPtr(input.GPUPtr()),
		C.long(len(input.Data)),
		C.float(probability),
		C.uint(seed),
		C.GPUPtr(output.GPUPtr()),
		C.GPUPtr(mask.GPUPtr()),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU dropout forward failed (code %d): %s", retCode, errMsg)
	}

	return nil
}

// DropoutBackward applies dropout mask during backward pass
func DropoutBackward(gradOutput, mask *tensor.Tensor, probability float32) (*tensor.Tensor, error) {
	// Create gradient input tensor
	gradInputData := make([]float32, len(gradOutput.Data))
	gradInput, err := tensor.NewTensor(gradOutput.Shape, gradInputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gradient input: %w", err)
	}

	// Ensure tensors are on GPU
	if err := gradOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move grad output to GPU: %w", err)
	}
	if err := mask.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move mask to GPU: %w", err)
	}
	if err := gradInput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move grad input to GPU: %w", err)
	}

	var cErr C.CError
	retCode := C.perform_dropout_backward(
		C.GPUPtr(gradOutput.GPUPtr()),
		C.GPUPtr(mask.GPUPtr()),
		C.long(len(gradOutput.Data)),
		C.float(probability),
		C.GPUPtr(gradInput.GPUPtr()),
		C.DevicePtr(gradOutput.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU dropout backward failed (code %d): %s", retCode, errMsg)
	}

	return gradInput, nil
}

// Cross-entropy loss with built-in softmax
func GradCrossEntropyLoss(predictions, targets *GradientTensor) (*GradientTensor, error) {
	// Apply softmax to predictions first
	softmaxOutput, err := GradSoftmax(predictions)
	if err != nil {
		return nil, fmt.Errorf("softmax failed in cross-entropy loss: %w", err)
	}

	// Compute cross-entropy loss
	loss, err := CategoricalCrossEntropyLoss(softmaxOutput.Tensor, targets.Tensor)
	if err != nil {
		return nil, fmt.Errorf("cross-entropy loss computation failed: %w", err)
	}

	// Create scalar loss tensor
	lossData := []float32{loss}
	lossTensor, err := tensor.NewTensor([]int{1}, lossData)
	if err != nil {
		return nil, fmt.Errorf("failed to create loss tensor: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       lossTensor,
		RequiresGrad: predictions.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpCrossEntropyLoss, []*GradientTensor{predictions, targets},
			[]*tensor.Tensor{softmaxOutput.Tensor, targets.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// Sparse cross-entropy loss for integer targets
func GradSparseCrossEntropyLoss(predictions *GradientTensor, targetIndices []int) (*GradientTensor, error) {
	// Compute sparse cross-entropy loss
	loss, err := SparseCategoricalCrossEntropyForward(predictions.Tensor, targetIndices)
	if err != nil {
		return nil, fmt.Errorf("sparse cross-entropy loss computation failed: %w", err)
	}

	// Create scalar loss tensor
	lossData := []float32{loss}
	lossTensor, err := tensor.NewTensor([]int{1}, lossData)
	if err != nil {
		return nil, fmt.Errorf("failed to create loss tensor: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       lossTensor,
		RequiresGrad: predictions.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		metadata := map[string]interface{}{
			"targetIndices": targetIndices,
		}
		resultGT.GradFn = &GradientFunction{
			OpType:   OpCrossEntropyLoss,
			Inputs:   []*GradientTensor{predictions},
			Metadata: metadata,
			BackwardFn: func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
				gradInput, err := SparseCategoricalCrossEntropyBackward(predictions.Tensor, targetIndices)
				if err != nil {
					return nil, fmt.Errorf("sparse cross-entropy backward failed: %w", err)
				}

				// Scale by incoming gradient if needed
				if gradOutput != nil && len(gradOutput.Data) == 1 && gradOutput.Data[0] != 1.0 {
					gradInput, err = ScalarMul(gradInput, gradOutput.Data[0])
					if err != nil {
						return nil, fmt.Errorf("failed to scale gradient: %w", err)
					}
				}

				return []*tensor.Tensor{gradInput}, nil
			},
		}
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// Sequence operations for RNNs and transformers

// Flatten reshapes a multi-dimensional tensor to 2D
func GradFlatten(input *GradientTensor, startDim int) (*GradientTensor, error) {
	if startDim < 0 || startDim >= len(input.Tensor.Shape) {
		return nil, fmt.Errorf("start dimension %d out of bounds", startDim)
	}

	// Calculate new shape
	flattenedSize := 1
	for i := startDim; i < len(input.Tensor.Shape); i++ {
		flattenedSize *= input.Tensor.Shape[i]
	}

	newShape := make([]int, startDim+1)
	copy(newShape[:startDim], input.Tensor.Shape[:startDim])
	newShape[startDim] = flattenedSize

	return GradReshape(input, newShape)
}

// Squeeze removes dimensions of size 1
func GradSqueeze(input *GradientTensor, dim int) (*GradientTensor, error) {
	if dim < 0 || dim >= len(input.Tensor.Shape) {
		return nil, fmt.Errorf("dimension %d out of bounds", dim)
	}

	if input.Tensor.Shape[dim] != 1 {
		return nil, fmt.Errorf("cannot squeeze dimension %d with size %d", dim, input.Tensor.Shape[dim])
	}

	// Create new shape without the squeezed dimension
	newShape := make([]int, 0, len(input.Tensor.Shape)-1)
	for i, size := range input.Tensor.Shape {
		if i != dim {
			newShape = append(newShape, size)
		}
	}

	return GradReshape(input, newShape)
}

// Unsqueeze adds a dimension of size 1
func GradUnsqueeze(input *GradientTensor, dim int) (*GradientTensor, error) {
	if dim < 0 || dim > len(input.Tensor.Shape) {
		return nil, fmt.Errorf("dimension %d out of bounds for unsqueeze", dim)
	}

	// Create new shape with an additional dimension of size 1
	newShape := make([]int, len(input.Tensor.Shape)+1)
	for i := 0; i < dim; i++ {
		newShape[i] = input.Tensor.Shape[i]
	}
	newShape[dim] = 1
	for i := dim; i < len(input.Tensor.Shape); i++ {
		newShape[i+1] = input.Tensor.Shape[i]
	}

	return GradReshape(input, newShape)
}

// Concatenation along a specific dimension
func GradCat(tensors []*GradientTensor, dim int) (*GradientTensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("cannot concatenate empty tensor list")
	}

	// Verify all tensors have the same shape except along concat dimension
	refShape := tensors[0].Tensor.Shape
	if dim < 0 || dim >= len(refShape) {
		return nil, fmt.Errorf("concatenation dimension %d out of bounds", dim)
	}

	for i, gt := range tensors[1:] {
		if len(gt.Tensor.Shape) != len(refShape) {
			return nil, fmt.Errorf("tensor %d has different number of dimensions", i+1)
		}
		for j, size := range gt.Tensor.Shape {
			if j != dim && size != refShape[j] {
				return nil, fmt.Errorf("tensor %d has incompatible shape at dimension %d", i+1, j)
			}
		}
	}

	// Calculate output shape
	outputShape := make([]int, len(refShape))
	copy(outputShape, refShape)
	for _, gt := range tensors[1:] {
		outputShape[dim] += gt.Tensor.Shape[dim]
	}

	// Perform concatenation on CPU for simplicity
	outputSize := 1
	for _, size := range outputShape {
		outputSize *= size
	}

	outputData := make([]float32, outputSize)
	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Simple concatenation implementation (could be optimized)
	err = concatenateTensors(tensors, output, dim)
	if err != nil {
		return nil, fmt.Errorf("concatenation failed: %w", err)
	}

	// Check if any input requires gradients
	requiresGrad := false
	for _, gt := range tensors {
		if gt.RequiresGrad {
			requiresGrad = true
			break
		}
	}

	resultGT := &GradientTensor{
		Tensor:       output,
		RequiresGrad: requiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		metadata := map[string]interface{}{
			"dim":    dim,
			"shapes": extractShapes(tensors),
		}

		// Collect input tensors
		inputTensors := make([]*tensor.Tensor, len(tensors))
		for i, gt := range tensors {
			inputTensors[i] = gt.Tensor
		}

		resultGT.GradFn = &GradientFunction{
			OpType:       OpReshape, // Reuse as placeholder
			Inputs:       tensors,
			SavedTensors: inputTensors,
			Metadata:     metadata,
			BackwardFn: func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
				return splitGradient(gradOutput, metadata["shapes"].([][]int), dim)
			},
		}
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// Helper functions for concatenation

func extractShapes(tensors []*GradientTensor) [][]int {
	shapes := make([][]int, len(tensors))
	for i, gt := range tensors {
		shapes[i] = make([]int, len(gt.Tensor.Shape))
		copy(shapes[i], gt.Tensor.Shape)
	}
	return shapes
}

func concatenateTensors(inputs []*GradientTensor, output *tensor.Tensor, dim int) error {
	// Retrieve all input data to CPU
	for _, input := range inputs {
		if err := input.Tensor.RetrieveCPU(); err != nil {
			return fmt.Errorf("failed to retrieve input tensor: %w", err)
		}
	}

	// Simple concatenation along specified dimension
	// This is a basic implementation - could be optimized with GPU kernels
	outputIdx := 0

	// Calculate strides for each dimension
	strides := make([]int, len(output.Shape))
	strides[len(strides)-1] = 1
	for i := len(strides) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * output.Shape[i+1]
	}

	// Copy data from each input tensor
	dimOffset := 0
	for _, input := range inputs {
		inputSize := len(input.Tensor.Data)
		copy(output.Data[outputIdx:outputIdx+inputSize], input.Tensor.Data)
		outputIdx += inputSize
		dimOffset += input.Tensor.Shape[dim]
	}

	return nil
}

func splitGradient(gradOutput *tensor.Tensor, shapes [][]int, dim int) ([]*tensor.Tensor, error) {
	// Split gradient back to original tensor shapes
	gradients := make([]*tensor.Tensor, len(shapes))

	if err := gradOutput.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve gradient output: %w", err)
	}

	dataOffset := 0
	for i, shape := range shapes {
		tensorSize := 1
		for _, s := range shape {
			tensorSize *= s
		}

		gradData := make([]float32, tensorSize)
		copy(gradData, gradOutput.Data[dataOffset:dataOffset+tensorSize])

		grad, err := tensor.NewTensor(shape, gradData)
		if err != nil {
			return nil, fmt.Errorf("failed to create gradient tensor %d: %w", i, err)
		}

		gradients[i] = grad
		dataOffset += tensorSize
	}

	return gradients, nil
}
