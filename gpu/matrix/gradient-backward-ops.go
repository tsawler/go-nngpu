package matrix

import (
	"fmt"

	"github.com/tsawler/gometal/tensor"
)

// This file contains the missing backward function implementations for
// convolution and batch normalization operations used in the gradient framework

// Update the gradient.go backward functions to include these implementations

// Add these backward function implementations to the GradientFunction methods

func (gf *GradientFunction) conv2dBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for Conv2D backward")
	}

	input := gf.SavedTensors[0]
	kernel := gf.SavedTensors[1]

	// Get parameters from metadata
	params, ok := gf.Metadata["params"].(Conv2DParams)
	if !ok {
		return nil, fmt.Errorf("missing or invalid Conv2D parameters in metadata")
	}

	// Compute gradients with respect to input
	gradInput, err := Conv2DBackwardInput(gradOutput, kernel, input.Shape, params)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Conv2D input gradients: %w", err)
	}

	// Compute gradients with respect to kernel
	gradKernel, err := Conv2DBackwardKernel(input, gradOutput, kernel.Shape, params)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Conv2D kernel gradients: %w", err)
	}

	return []*tensor.Tensor{gradInput, gradKernel}, nil
}

func (gf *GradientFunction) maxPool2dBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.Inputs) < 1 {
		return nil, fmt.Errorf("no input tensors for MaxPool2D backward")
	}

	input := gf.Inputs[0].Tensor

	// Get parameters and indices from metadata
	params, ok := gf.Metadata["params"].(Pool2DParams)
	if !ok {
		return nil, fmt.Errorf("missing or invalid MaxPool2D parameters in metadata")
	}

	indices, ok := gf.Metadata["indices"].(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("missing or invalid indices in MaxPool2D metadata")
	}

	// Compute gradients with respect to input
	gradInput, err := MaxPool2DBackward(gradOutput, indices, input.Shape, params)
	if err != nil {
		return nil, fmt.Errorf("failed to compute MaxPool2D input gradients: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) avgPool2dBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.Inputs) < 1 {
		return nil, fmt.Errorf("no input tensors for AvgPool2D backward")
	}

	input := gf.Inputs[0].Tensor

	// Get parameters from metadata
	params, ok := gf.Metadata["params"].(Pool2DParams)
	if !ok {
		return nil, fmt.Errorf("missing or invalid AvgPool2D parameters in metadata")
	}

	// Compute gradients with respect to input
	gradInput, err := AvgPool2DBackward(gradOutput, input.Shape, params)
	if err != nil {
		return nil, fmt.Errorf("failed to compute AvgPool2D input gradients: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) batchNormBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 3 {
		return nil, fmt.Errorf("insufficient saved tensors for BatchNorm backward")
	}

	input := gf.SavedTensors[0]
	gamma := gf.SavedTensors[1]
	// beta := gf.SavedTensors[2]

	// Get parameters from metadata
	epsilon, ok := gf.Metadata["epsilon"].(float32)
	if !ok {
		return nil, fmt.Errorf("missing or invalid epsilon in BatchNorm metadata")
	}

	mean, ok := gf.Metadata["mean"].(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("missing or invalid mean in BatchNorm metadata")
	}

	variance, ok := gf.Metadata["variance"].(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("missing or invalid variance in BatchNorm metadata")
	}

	// Compute gradients
	gradients, err := BatchNormBackward(gradOutput, input, mean, variance, gamma, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to compute BatchNorm gradients: %w", err)
	}

	return []*tensor.Tensor{gradients.GradInput, gradients.GradGamma, gradients.GradBeta}, nil
}

func (gf *GradientFunction) layerNormBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 3 {
		return nil, fmt.Errorf("insufficient saved tensors for LayerNorm backward")
	}

	input := gf.SavedTensors[0]
	gamma := gf.SavedTensors[1]
	// beta := gf.SavedTensors[2]

	// Get parameters from metadata
	epsilon, ok := gf.Metadata["epsilon"].(float32)
	if !ok {
		return nil, fmt.Errorf("missing or invalid epsilon in LayerNorm metadata")
	}

	mean, ok := gf.Metadata["mean"].(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("missing or invalid mean in LayerNorm metadata")
	}

	variance, ok := gf.Metadata["variance"].(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("missing or invalid variance in LayerNorm metadata")
	}

	// Compute gradients
	gradients, err := LayerNormBackward(gradOutput, input, mean, variance, gamma, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to compute LayerNorm gradients: %w", err)
	}

	return []*tensor.Tensor{gradients.GradInput, gradients.GradGamma, gradients.GradBeta}, nil
}

// Update the CreateBackwardFunction to include the new operations
func UpdateCreateBackwardFunction() {
	// This function would update the CreateBackwardFunction to include:
	// case OpConv2D: gf.BackwardFn = gf.conv2dBackward
	// case OpMaxPool2D: gf.BackwardFn = gf.maxPool2dBackward
	// case OpAvgPool2D: gf.BackwardFn = gf.avgPool2dBackward
	// case OpBatchNorm: gf.BackwardFn = gf.batchNormBackward
	// case OpLayerNorm: gf.BackwardFn = gf.layerNormBackward
}

// Enhanced activation function backward implementations with better error handling

func (gf *GradientFunction) leakyReluBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for LeakyReLU backward")
	}

	activationOutput := gf.SavedTensors[0]

	// Get alpha parameter from metadata
	alpha, ok := gf.Metadata["alpha"].(float32)
	if !ok {
		alpha = 0.01 // Default value
	}

	gradInput, err := LeakyReLUBackward(gradOutput, activationOutput, alpha)
	if err != nil {
		return nil, fmt.Errorf("failed to compute LeakyReLU backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) eluBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for ELU backward")
	}

	activationOutput := gf.SavedTensors[0]

	// Get alpha parameter from metadata
	alpha, ok := gf.Metadata["alpha"].(float32)
	if !ok {
		alpha = 1.0 // Default value
	}

	gradInput, err := ELUBackward(gradOutput, activationOutput, alpha)
	if err != nil {
		return nil, fmt.Errorf("failed to compute ELU backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) swishBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for Swish backward")
	}

	activationOutput := gf.SavedTensors[0]

	gradInput, err := SwishBackward(gradOutput, activationOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Swish backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) geluBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for GELU backward")
	}

	activationOutput := gf.SavedTensors[0]

	gradInput, err := GELUBackward(gradOutput, activationOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute GELU backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

// Additional loss function backward implementations

func (gf *GradientFunction) binaryCrossEntropyLossBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for binary cross-entropy loss backward")
	}

	predictions := gf.SavedTensors[0]
	targets := gf.SavedTensors[1]

	gradInput, err := BinaryCrossEntropyLossGradients(predictions, targets)
	if err != nil {
		return nil, fmt.Errorf("failed to compute binary cross-entropy loss gradients: %w", err)
	}

	// Scale by incoming gradient
	if gradOutput != nil {
		gradInput, err = Mul(gradInput, gradOutput)
		if err != nil {
			return nil, fmt.Errorf("failed to scale gradient: %w", err)
		}
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) huberLossBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for Huber loss backward")
	}

	predictions := gf.SavedTensors[0]
	targets := gf.SavedTensors[1]

	// Get delta parameter from metadata
	delta, ok := gf.Metadata["delta"].(float32)
	if !ok {
		return nil, fmt.Errorf("missing or invalid delta in Huber loss metadata")
	}

	gradInput, err := HuberLossGradients(predictions, targets, delta)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Huber loss gradients: %w", err)
	}

	// Scale by incoming gradient
	if gradOutput != nil {
		gradInput, err = Mul(gradInput, gradOutput)
		if err != nil {
			return nil, fmt.Errorf("failed to scale gradient: %w", err)
		}
	}

	return []*tensor.Tensor{gradInput}, nil
}

// Utility functions for gradient operations

// CloneGradientTensor creates a deep copy of a gradient tensor
func CloneGradientTensor(gt *GradientTensor) (*GradientTensor, error) {
	// Clone the underlying tensor
	clonedData := make([]float32, len(gt.Tensor.Data))
	copy(clonedData, gt.Tensor.Data)

	clonedTensor, err := tensor.NewTensor(gt.Tensor.Shape, clonedData)
	if err != nil {
		return nil, fmt.Errorf("failed to clone tensor: %w", err)
	}

	clonedGT := &GradientTensor{
		Tensor:       clonedTensor,
		RequiresGrad: gt.RequiresGrad,
		IsLeaf:       gt.IsLeaf,
	}

	// Clone gradient if it exists
	if gt.Gradient != nil {
		gradData := make([]float32, len(gt.Gradient.Data))
		copy(gradData, gt.Gradient.Data)

		clonedGrad, err := tensor.NewTensor(gt.Gradient.Shape, gradData)
		if err != nil {
			return nil, fmt.Errorf("failed to clone gradient: %w", err)
		}
		clonedGT.Gradient = clonedGrad
	}

	return clonedGT, nil
}

// SaveForBackward saves tensors that will be needed during the backward pass
func SaveForBackward(tensors ...*tensor.Tensor) []*tensor.Tensor {
	saved := make([]*tensor.Tensor, len(tensors))
	for i, t := range tensors {
		// Create a copy to avoid issues with in-place operations
		data := make([]float32, len(t.Data))
		if err := t.RetrieveCPU(); err == nil {
			copy(data, t.Data)
		}

		savedTensor, err := tensor.NewTensor(t.Shape, data)
		if err != nil {
			// If we can't create a copy, save the original (risky but better than failing)
			saved[i] = t
		} else {
			saved[i] = savedTensor
		}
	}
	return saved
}

// GradientScaleAndClip combines gradient scaling and clipping in one operation
func GradientScaleAndClip(gradTensors []*GradientTensor, scale float32, maxNorm float32) error {
	// First scale all gradients
	for _, gt := range gradTensors {
		if gt.Gradient != nil {
			scaled, err := ScalarMul(gt.Gradient, scale)
			if err != nil {
				return fmt.Errorf("failed to scale gradient: %w", err)
			}
			gt.Gradient.ReleaseGPU()
			gt.Gradient = scaled
		}
	}

	// Then clip by norm
	return ClipGradientNorm(gradTensors, maxNorm)
}

// GetGradientStats returns statistics about gradients for debugging
func GetGradientStats(gradTensors []*GradientTensor) (map[string]float32, error) {
	stats := make(map[string]float32)

	if len(gradTensors) == 0 {
		return stats, nil
	}

	// Calculate gradient norm
	norm, err := CalculateGradientNorm(gradTensors)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate gradient norm: %w", err)
	}
	stats["norm"] = norm

	// Calculate min, max, mean of all gradients
	var allGrads []float32
	for _, gt := range gradTensors {
		if gt.Gradient != nil {
			if err := gt.Gradient.RetrieveCPU(); err != nil {
				continue
			}
			allGrads = append(allGrads, gt.Gradient.Data...)
		}
	}

	if len(allGrads) > 0 {
		min, max := allGrads[0], allGrads[0]
		sum := float32(0.0)

		for _, val := range allGrads {
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
			sum += val
		}

		stats["min"] = min
		stats["max"] = max
		stats["mean"] = sum / float32(len(allGrads))
		stats["count"] = float32(len(allGrads))
	}

	return stats, nil
}
