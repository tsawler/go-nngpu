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

// ActivationType represents different activation function types
type ActivationType int

const (
	ReLU ActivationType = iota
	Sigmoid
	Tanh
	Softmax
	LeakyReLU
	ELU
	Swish
	GELU
)

// String returns string representation of activation type
func (at ActivationType) String() string {
	switch at {
	case ReLU:
		return "ReLU"
	case Sigmoid:
		return "Sigmoid"
	case Tanh:
		return "Tanh"
	case Softmax:
		return "Softmax"
	case LeakyReLU:
		return "LeakyReLU"
	case ELU:
		return "ELU"
	case Swish:
		return "Swish"
	case GELU:
		return "GELU"
	default:
		return "Unknown"
	}
}

// ActivationForward applies an activation function to the input tensor
func ActivationForward(input *tensor.Tensor, activationType ActivationType, alpha float32) (*tensor.Tensor, error) {
	if len(input.Shape) == 0 {
		return nil, fmt.Errorf("cannot apply activation to empty tensor")
	}

	// Create result tensor with same shape
	resultSize := 1
	for _, dim := range input.Shape {
		resultSize *= dim
	}
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor(input.Shape, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	// Ensure input tensor is on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input tensor to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	var retCode C.int

	// Route to appropriate activation function based on type
	switch activationType {
	case ReLU:
		retCode = C.perform_activation_relu_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case Sigmoid:
		retCode = C.perform_activation_sigmoid_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case Tanh:
		retCode = C.perform_activation_tanh_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case LeakyReLU:
		retCode = C.perform_activation_leaky_relu_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.float(alpha),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case ELU:
		retCode = C.perform_activation_elu_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.float(alpha),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case Swish:
		retCode = C.perform_activation_swish_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case GELU:
		retCode = C.perform_activation_gelu_forward(
			C.GPUPtr(input.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(input.DevicePtr()),
			&cErr,
		)
	case Softmax:
		// Softmax requires special handling for dimensions
		if len(input.Shape) == 1 {
			// 1D softmax
			retCode = C.perform_activation_softmax_1d_forward(
				C.GPUPtr(input.GPUPtr()), C.long(input.Shape[0]),
				C.GPUPtr(resultTensor.GPUPtr()),
				C.DevicePtr(input.DevicePtr()),
				&cErr,
			)
		} else if len(input.Shape) == 2 {
			// 2D softmax (apply along last dimension)
			retCode = C.perform_activation_softmax_2d_forward(
				C.GPUPtr(input.GPUPtr()), C.long(input.Shape[0]), C.long(input.Shape[1]),
				C.GPUPtr(resultTensor.GPUPtr()),
				C.DevicePtr(input.DevicePtr()),
				&cErr,
			)
		} else {
			return nil, fmt.Errorf("softmax currently supports only 1D and 2D tensors")
		}
	default:
		return nil, fmt.Errorf("unsupported activation type: %v", activationType)
	}

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU activation forward failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// ActivationBackward computes the gradient of an activation function
func ActivationBackward(gradOutput, activationOutput *tensor.Tensor, activationType ActivationType, alpha float32) (*tensor.Tensor, error) {
	if len(gradOutput.Shape) == 0 || len(activationOutput.Shape) == 0 {
		return nil, fmt.Errorf("cannot compute activation gradient for empty tensors")
	}

	// Verify shapes match
	if len(gradOutput.Shape) != len(activationOutput.Shape) {
		return nil, fmt.Errorf("gradient and activation output must have same dimensions")
	}
	for i, dim := range gradOutput.Shape {
		if dim != activationOutput.Shape[i] {
			return nil, fmt.Errorf("gradient and activation output shape mismatch at dimension %d", i)
		}
	}

	// Create result tensor with same shape
	resultSize := 1
	for _, dim := range gradOutput.Shape {
		resultSize *= dim
	}
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor(gradOutput.Shape, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := gradOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move gradient output tensor to GPU: %w", err)
	}
	if err := activationOutput.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move activation output tensor to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	var retCode C.int

	// Route to appropriate activation gradient function based on type
	switch activationType {
	case ReLU:
		retCode = C.perform_activation_relu_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case Sigmoid:
		retCode = C.perform_activation_sigmoid_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case Tanh:
		retCode = C.perform_activation_tanh_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case LeakyReLU:
		retCode = C.perform_activation_leaky_relu_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.float(alpha),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case ELU:
		retCode = C.perform_activation_elu_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.float(alpha),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case Swish:
		retCode = C.perform_activation_swish_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case GELU:
		retCode = C.perform_activation_gelu_backward(
			C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(resultSize),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(gradOutput.DevicePtr()),
			&cErr,
		)
	case Softmax:
		// Softmax gradient requires special handling
		if len(gradOutput.Shape) == 1 {
			retCode = C.perform_activation_softmax_1d_backward(
				C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()), C.long(gradOutput.Shape[0]),
				C.GPUPtr(resultTensor.GPUPtr()),
				C.DevicePtr(gradOutput.DevicePtr()),
				&cErr,
			)
		} else if len(gradOutput.Shape) == 2 {
			retCode = C.perform_activation_softmax_2d_backward(
				C.GPUPtr(gradOutput.GPUPtr()), C.GPUPtr(activationOutput.GPUPtr()),
				C.long(gradOutput.Shape[0]), C.long(gradOutput.Shape[1]),
				C.GPUPtr(resultTensor.GPUPtr()),
				C.DevicePtr(gradOutput.DevicePtr()),
				&cErr,
			)
		} else {
			return nil, fmt.Errorf("softmax gradient currently supports only 1D and 2D tensors")
		}
	default:
		return nil, fmt.Errorf("unsupported activation type for gradient: %v", activationType)
	}

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU activation backward failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// Convenience functions for common activations

// ReLUForward applies ReLU activation function
func ReLUForward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationForward(input, ReLU, 0.0)
}

// ReLUBackward computes ReLU gradient
func ReLUBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, ReLU, 0.0)
}

// SigmoidForward applies Sigmoid activation function
func SigmoidForward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationForward(input, Sigmoid, 0.0)
}

// SigmoidBackward computes Sigmoid gradient
func SigmoidBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, Sigmoid, 0.0)
}

// TanhForward applies Tanh activation function
func TanhForward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationForward(input, Tanh, 0.0)
}

// TanhBackward computes Tanh gradient
func TanhBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, Tanh, 0.0)
}

// SoftmaxForward applies Softmax activation function
func SoftmaxForward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationForward(input, Softmax, 0.0)
}

// SoftmaxBackward computes Softmax gradient
func SoftmaxBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, Softmax, 0.0)
}

// LeakyReLUForward applies Leaky ReLU activation function
func LeakyReLUForward(input *tensor.Tensor, alpha float32) (*tensor.Tensor, error) {
	return ActivationForward(input, LeakyReLU, alpha)
}

// LeakyReLUBackward computes Leaky ReLU gradient
func LeakyReLUBackward(gradOutput, activationOutput *tensor.Tensor, alpha float32) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, LeakyReLU, alpha)
}

// ELUForward applies ELU (Exponential Linear Unit) activation function
func ELUForward(input *tensor.Tensor, alpha float32) (*tensor.Tensor, error) {
	return ActivationForward(input, ELU, alpha)
}

// ELUBackward computes ELU gradient
func ELUBackward(gradOutput, activationOutput *tensor.Tensor, alpha float32) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, ELU, alpha)
}

// SwishForward applies Swish activation function (x * sigmoid(x))
func SwishForward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationForward(input, Swish, 0.0)
}

// SwishBackward computes Swish gradient
func SwishBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, Swish, 0.0)
}

// GELUForward applies GELU (Gaussian Error Linear Unit) activation function
func GELUForward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationForward(input, GELU, 0.0)
}

// GELUBackward computes GELU gradient
func GELUBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return ActivationBackward(gradOutput, activationOutput, GELU, 0.0)
}

// Batch processing functions for multiple tensors

// BatchActivationForward applies activation function to multiple tensors
func BatchActivationForward(inputs []*tensor.Tensor, activationType ActivationType, alpha float32) ([]*tensor.Tensor, error) {
	results := make([]*tensor.Tensor, len(inputs))
	for i, input := range inputs {
		result, err := ActivationForward(input, activationType, alpha)
		if err != nil {
			// Clean up any results that were already computed
			for j := 0; j < i; j++ {
				results[j].ReleaseGPU()
			}
			return nil, fmt.Errorf("failed to apply activation to tensor %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// BatchActivationBackward computes gradients for multiple tensors
func BatchActivationBackward(gradOutputs, activationOutputs []*tensor.Tensor, activationType ActivationType, alpha float32) ([]*tensor.Tensor, error) {
	if len(gradOutputs) != len(activationOutputs) {
		return nil, fmt.Errorf("number of gradient outputs (%d) must match activation outputs (%d)", len(gradOutputs), len(activationOutputs))
	}

	results := make([]*tensor.Tensor, len(gradOutputs))
	for i := range gradOutputs {
		result, err := ActivationBackward(gradOutputs[i], activationOutputs[i], activationType, alpha)
		if err != nil {
			// Clean up any results that were already computed
			for j := 0; j < i; j++ {
				results[j].ReleaseGPU()
			}
			return nil, fmt.Errorf("failed to compute activation gradient for tensor %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}
