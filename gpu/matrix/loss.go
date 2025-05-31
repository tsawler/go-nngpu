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

// LossType represents different loss function types
type LossType int

const (
	MSE LossType = iota
	BinaryCrossEntropy
	CategoricalCrossEntropy
	SparseCategoricalCrossEntropy
	Huber
	MAE
	Hinge
)

// String returns string representation of loss type
func (lt LossType) String() string {
	switch lt {
	case MSE:
		return "MSE"
	case BinaryCrossEntropy:
		return "BinaryCrossEntropy"
	case CategoricalCrossEntropy:
		return "CategoricalCrossEntropy"
	case SparseCategoricalCrossEntropy:
		return "SparseCategoricalCrossEntropy"
	case Huber:
		return "Huber"
	case MAE:
		return "MAE"
	case Hinge:
		return "Hinge"
	default:
		return "Unknown"
	}
}

// LossResult contains the computed loss value and gradients
type LossResult struct {
	Loss      float32
	Gradients *tensor.Tensor
}

// ReleaseGPU releases GPU resources for the loss result
func (lr *LossResult) ReleaseGPU() {
	if lr.Gradients != nil {
		lr.Gradients.ReleaseGPU()
	}
}

// LossForward computes the forward pass of a loss function
func LossForward(predictions, targets *tensor.Tensor, lossType LossType, params ...float32) (float32, error) {
	if len(predictions.Shape) == 0 || len(targets.Shape) == 0 {
		return 0, fmt.Errorf("cannot compute loss for empty tensors")
	}

	// Verify shapes match for most loss functions
	if lossType != SparseCategoricalCrossEntropy {
		if len(predictions.Shape) != len(targets.Shape) {
			return 0, fmt.Errorf("predictions and targets must have same dimensions")
		}
		for i, dim := range predictions.Shape {
			if dim != targets.Shape[i] {
				return 0, fmt.Errorf("predictions and targets shape mismatch at dimension %d", i)
			}
		}
	}

	// Ensure tensors are on GPU
	if err := predictions.EnsureGPU(); err != nil {
		return 0, fmt.Errorf("failed to move predictions tensor to GPU: %w", err)
	}
	if err := targets.EnsureGPU(); err != nil {
		return 0, fmt.Errorf("failed to move targets tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	var retCode C.int
	var loss C.float

	switch lossType {
	case MSE:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_mse_forward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			&loss,
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case BinaryCrossEntropy:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_binary_crossentropy_forward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			&loss,
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case CategoricalCrossEntropy:
		if len(predictions.Shape) != 2 {
			return 0, fmt.Errorf("categorical cross-entropy requires 2D tensors (batch_size, num_classes)")
		}
		batchSize := predictions.Shape[0]
		numClasses := predictions.Shape[1]
		retCode = C.perform_loss_categorical_crossentropy_forward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()),
			C.long(batchSize), C.long(numClasses),
			&loss,
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case Huber:
		if len(params) == 0 {
			return 0, fmt.Errorf("Huber loss requires delta parameter")
		}
		delta := params[0]
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_huber_forward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			C.float(delta),
			&loss,
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case MAE:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_mae_forward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			&loss,
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case Hinge:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_hinge_forward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			&loss,
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	default:
		return 0, fmt.Errorf("unsupported loss type: %v", lossType)
	}

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return 0, fmt.Errorf("GPU loss forward failed (code %d): %s", retCode, errMsg)
	}

	return float32(loss), nil
}

// LossBackward computes the backward pass (gradients) of a loss function
func LossBackward(predictions, targets *tensor.Tensor, lossType LossType, params ...float32) (*tensor.Tensor, error) {
	if len(predictions.Shape) == 0 || len(targets.Shape) == 0 {
		return nil, fmt.Errorf("cannot compute loss gradients for empty tensors")
	}

	// Verify shapes match for most loss functions
	if lossType != SparseCategoricalCrossEntropy {
		if len(predictions.Shape) != len(targets.Shape) {
			return nil, fmt.Errorf("predictions and targets must have same dimensions")
		}
		for i, dim := range predictions.Shape {
			if dim != targets.Shape[i] {
				return nil, fmt.Errorf("predictions and targets shape mismatch at dimension %d", i)
			}
		}
	}

	// Create result tensor with same shape as predictions
	resultSize := 1
	for _, dim := range predictions.Shape {
		resultSize *= dim
	}
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor(predictions.Shape, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := predictions.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move predictions tensor to GPU: %w", err)
	}
	if err := targets.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move targets tensor to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	var retCode C.int

	switch lossType {
	case MSE:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_mse_backward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case BinaryCrossEntropy:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_binary_crossentropy_backward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case CategoricalCrossEntropy:
		if len(predictions.Shape) != 2 {
			return nil, fmt.Errorf("categorical cross-entropy requires 2D tensors (batch_size, num_classes)")
		}
		batchSize := predictions.Shape[0]
		numClasses := predictions.Shape[1]
		retCode = C.perform_loss_categorical_crossentropy_backward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()),
			C.long(batchSize), C.long(numClasses),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case Huber:
		if len(params) == 0 {
			return nil, fmt.Errorf("Huber loss requires delta parameter")
		}
		delta := params[0]
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_huber_backward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			C.float(delta),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case MAE:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_mae_backward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	case Hinge:
		size := 1
		for _, dim := range predictions.Shape {
			size *= dim
		}
		retCode = C.perform_loss_hinge_backward(
			C.GPUPtr(predictions.GPUPtr()), C.GPUPtr(targets.GPUPtr()), C.long(size),
			C.GPUPtr(resultTensor.GPUPtr()),
			C.DevicePtr(predictions.DevicePtr()),
			&cErr,
		)

	default:
		return nil, fmt.Errorf("unsupported loss type: %v", lossType)
	}

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU loss backward failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// LossForwardBackward computes both forward and backward passes efficiently
func LossForwardBackward(predictions, targets *tensor.Tensor, lossType LossType, params ...float32) (*LossResult, error) {
	loss, err := LossForward(predictions, targets, lossType, params...)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}

	gradients, err := LossBackward(predictions, targets, lossType, params...)
	if err != nil {
		return nil, fmt.Errorf("backward pass failed: %w", err)
	}

	return &LossResult{
		Loss:      loss,
		Gradients: gradients,
	}, nil
}

// SparseCategoricalCrossEntropyForward computes sparse categorical cross-entropy with integer targets
func SparseCategoricalCrossEntropyForward(predictions *tensor.Tensor, targetIndices []int) (float32, error) {
	if len(predictions.Shape) != 2 {
		return 0, fmt.Errorf("sparse categorical cross-entropy requires 2D predictions tensor (batch_size, num_classes)")
	}

	batchSize := predictions.Shape[0]
	numClasses := predictions.Shape[1]

	if len(targetIndices) != batchSize {
		return 0, fmt.Errorf("number of target indices (%d) must match batch size (%d)", len(targetIndices), batchSize)
	}

	// Convert Go slice to C array
	cTargetIndices := make([]C.int, len(targetIndices))
	for i, idx := range targetIndices {
		if idx < 0 || idx >= numClasses {
			return 0, fmt.Errorf("target index %d out of bounds [0, %d)", idx, numClasses)
		}
		cTargetIndices[i] = C.int(idx)
	}

	// Ensure predictions tensor is on GPU
	if err := predictions.EnsureGPU(); err != nil {
		return 0, fmt.Errorf("failed to move predictions tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	var loss C.float

	retCode := C.perform_loss_sparse_categorical_crossentropy_forward(
		C.GPUPtr(predictions.GPUPtr()), (*C.int)(unsafe.Pointer(&cTargetIndices[0])),
		C.long(batchSize), C.long(numClasses),
		&loss,
		C.DevicePtr(predictions.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return 0, fmt.Errorf("GPU sparse categorical cross-entropy forward failed (code %d): %s", retCode, errMsg)
	}

	return float32(loss), nil
}

// SparseCategoricalCrossEntropyBackward computes gradients for sparse categorical cross-entropy
func SparseCategoricalCrossEntropyBackward(predictions *tensor.Tensor, targetIndices []int) (*tensor.Tensor, error) {
	if len(predictions.Shape) != 2 {
		return nil, fmt.Errorf("sparse categorical cross-entropy requires 2D predictions tensor (batch_size, num_classes)")
	}

	batchSize := predictions.Shape[0]
	numClasses := predictions.Shape[1]

	if len(targetIndices) != batchSize {
		return nil, fmt.Errorf("number of target indices (%d) must match batch size (%d)", len(targetIndices), batchSize)
	}

	// Convert Go slice to C array
	cTargetIndices := make([]C.int, len(targetIndices))
	for i, idx := range targetIndices {
		if idx < 0 || idx >= numClasses {
			return nil, fmt.Errorf("target index %d out of bounds [0, %d)", idx, numClasses)
		}
		cTargetIndices[i] = C.int(idx)
	}

	// Create result tensor with same shape as predictions
	resultSize := batchSize * numClasses
	resultData := make([]float32, resultSize)
	resultTensor, err := tensor.NewTensor(predictions.Shape, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := predictions.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move predictions tensor to GPU: %w", err)
	}
	if err := resultTensor.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move result tensor to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError

	retCode := C.perform_loss_sparse_categorical_crossentropy_backward(
		C.GPUPtr(predictions.GPUPtr()), (*C.int)(unsafe.Pointer(&cTargetIndices[0])),
		C.long(batchSize), C.long(numClasses),
		C.GPUPtr(resultTensor.GPUPtr()),
		C.DevicePtr(predictions.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sparse categorical cross-entropy backward failed (code %d): %s", retCode, errMsg)
	}

	return resultTensor, nil
}

// SparseCategoricalCrossEntropyForwardBackward computes both forward and backward passes for sparse categorical cross-entropy
func SparseCategoricalCrossEntropyForwardBackward(predictions *tensor.Tensor, targetIndices []int) (*LossResult, error) {
	loss, err := SparseCategoricalCrossEntropyForward(predictions, targetIndices)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}

	gradients, err := SparseCategoricalCrossEntropyBackward(predictions, targetIndices)
	if err != nil {
		return nil, fmt.Errorf("backward pass failed: %w", err)
	}

	return &LossResult{
		Loss:      loss,
		Gradients: gradients,
	}, nil
}

// Convenience functions for common loss functions

// MSELoss computes Mean Squared Error loss
func MSELoss(predictions, targets *tensor.Tensor) (float32, error) {
	return LossForward(predictions, targets, MSE)
}

// MSELossGradients computes MSE gradients
func MSELossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error) {
	return LossBackward(predictions, targets, MSE)
}

// BinaryCrossEntropyLoss computes Binary Cross-Entropy loss
func BinaryCrossEntropyLoss(predictions, targets *tensor.Tensor) (float32, error) {
	return LossForward(predictions, targets, BinaryCrossEntropy)
}

// BinaryCrossEntropyLossGradients computes Binary Cross-Entropy gradients
func BinaryCrossEntropyLossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error) {
	return LossBackward(predictions, targets, BinaryCrossEntropy)
}

// CategoricalCrossEntropyLoss computes Categorical Cross-Entropy loss
func CategoricalCrossEntropyLoss(predictions, targets *tensor.Tensor) (float32, error) {
	return LossForward(predictions, targets, CategoricalCrossEntropy)
}

// CategoricalCrossEntropyLossGradients computes Categorical Cross-Entropy gradients
func CategoricalCrossEntropyLossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error) {
	return LossBackward(predictions, targets, CategoricalCrossEntropy)
}

// HuberLoss computes Huber loss with specified delta
func HuberLoss(predictions, targets *tensor.Tensor, delta float32) (float32, error) {
	return LossForward(predictions, targets, Huber, delta)
}

// HuberLossGradients computes Huber loss gradients
func HuberLossGradients(predictions, targets *tensor.Tensor, delta float32) (*tensor.Tensor, error) {
	return LossBackward(predictions, targets, Huber, delta)
}

// MAELoss computes Mean Absolute Error loss
func MAELoss(predictions, targets *tensor.Tensor) (float32, error) {
	return LossForward(predictions, targets, MAE)
}

// MAELossGradients computes MAE gradients
func MAELossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error) {
	return LossBackward(predictions, targets, MAE)
}

// HingeLoss computes Hinge loss
func HingeLoss(predictions, targets *tensor.Tensor) (float32, error) {
	return LossForward(predictions, targets, Hinge)
}

// HingeLossGradients computes Hinge loss gradients
func HingeLossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error) {
	return LossBackward(predictions, targets, Hinge)
}