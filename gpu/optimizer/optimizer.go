package optimizer

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

	_ "github.com/tsawler/gometal/internal/cgo"
	"github.com/tsawler/gometal/tensor"
)

// Optimizer represents a generic optimizer interface
type Optimizer interface {
	Step(params []*tensor.Tensor, grads []*tensor.Tensor) error
	ZeroGrad(grads []*tensor.Tensor) error
	GetLearningRate() float32
	SetLearningRate(lr float32)
	GetStepCount() int64
	ReleaseGPU()
}

// OptimizerConfig holds common configuration for all optimizers
type OptimizerConfig struct {
	LearningRate float32
	WeightDecay  float32
}

// SGDConfig holds configuration specific to SGD optimizer
type SGDConfig struct {
	OptimizerConfig
	Momentum float32
}

// SGDOptimizer implements Stochastic Gradient Descent with momentum
type SGDOptimizer struct {
	config          SGDConfig
	momentumBuffers []*tensor.Tensor
	stepCount       int64
}

// NewSGD creates a new SGD optimizer
func NewSGD(config SGDConfig) *SGDOptimizer {
	return &SGDOptimizer{
		config:          config,
		momentumBuffers: nil,
		stepCount:       0,
	}
}

// Step performs one optimization step
func (opt *SGDOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error {
	if len(params) != len(grads) {
		return fmt.Errorf("params and grads must have the same length")
	}

	// Initialize momentum buffers if needed
	if opt.momentumBuffers == nil && opt.config.Momentum != 0.0 {
		opt.momentumBuffers = make([]*tensor.Tensor, len(params))
		for i, param := range params {
			momentumData := make([]float32, len(param.Data))
			momentumTensor, err := tensor.NewTensor(param.Shape, momentumData)
			if err != nil {
				return fmt.Errorf("failed to create momentum buffer %d: %w", i, err)
			}
			if err := momentumTensor.EnsureGPU(); err != nil {
				return fmt.Errorf("failed to move momentum buffer %d to GPU: %w", i, err)
			}
			opt.momentumBuffers[i] = momentumTensor
		}
	}

	opt.stepCount++

	for i, param := range params {
		grad := grads[i]

		if err := param.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move param %d to GPU: %w", i, err)
		}
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		var momentumPtr C.GPUPtr
		if opt.momentumBuffers != nil {
			momentumPtr = C.GPUPtr(opt.momentumBuffers[i].GPUPtr())
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_sgd_step(
			C.GPUPtr(param.GPUPtr()),
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(param.Data)),
			C.float(opt.config.LearningRate),
			C.float(opt.config.Momentum),
			momentumPtr,
			C.float(opt.config.WeightDecay),
			C.DevicePtr(param.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("SGD step failed for param %d (code %d): %s", i, retCode, errMsg)
		}
	}

	return nil
}

// ZeroGrad zeros all gradients
func (opt *SGDOptimizer) ZeroGrad(grads []*tensor.Tensor) error {
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_tensor_fill(
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(grad.Data)),
			C.float(0.0),
			C.DevicePtr(grad.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("failed to zero grad %d (code %d): %s", i, retCode, errMsg)
		}
	}
	return nil
}

// GetLearningRate returns the current learning rate
func (opt *SGDOptimizer) GetLearningRate() float32 {
	return opt.config.LearningRate
}

// SetLearningRate sets the learning rate
func (opt *SGDOptimizer) SetLearningRate(lr float32) {
	opt.config.LearningRate = lr
}

// GetStepCount returns the current step count
func (opt *SGDOptimizer) GetStepCount() int64 {
	return opt.stepCount
}

// ReleaseGPU releases GPU resources
func (opt *SGDOptimizer) ReleaseGPU() {
	if opt.momentumBuffers != nil {
		for _, buffer := range opt.momentumBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.momentumBuffers = nil
	}
}

// AdamConfig holds configuration specific to Adam optimizer
type AdamConfig struct {
	OptimizerConfig
	Beta1   float32
	Beta2   float32
	Epsilon float32
}

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	config    AdamConfig
	mBuffers  []*tensor.Tensor // First moment buffers
	vBuffers  []*tensor.Tensor // Second moment buffers
	stepCount int64
}

// NewAdam creates a new Adam optimizer
func NewAdam(config AdamConfig) *AdamOptimizer {
	return &AdamOptimizer{
		config:    config,
		mBuffers:  nil,
		vBuffers:  nil,
		stepCount: 0,
	}
}

// Step performs one optimization step
func (opt *AdamOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error {
	if len(params) != len(grads) {
		return fmt.Errorf("params and grads must have the same length")
	}

	// Initialize moment buffers if needed
	if opt.mBuffers == nil {
		opt.mBuffers = make([]*tensor.Tensor, len(params))
		opt.vBuffers = make([]*tensor.Tensor, len(params))

		for i, param := range params {
			// Create first moment buffer
			mData := make([]float32, len(param.Data))
			mTensor, err := tensor.NewTensor(param.Shape, mData)
			if err != nil {
				return fmt.Errorf("failed to create first moment buffer %d: %w", i, err)
			}
			if err := mTensor.EnsureGPU(); err != nil {
				return fmt.Errorf("failed to move first moment buffer %d to GPU: %w", i, err)
			}
			opt.mBuffers[i] = mTensor

			// Create second moment buffer
			vData := make([]float32, len(param.Data))
			vTensor, err := tensor.NewTensor(param.Shape, vData)
			if err != nil {
				return fmt.Errorf("failed to create second moment buffer %d: %w", i, err)
			}
			if err := vTensor.EnsureGPU(); err != nil {
				return fmt.Errorf("failed to move second moment buffer %d to GPU: %w", i, err)
			}
			opt.vBuffers[i] = vTensor
		}
	}

	opt.stepCount++

	for i, param := range params {
		grad := grads[i]

		if err := param.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move param %d to GPU: %w", i, err)
		}
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_adam_step(
			C.GPUPtr(param.GPUPtr()),
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(param.Data)),
			C.float(opt.config.LearningRate),
			C.float(opt.config.Beta1),
			C.float(opt.config.Beta2),
			C.float(opt.config.Epsilon),
			C.float(opt.config.WeightDecay),
			C.GPUPtr(opt.mBuffers[i].GPUPtr()),
			C.GPUPtr(opt.vBuffers[i].GPUPtr()),
			C.long(opt.stepCount),
			C.DevicePtr(param.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("the Adam step failed for param %d (code %d): %s", i, retCode, errMsg)
		}
	}

	return nil
}

// ZeroGrad zeros all gradients
func (opt *AdamOptimizer) ZeroGrad(grads []*tensor.Tensor) error {
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_tensor_fill(
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(grad.Data)),
			C.float(0.0),
			C.DevicePtr(grad.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("failed to zero grad %d (code %d): %s", i, retCode, errMsg)
		}
	}
	return nil
}

// GetLearningRate returns the current learning rate
func (opt *AdamOptimizer) GetLearningRate() float32 {
	return opt.config.LearningRate
}

// SetLearningRate sets the learning rate
func (opt *AdamOptimizer) SetLearningRate(lr float32) {
	opt.config.LearningRate = lr
}

// GetStepCount returns the current step count
func (opt *AdamOptimizer) GetStepCount() int64 {
	return opt.stepCount
}

// ReleaseGPU releases GPU resources
func (opt *AdamOptimizer) ReleaseGPU() {
	if opt.mBuffers != nil {
		for _, buffer := range opt.mBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.mBuffers = nil
	}
	if opt.vBuffers != nil {
		for _, buffer := range opt.vBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.vBuffers = nil
	}
}

// RMSpropConfig holds configuration specific to RMSprop optimizer
type RMSpropConfig struct {
	OptimizerConfig
	Alpha    float32 // Smoothing constant
	Epsilon  float32
	Momentum float32
}

// RMSpropOptimizer implements the RMSprop optimization algorithm
type RMSpropOptimizer struct {
	config             RMSpropConfig
	squaredGradBuffers []*tensor.Tensor
	momentumBuffers    []*tensor.Tensor
	stepCount          int64
}

// NewRMSprop creates a new RMSprop optimizer
func NewRMSprop(config RMSpropConfig) *RMSpropOptimizer {
	return &RMSpropOptimizer{
		config:             config,
		squaredGradBuffers: nil,
		momentumBuffers:    nil,
		stepCount:          0,
	}
}

// Step performs one optimization step
func (opt *RMSpropOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error {
	if len(params) != len(grads) {
		return fmt.Errorf("params and grads must have the same length")
	}

	// Initialize buffers if needed
	if opt.squaredGradBuffers == nil {
		opt.squaredGradBuffers = make([]*tensor.Tensor, len(params))
		if opt.config.Momentum != 0.0 {
			opt.momentumBuffers = make([]*tensor.Tensor, len(params))
		}

		for i, param := range params {
			// Create squared gradient buffer
			sqGradData := make([]float32, len(param.Data))
			sqGradTensor, err := tensor.NewTensor(param.Shape, sqGradData)
			if err != nil {
				return fmt.Errorf("failed to create squared gradient buffer %d: %w", i, err)
			}
			if err := sqGradTensor.EnsureGPU(); err != nil {
				return fmt.Errorf("failed to move squared gradient buffer %d to GPU: %w", i, err)
			}
			opt.squaredGradBuffers[i] = sqGradTensor

			// Create momentum buffer if needed
			if opt.config.Momentum != 0.0 {
				momentumData := make([]float32, len(param.Data))
				momentumTensor, err := tensor.NewTensor(param.Shape, momentumData)
				if err != nil {
					return fmt.Errorf("failed to create momentum buffer %d: %w", i, err)
				}
				if err := momentumTensor.EnsureGPU(); err != nil {
					return fmt.Errorf("failed to move momentum buffer %d to GPU: %w", i, err)
				}
				opt.momentumBuffers[i] = momentumTensor
			}
		}
	}

	opt.stepCount++

	for i, param := range params {
		grad := grads[i]

		if err := param.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move param %d to GPU: %w", i, err)
		}
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		var momentumPtr C.GPUPtr
		if opt.momentumBuffers != nil {
			momentumPtr = C.GPUPtr(opt.momentumBuffers[i].GPUPtr())
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_rmsprop_step(
			C.GPUPtr(param.GPUPtr()),
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(param.Data)),
			C.float(opt.config.LearningRate),
			C.float(opt.config.Alpha),
			C.float(opt.config.Epsilon),
			C.float(opt.config.Momentum),
			C.float(opt.config.WeightDecay),
			C.GPUPtr(opt.squaredGradBuffers[i].GPUPtr()),
			momentumPtr,
			C.DevicePtr(param.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("RMSprop step failed for param %d (code %d): %s", i, retCode, errMsg)
		}
	}

	return nil
}

// ZeroGrad zeros all gradients
func (opt *RMSpropOptimizer) ZeroGrad(grads []*tensor.Tensor) error {
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_tensor_fill(
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(grad.Data)),
			C.float(0.0),
			C.DevicePtr(grad.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("failed to zero grad %d (code %d): %s", i, retCode, errMsg)
		}
	}
	return nil
}

// GetLearningRate returns the current learning rate
func (opt *RMSpropOptimizer) GetLearningRate() float32 {
	return opt.config.LearningRate
}

// SetLearningRate sets the learning rate
func (opt *RMSpropOptimizer) SetLearningRate(lr float32) {
	opt.config.LearningRate = lr
}

// GetStepCount returns the current step count
func (opt *RMSpropOptimizer) GetStepCount() int64 {
	return opt.stepCount
}

// ReleaseGPU releases GPU resources
func (opt *RMSpropOptimizer) ReleaseGPU() {
	if opt.squaredGradBuffers != nil {
		for _, buffer := range opt.squaredGradBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.squaredGradBuffers = nil
	}
	if opt.momentumBuffers != nil {
		for _, buffer := range opt.momentumBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.momentumBuffers = nil
	}
}

// AdamWConfig holds configuration specific to AdamW optimizer
type AdamWConfig struct {
	OptimizerConfig
	Beta1   float32
	Beta2   float32
	Epsilon float32
}

// AdamWOptimizer implements the AdamW optimization algorithm (Adam with decoupled weight decay)
type AdamWOptimizer struct {
	config    AdamWConfig
	mBuffers  []*tensor.Tensor // First moment buffers
	vBuffers  []*tensor.Tensor // Second moment buffers
	stepCount int64
}

// NewAdamW creates a new AdamW optimizer
func NewAdamW(config AdamWConfig) *AdamWOptimizer {
	return &AdamWOptimizer{
		config:    config,
		mBuffers:  nil,
		vBuffers:  nil,
		stepCount: 0,
	}
}

// Step performs one optimization step
func (opt *AdamWOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error {
	if len(params) != len(grads) {
		return fmt.Errorf("params and grads must have the same length")
	}

	// Initialize moment buffers if needed
	if opt.mBuffers == nil {
		opt.mBuffers = make([]*tensor.Tensor, len(params))
		opt.vBuffers = make([]*tensor.Tensor, len(params))

		for i, param := range params {
			// Create first moment buffer
			mData := make([]float32, len(param.Data))
			mTensor, err := tensor.NewTensor(param.Shape, mData)
			if err != nil {
				return fmt.Errorf("failed to create first moment buffer %d: %w", i, err)
			}
			if err := mTensor.EnsureGPU(); err != nil {
				return fmt.Errorf("failed to move first moment buffer %d to GPU: %w", i, err)
			}
			opt.mBuffers[i] = mTensor

			// Create second moment buffer
			vData := make([]float32, len(param.Data))
			vTensor, err := tensor.NewTensor(param.Shape, vData)
			if err != nil {
				return fmt.Errorf("failed to create second moment buffer %d: %w", i, err)
			}
			if err := vTensor.EnsureGPU(); err != nil {
				return fmt.Errorf("failed to move second moment buffer %d to GPU: %w", i, err)
			}
			opt.vBuffers[i] = vTensor
		}
	}

	opt.stepCount++

	for i, param := range params {
		grad := grads[i]

		if err := param.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move param %d to GPU: %w", i, err)
		}
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_adamw_step(
			C.GPUPtr(param.GPUPtr()),
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(param.Data)),
			C.float(opt.config.LearningRate),
			C.float(opt.config.Beta1),
			C.float(opt.config.Beta2),
			C.float(opt.config.Epsilon),
			C.float(opt.config.WeightDecay),
			C.GPUPtr(opt.mBuffers[i].GPUPtr()),
			C.GPUPtr(opt.vBuffers[i].GPUPtr()),
			C.long(opt.stepCount),
			C.DevicePtr(param.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("AdamW step failed for param %d (code %d): %s", i, retCode, errMsg)
		}
	}

	return nil
}

// ZeroGrad zeros all gradients
func (opt *AdamWOptimizer) ZeroGrad(grads []*tensor.Tensor) error {
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_tensor_fill(
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(grad.Data)),
			C.float(0.0),
			C.DevicePtr(grad.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("failed to zero grad %d (code %d): %s", i, retCode, errMsg)
		}
	}
	return nil
}

// GetLearningRate returns the current learning rate
func (opt *AdamWOptimizer) GetLearningRate() float32 {
	return opt.config.LearningRate
}

// SetLearningRate sets the learning rate
func (opt *AdamWOptimizer) SetLearningRate(lr float32) {
	opt.config.LearningRate = lr
}

// GetStepCount returns the current step count
func (opt *AdamWOptimizer) GetStepCount() int64 {
	return opt.stepCount
}

// ReleaseGPU releases GPU resources
func (opt *AdamWOptimizer) ReleaseGPU() {
	if opt.mBuffers != nil {
		for _, buffer := range opt.mBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.mBuffers = nil
	}
	if opt.vBuffers != nil {
		for _, buffer := range opt.vBuffers {
			if buffer != nil {
				buffer.ReleaseGPU()
			}
		}
		opt.vBuffers = nil
	}
}

// Utility functions for gradient clipping

// ClipGradsByNorm clips gradients by global norm
func ClipGradsByNorm(grads []*tensor.Tensor, maxNorm float32) (float32, error) {
	if len(grads) == 0 {
		return 0.0, nil
	}

	// Ensure all gradients are on GPU
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return 0.0, fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}
	}

	// Create C arrays for GPU pointers and sizes
	gradPtrs := make([]C.GPUPtr, len(grads))
	sizes := make([]C.long, len(grads))

	for i, grad := range grads {
		gradPtrs[i] = C.GPUPtr(grad.GPUPtr())
		sizes[i] = C.long(len(grad.Data))
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var actualNorm C.float
	var cErr C.CError
	retCode := C.perform_global_gradient_clip(
		(*C.GPUPtr)(unsafe.Pointer(&gradPtrs[0])),
		(*C.long)(unsafe.Pointer(&sizes[0])),
		C.long(len(grads)),
		C.float(maxNorm),
		&actualNorm,
		C.DevicePtr(grads[0].DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return 0.0, fmt.Errorf("gradient clipping failed (code %d): %s", retCode, errMsg)
	}

	return float32(actualNorm), nil
}

// ClipGradsByValue clips gradients by value
func ClipGradsByValue(grads []*tensor.Tensor, minValue, maxValue float32) error {
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_gradient_clip_by_value(
			C.GPUPtr(grad.GPUPtr()),
			C.long(len(grad.Data)),
			C.float(minValue),
			C.float(maxValue),
			C.DevicePtr(grad.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("gradient value clipping failed for grad %d (code %d): %s", i, retCode, errMsg)
		}
	}
	return nil
}

// ComputeGradNorm computes the global gradient norm
func ComputeGradNorm(grads []*tensor.Tensor) (float32, error) {
	if len(grads) == 0 {
		return 0.0, nil
	}

	// Ensure all gradients are on GPU
	for i, grad := range grads {
		if err := grad.EnsureGPU(); err != nil {
			return 0.0, fmt.Errorf("failed to move grad %d to GPU: %w", i, err)
		}
	}

	// Create C arrays for GPU pointers and sizes
	gradPtrs := make([]C.GPUPtr, len(grads))
	sizes := make([]C.long, len(grads))

	for i, grad := range grads {
		gradPtrs[i] = C.GPUPtr(grad.GPUPtr())
		sizes[i] = C.long(len(grad.Data))
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var globalNorm C.float
	var cErr C.CError
	retCode := C.perform_global_gradient_norm(
		(*C.GPUPtr)(unsafe.Pointer(&gradPtrs[0])),
		(*C.long)(unsafe.Pointer(&sizes[0])),
		C.long(len(grads)),
		&globalNorm,
		C.DevicePtr(grads[0].DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return 0.0, fmt.Errorf("gradient norm computation failed (code %d): %s", retCode, errMsg)
	}

	return float32(globalNorm), nil
}
