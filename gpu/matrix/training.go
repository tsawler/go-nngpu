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
	"time"

	"github.com/tsawler/go-nngpu/gpu/optimizer"
	_ "github.com/tsawler/go-nngpu/internal/cgo"
	"github.com/tsawler/go-nngpu/tensor"
)

// TrainingConfig contains configuration for memory-efficient training
type TrainingConfig struct {
	// Memory management
	GradientAccumulationSteps int   // Number of mini-batches to accumulate gradients
	MaxMemoryUsage            int64 // Maximum GPU memory usage in bytes (0 = no limit)
	EnableGradientCheckpoint  bool  // Enable gradient checkpointing to save memory
	EnableMixedPrecision      bool  // Enable mixed precision training

	// Training parameters
	LearningRate     float32
	WeightDecay      float32
	GradientClipping float32 // Max gradient norm (0 = no clipping)
	EpochCount       int
	BatchSize        int
	ValidationFreq   int // Validate every N batches (0 = no validation)

	// Optimizer settings
	OptimizerType OptimizerType
	Beta1         float32 // For Adam/AdamW
	Beta2         float32 // For Adam/AdamW
	Epsilon       float32 // For Adam/AdamW
	Momentum      float32 // For SGD with momentum

	// Scheduler settings
	SchedulerType     SchedulerType
	SchedulerGamma    float32 // Decay factor for step/exponential schedulers
	SchedulerStepSize int     // Step size for step scheduler
	WarmupSteps       int     // Number of warmup steps
	TotalSteps        int     // Total training steps (for cosine annealing)

	// Memory optimization
	ClearCacheFreq   int  // Clear GPU cache every N batches
	PrefetchBatches  int  // Number of batches to prefetch (0 = disable)
	AsyncDataLoading bool // Enable async data loading
}

// OptimizerType represents different optimizer types (matching your existing ones)
type OptimizerType int

const (
	OptimizerSGD OptimizerType = iota
	OptimizerAdam
	OptimizerAdamW
	OptimizerRMSprop
)

// SchedulerType represents different learning rate scheduler types
type SchedulerType int

const (
	NoScheduler SchedulerType = iota
	StepLR
	ExponentialLR
	CosineAnnealingLR
	PolynomialLR
	WarmupLR
)

// TrainingState tracks the current state of training
type TrainingState struct {
	Epoch          int
	Batch          int
	Step           int
	Loss           float32
	ValidationLoss float32
	LearningRate   float32
	GradientNorm   float32
	MemoryUsage    int64
	BatchTime      time.Duration

	// Accumulated gradients
	AccumulatedGrads  map[string]*tensor.Tensor
	AccumulationCount int

	// Optimizer state
	OptimizerInstance optimizer.Optimizer
	SchedulerInstance optimizer.LRScheduler

	// Memory management
	MemoryPool  *GPUMemoryPool
	TensorCache *TensorCache
}

// TrainingMetrics tracks training progress and performance
type TrainingMetrics struct {
	Losses           []float32
	ValidationLosses []float32
	LearningRates    []float32
	GradientNorms    []float32
	BatchTimes       []time.Duration
	MemoryUsages     []int64
	Epochs           []int

	// Statistics
	BestValidationLoss float32
	BestEpoch          int
	TotalTrainingTime  time.Duration
	AverageEpochTime   time.Duration
}

// Model interface for training
type TrainableModel interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*GradientTensor
	ZeroGrad()
	SetTraining(training bool)
	GetName() string
}

// ValidationCallback is called during validation
type ValidationCallback func(epoch int, batch int, validationLoss float32) error

// Trainer manages memory-efficient training
type Trainer struct {
	Config  *TrainingConfig
	State   *TrainingState
	Metrics *TrainingMetrics
	Model   TrainableModel

	// Callbacks
	OnEpochStart    func(epoch int) error
	OnEpochEnd      func(epoch int, metrics *TrainingMetrics) error
	OnBatchStart    func(epoch, batch int) error
	OnBatchEnd      func(epoch, batch int, loss float32) error
	OnValidation    ValidationCallback
	OnTrainingStart func() error
	OnTrainingEnd   func(metrics *TrainingMetrics) error
}

// NewTrainer creates a new memory-efficient trainer
func NewTrainer(config *TrainingConfig, model TrainableModel) (*Trainer, error) {
	if config == nil {
		return nil, fmt.Errorf("training config cannot be nil")
	}
	if model == nil {
		return nil, fmt.Errorf("model cannot be nil")
	}

	// Initialize memory pool
	memoryPool, err := NewGPUMemoryPool(config.MaxMemoryUsage)
	if err != nil {
		return nil, fmt.Errorf("failed to create GPU memory pool: %w", err)
	}

	// Initialize tensor cache
	tensorCache := NewTensorCache(1000) // Cache up to 1000 tensors

	// Initialize training state
	state := &TrainingState{
		AccumulatedGrads: make(map[string]*tensor.Tensor),
		MemoryPool:       memoryPool,
		TensorCache:      tensorCache,
	}

	// Initialize metrics
	metrics := &TrainingMetrics{
		BestValidationLoss: float32(1e9),
		BestEpoch:          -1,
	}

	trainer := &Trainer{
		Config:  config,
		State:   state,
		Metrics: metrics,
		Model:   model,
	}

	// Initialize optimizer
	err = trainer.initializeOptimizer()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize optimizer: %w", err)
	}

	// Initialize scheduler
	err = trainer.initializeScheduler()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize scheduler: %w", err)
	}

	return trainer, nil
}

// Train performs memory-efficient training
func (t *Trainer) Train(trainLoader, validLoader DataLoader) error {
	startTime := time.Now()

	// Call training start callback
	if t.OnTrainingStart != nil {
		if err := t.OnTrainingStart(); err != nil {
			return fmt.Errorf("training start callback failed: %w", err)
		}
	}

	// Enable gradient computation
	SetGradientMode(Grad)
	defer SetGradientMode(NoGrad)

	for epoch := 0; epoch < t.Config.EpochCount; epoch++ {
		t.State.Epoch = epoch
		epochStartTime := time.Now()

		// Call epoch start callback
		if t.OnEpochStart != nil {
			if err := t.OnEpochStart(epoch); err != nil {
				return fmt.Errorf("epoch start callback failed: %w", err)
			}
		}

		// Train one epoch
		err := t.trainEpoch(trainLoader, validLoader)
		if err != nil {
			return fmt.Errorf("training epoch %d failed: %w", epoch, err)
		}

		epochTime := time.Since(epochStartTime)
		t.Metrics.AverageEpochTime = time.Duration(int64(t.Metrics.AverageEpochTime)*int64(epoch)+int64(epochTime)) / time.Duration(epoch+1)

		// Call epoch end callback
		if t.OnEpochEnd != nil {
			if err := t.OnEpochEnd(epoch, t.Metrics); err != nil {
				return fmt.Errorf("epoch end callback failed: %w", err)
			}
		}

		// Reset data loader for next epoch
		if err := trainLoader.Reset(); err != nil {
			return fmt.Errorf("failed to reset train loader: %w", err)
		}

		// Clear cache periodically
		if epoch%10 == 0 {
			t.State.TensorCache.Clear()
			runtime.GC()
		}
	}

	t.Metrics.TotalTrainingTime = time.Since(startTime)

	// Call training end callback
	if t.OnTrainingEnd != nil {
		if err := t.OnTrainingEnd(t.Metrics); err != nil {
			return fmt.Errorf("training end callback failed: %w", err)
		}
	}

	return nil
}

// trainEpoch trains one epoch with memory optimization
func (t *Trainer) trainEpoch(trainLoader, validLoader DataLoader) error {
	t.Model.SetTraining(true)
	epochLoss := float32(0.0)
	batchCount := trainLoader.BatchCount()

	// Shuffle training data
	if err := trainLoader.Shuffle(); err != nil {
		return fmt.Errorf("failed to shuffle training data: %w", err)
	}

	for batch := 0; batch < batchCount; batch++ {
		t.State.Batch = batch
		batchStartTime := time.Now()

		// Call batch start callback
		if t.OnBatchStart != nil {
			if err := t.OnBatchStart(t.State.Epoch, batch); err != nil {
				return fmt.Errorf("batch start callback failed: %w", err)
			}
		}

		// Train one batch with gradient accumulation
		batchLoss, err := t.trainBatch(trainLoader, batch)
		if err != nil {
			return fmt.Errorf("training batch %d failed: %w", batch, err)
		}

		epochLoss += batchLoss
		t.State.Loss = batchLoss
		t.State.BatchTime = time.Since(batchStartTime)

		// Update metrics
		t.Metrics.Losses = append(t.Metrics.Losses, batchLoss)
		t.Metrics.BatchTimes = append(t.Metrics.BatchTimes, t.State.BatchTime)
		t.Metrics.LearningRates = append(t.Metrics.LearningRates, t.State.LearningRate)
		t.Metrics.GradientNorms = append(t.Metrics.GradientNorms, t.State.GradientNorm)

		// Call batch end callback
		if t.OnBatchEnd != nil {
			if err := t.OnBatchEnd(t.State.Epoch, batch, batchLoss); err != nil {
				return fmt.Errorf("batch end callback failed: %w", err)
			}
		}

		// Validation
		if t.Config.ValidationFreq > 0 && batch%t.Config.ValidationFreq == 0 && validLoader != nil {
			validLoss, err := t.validate(validLoader)
			if err != nil {
				return fmt.Errorf("validation failed: %w", err)
			}

			t.State.ValidationLoss = validLoss
			t.Metrics.ValidationLosses = append(t.Metrics.ValidationLosses, validLoss)

			// Track best validation loss
			if validLoss < t.Metrics.BestValidationLoss {
				t.Metrics.BestValidationLoss = validLoss
				t.Metrics.BestEpoch = t.State.Epoch
			}

			// Call validation callback
			if t.OnValidation != nil {
				if err := t.OnValidation(t.State.Epoch, batch, validLoss); err != nil {
					return fmt.Errorf("validation callback failed: %w", err)
				}
			}
		}

		// Clear cache periodically
		if t.Config.ClearCacheFreq > 0 && batch%t.Config.ClearCacheFreq == 0 {
			t.State.TensorCache.Clear()
			runtime.GC()
		}

		// Update memory usage tracking
		t.State.MemoryUsage = t.State.MemoryPool.GetUsage()
		t.Metrics.MemoryUsages = append(t.Metrics.MemoryUsages, t.State.MemoryUsage)
	}

	return nil
}

// trainBatch trains a single batch with gradient accumulation
func (t *Trainer) trainBatch(trainLoader DataLoader, batchIdx int) (float32, error) {
	totalLoss := float32(0.0)

	// Gradient accumulation loop
	for accumStep := 0; accumStep < t.Config.GradientAccumulationSteps; accumStep++ {
		// Calculate the true batch index considering accumulation steps
		realBatchIdx := batchIdx*t.Config.GradientAccumulationSteps + accumStep
		if realBatchIdx >= trainLoader.BatchCount() {
			break // No more data in this accumulation window
		}

		// Get batch data
		// inputs: *tensor.Tensor (e.g., [batchSize, H, W, C])
		// targets: *tensor.Tensor (a dummy tensor holding integer class indices as float32)
		inputs, targets, err := trainLoader.GetBatch(realBatchIdx)
		if err != nil {
			return 0, fmt.Errorf("failed to get batch %d: %w", realBatchIdx, err)
		}
		// Ensure inputs/targets are released even on early exit from this loop
		defer inputs.ReleaseGPU()
		defer targets.ReleaseGPU() // Targets will be released after data extraction in `backward`

		// Forward pass
		var outputs *tensor.Tensor
		if t.Config.EnableGradientCheckpoint {
			// Future: Implement gradient checkpointing here for memory efficiency
			// For now, it's false, so it uses regular forward.
			outputs, err = t.Model.Forward(inputs)
		} else {
			outputs, err = t.Model.Forward(inputs)
		}
		if err != nil {
			return 0, fmt.Errorf("forward pass failed for batch %d: %w", realBatchIdx, err)
		}
		defer outputs.ReleaseGPU() // Ensure outputs are released after backward pass

		// Compute loss (scalar value for logging/display)
		// We need to retrieve target indices from the tensor for SparseCategoricalCrossEntropyForward.
		// Note: The `targets` tensor is released within `t.backward` after its data is extracted.
		if err := targets.RetrieveCPU(); err != nil { // Retrieve target indices to CPU for this scalar loss computation
			return 0, fmt.Errorf("failed to retrieve target indices for loss computation to CPU: %w", err)
		}
		currentTargetIndicesForLoss := float32SliceToIntSlice(targets.Data) // Convert to []int
		targets.ReleaseGPU() // Release GPU memory of targets since we've retrieved to CPU and are done with this tensor.

		currentLoss, err := SparseCategoricalCrossEntropyForward(outputs, currentTargetIndicesForLoss)
		if err != nil {
			return 0, fmt.Errorf("loss computation failed for batch %d: %w", realBatchIdx, err)
		}

		// Scale loss for gradient accumulation (this is the value, not the gradient itself)
		scaledLoss := currentLoss / float32(t.Config.GradientAccumulationSteps)
		totalLoss += scaledLoss

		// Backward pass: This is where gradients are computed and stored in p.Gradient
		// Pass the outputs and the original `targets` tensor (which will be processed internally by `backward`)
		err = t.backward(outputs, targets) // `targets` here is the *original* tensor passed in, which has been retrieved to CPU by this point.
		if err != nil {
			return 0, fmt.Errorf("backward pass failed for batch %d: %w", realBatchIdx, err)
		}

		// Accumulate gradients: This will read from `p.Gradient` (populated by `lossGT.Backward()`)
		// and sum them into `trainer.State.AccumulatedGrads`.
		err = t.accumulateGradients()
		if err != nil {
			return 0, fmt.Errorf("gradient accumulation failed for batch %d: %w", realBatchIdx, err)
		}

		// `inputs` and `outputs` are deferred to be released. `targets` was released inside this loop.
	} // End of gradient accumulation loop

	// Apply accumulated gradients (only if accumulation steps reached)
	if t.State.AccumulationCount >= t.Config.GradientAccumulationSteps {
		err := t.applyGradients()
		if err != nil {
			return 0, fmt.Errorf("gradient application failed: %w", err)
		}

		// Reset accumulation state for the next application cycle
		t.resetGradientAccumulation()
		t.State.Step++ // Increment trainer step count
	}

	return totalLoss, nil
}

// forwardWithCheckpointing performs forward pass with gradient checkpointing
func (t *Trainer) forwardWithCheckpointing(inputs *tensor.Tensor) (*tensor.Tensor, error) {
	// This is a simplified implementation
	// In practice, you would implement model-specific checkpointing
	return t.Model.Forward(inputs)
}

// computeLoss computes the loss for the current batch
func (t *Trainer) computeLoss(outputs, targets *tensor.Tensor) (float32, error) {
	// This is a placeholder - implement based on your loss function
	// For now, assume MSE loss
	return MSELoss(outputs, targets)
}


// float32SliceToIntSlice converts a slice of float32 to a slice of int.
// This is a helper function used internally by the matrix package,
// especially for handling target indices from data loaders.
func float32SliceToIntSlice(f []float32) []int {
	i := make([]int, len(f))
	for idx, val := range f {
		i[idx] = int(val)
	}
	return i
}

// backward performs backward pass
// It takes the model's outputs and the true targets (as a tensor containing integer indices).
// It computes the loss and then triggers the backpropagation.
func (t *Trainer) backward(outputTensor *tensor.Tensor, targetTensorFromDataLoader *tensor.Tensor) error {
	// 1. Convert model outputs to a GradientTensor (requires gradients)
	outputsGT := NewGradientTensor(outputTensor, true) // Predictions usually require gradients

	// 2. Retrieve target indices from the targetTensorFromDataLoader.
	// This tensor is a "dummy" used by the DataLoader interface to pass integer indices.
	// We need to retrieve its data to CPU and convert it back to []int.
	if err := targetTensorFromDataLoader.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve target tensor from data loader to CPU: %w", err)
	}
	// IMPORTANT: The `float32SliceToIntSlice` helper needs to be defined in `main.go`
	// or the `matrix` package if it's a common utility. For this example, it's in `main.go`.
	targetIndices := float32SliceToIntSlice(targetTensorFromDataLoader.Data)

	// 3. Release the target tensor's GPU memory since its data has been copied to CPU.
	targetTensorFromDataLoader.ReleaseGPU()

	// 4. Compute sparse cross-entropy loss with gradient tracking.
	// This operation will build part of the computational graph and register its backward function.
	lossGT, err := GradSparseCrossEntropyLoss(outputsGT, targetIndices)
	if err != nil {
		// Release outputGT's tensor if loss computation fails
		if outputsGT != nil && outputsGT.Tensor != nil {
			outputsGT.Tensor.ReleaseGPU()
		}
		return fmt.Errorf("sparse cross-entropy loss failed in backward pass: %w", err)
	}

	// 5. Trigger backpropagation: Call Backward() on the final loss GradientTensor.
	// This is the core autograd step that traverses the computational graph
	// and populates the `Gradient` fields of all `RequiresGrad` GradientTensors
	// (i.e., your model's parameters).
	if err := lossGT.Backward(); err != nil {
		// Release outputGT's tensor and lossGT's tensor if backprop fails
		if outputsGT != nil && outputsGT.Tensor != nil {
			outputsGT.Tensor.ReleaseGPU()
		}
		if lossGT != nil && lossGT.Tensor != nil {
			lossGT.Tensor.ReleaseGPU()
		}
		return fmt.Errorf("backpropagation failed: %w", err)
	}

	// Release the tensors used in the backward pass that are no longer needed.
	// The `outputTensor` was passed in, and `lossGT.Tensor` was created.
	// `outputsGT.Tensor` directly points to `outputTensor`, so it's released outside.
	// `lossGT.Tensor` needs explicit release.
	if lossGT != nil && lossGT.Tensor != nil {
		lossGT.Tensor.ReleaseGPU()
	}

	// Note: `outputTensor` (model's forward output) is released outside this `backward` function
	// in `trainBatch` after this call.

	return nil
}

// accumulateGradients accumulates gradients for gradient accumulation
func (t *Trainer) accumulateGradients() error {
	params := t.Model.Parameters()

	for i, param := range params {
		if param.Gradient == nil {
			continue
		}

		paramName := fmt.Sprintf("param_%d", i)

		if accGrad, exists := t.State.AccumulatedGrads[paramName]; exists {
			// Accumulate with existing gradient
			err := AccumulateGradient(accGrad, param.Gradient)
			if err != nil {
				return fmt.Errorf("failed to accumulate gradient for %s: %w", paramName, err)
			}
		} else {
			// First gradient for this parameter
			gradData := make([]float32, len(param.Gradient.Data))
			copy(gradData, param.Gradient.Data)

			accGrad, err := tensor.NewTensor(param.Gradient.Shape, gradData)
			if err != nil {
				return fmt.Errorf("failed to create accumulated gradient for %s: %w", paramName, err)
			}

			t.State.AccumulatedGrads[paramName] = accGrad
		}
	}

	t.State.AccumulationCount++
	return nil
}

// applyGradients applies accumulated gradients using the optimizer
func (t *Trainer) applyGradients() error {
	params := t.Model.Parameters()

	// Compute gradient norm for monitoring
	gradNorm, err := t.computeGradientNorm()
	if err != nil {
		return fmt.Errorf("failed to compute gradient norm: %w", err)
	}
	t.State.GradientNorm = gradNorm

	// Convert accumulated gradients to tensor slices
	paramTensors := make([]*tensor.Tensor, len(params))
	gradTensors := make([]*tensor.Tensor, len(params))

	for i, param := range params {
		paramTensors[i] = param.Tensor
		paramName := fmt.Sprintf("param_%d", i)
		if accGrad, exists := t.State.AccumulatedGrads[paramName]; exists {
			gradTensors[i] = accGrad
		} else {
			// Create zero gradient if not accumulated
			gradData := make([]float32, len(param.Tensor.Data))
			gradTensor, err := tensor.NewTensor(param.Tensor.Shape, gradData)
			if err != nil {
				return fmt.Errorf("failed to create zero gradient: %w", err)
			}
			gradTensors[i] = gradTensor
		}
	}

	// Apply gradient clipping if enabled
	if t.Config.GradientClipping > 0 {
		actualNorm, err := optimizer.ClipGradsByNorm(gradTensors, t.Config.GradientClipping)
		if err != nil {
			return fmt.Errorf("gradient clipping failed: %w", err)
		}
		t.State.GradientNorm = actualNorm
	}

	// Apply optimizer step
	err = t.State.OptimizerInstance.Step(paramTensors, gradTensors)
	if err != nil {
		return fmt.Errorf("optimizer step failed: %w", err)
	}

	// Update learning rate with scheduler
	if t.State.SchedulerInstance != nil {
		err = t.State.SchedulerInstance.Step(int64(t.State.Step))
		if err != nil {
			return fmt.Errorf("scheduler step failed: %w", err)
		}
		t.State.LearningRate = t.State.SchedulerInstance.GetLR()
	} else {
		t.State.LearningRate = t.State.OptimizerInstance.GetLearningRate()
	}

	return nil
}

// computeGradientNorm computes the L2 norm of accumulated gradients
func (t *Trainer) computeGradientNorm() (float32, error) {
	gradTensors := make([]*tensor.Tensor, 0, len(t.State.AccumulatedGrads))
	for _, grad := range t.State.AccumulatedGrads {
		gradTensors = append(gradTensors, grad)
	}

	if len(gradTensors) == 0 {
		return 0.0, nil
	}

	return optimizer.ComputeGradNorm(gradTensors)
}

// clipGradients applies gradient clipping (removed since we use optimizer.ClipGradsByNorm now)

// resetGradientAccumulation resets gradient accumulation state
func (t *Trainer) resetGradientAccumulation() {
	// Release accumulated gradients
	for _, grad := range t.State.AccumulatedGrads {
		grad.ReleaseGPU()
	}

	// Clear the map
	t.State.AccumulatedGrads = make(map[string]*tensor.Tensor)
	t.State.AccumulationCount = 0
}

// validate performs validation on the validation dataset
func (t *Trainer) validate(validLoader DataLoader) (float32, error) {
	t.Model.SetTraining(false)
	defer t.Model.SetTraining(true)

	NoGradContext(func() {
		totalLoss := float32(0.0)
		batchCount := validLoader.BatchCount()

		for batch := 0; batch < batchCount; batch++ {
			inputs, targets, err := validLoader.GetBatch(batch)
			if err != nil {
				continue
			}

			outputs, err := t.Model.Forward(inputs)
			if err != nil {
				inputs.ReleaseGPU()
				targets.ReleaseGPU()
				continue
			}

			loss, err := t.computeLoss(outputs, targets)
			if err != nil {
				inputs.ReleaseGPU()
				targets.ReleaseGPU()
				outputs.ReleaseGPU()
				continue
			}

			totalLoss += loss

			// Release tensors
			inputs.ReleaseGPU()
			targets.ReleaseGPU()
			outputs.ReleaseGPU()
		}

		t.State.ValidationLoss = totalLoss / float32(batchCount)
	})

	return t.State.ValidationLoss, nil
}

// initializeOptimizer creates and initializes the optimizer
func (t *Trainer) initializeOptimizer() error {
	switch t.Config.OptimizerType {
	case OptimizerSGD:
		config := optimizer.SGDConfig{
			OptimizerConfig: optimizer.OptimizerConfig{
				LearningRate: t.Config.LearningRate,
				WeightDecay:  t.Config.WeightDecay,
			},
			Momentum: t.Config.Momentum,
		}
		t.State.OptimizerInstance = optimizer.NewSGD(config)

	case OptimizerAdam:
		config := optimizer.AdamConfig{
			OptimizerConfig: optimizer.OptimizerConfig{
				LearningRate: t.Config.LearningRate,
				WeightDecay:  t.Config.WeightDecay,
			},
			Beta1:   t.Config.Beta1,
			Beta2:   t.Config.Beta2,
			Epsilon: t.Config.Epsilon,
		}
		t.State.OptimizerInstance = optimizer.NewAdam(config)

	case OptimizerAdamW:
		config := optimizer.AdamWConfig{
			OptimizerConfig: optimizer.OptimizerConfig{
				LearningRate: t.Config.LearningRate,
				WeightDecay:  t.Config.WeightDecay,
			},
			Beta1:   t.Config.Beta1,
			Beta2:   t.Config.Beta2,
			Epsilon: t.Config.Epsilon,
		}
		t.State.OptimizerInstance = optimizer.NewAdamW(config)

	case OptimizerRMSprop:
		config := optimizer.RMSpropConfig{
			OptimizerConfig: optimizer.OptimizerConfig{
				LearningRate: t.Config.LearningRate,
				WeightDecay:  t.Config.WeightDecay,
			},
			Alpha:    0.99, // Default alpha
			Epsilon:  t.Config.Epsilon,
			Momentum: t.Config.Momentum,
		}
		t.State.OptimizerInstance = optimizer.NewRMSprop(config)

	default:
		return fmt.Errorf("unsupported optimizer type: %d", t.Config.OptimizerType)
	}

	return nil
}

// initializeScheduler creates and initializes the learning rate scheduler
func (t *Trainer) initializeScheduler() error {
	if t.Config.SchedulerType == NoScheduler {
		return nil
	}

	switch t.Config.SchedulerType {
	case StepLR:
		t.State.SchedulerInstance = optimizer.NewStepDecayScheduler(
			t.Config.LearningRate,
			t.Config.SchedulerGamma,
			int64(t.Config.SchedulerStepSize),
		)

	case ExponentialLR:
		t.State.SchedulerInstance = optimizer.NewExponentialDecayScheduler(
			t.Config.LearningRate,
			t.Config.SchedulerGamma,
			int64(t.Config.SchedulerStepSize),
		)

	case CosineAnnealingLR:
		t.State.SchedulerInstance = optimizer.NewCosineAnnealingScheduler(
			t.Config.LearningRate,
			t.Config.LearningRate*0.01, // min LR as 1% of initial
			int64(t.Config.TotalSteps),
		)

	case PolynomialLR:
		t.State.SchedulerInstance = optimizer.NewPolynomialDecayScheduler(
			t.Config.LearningRate,
			t.Config.LearningRate*0.01, // final LR as 1% of initial
			int64(t.Config.TotalSteps),
			1.0, // Linear decay
		)

	case WarmupLR:
		t.State.SchedulerInstance = optimizer.NewWarmupScheduler(
			t.Config.LearningRate,
			int64(t.Config.WarmupSteps),
			0, // Linear warmup
		)

	default:
		return fmt.Errorf("unsupported scheduler type: %d", t.Config.SchedulerType)
	}

	if t.State.SchedulerInstance != nil {
		t.State.SchedulerInstance.SetOptimizer(t.State.OptimizerInstance)
	}

	return nil
}

// GetMetrics returns current training metrics
func (t *Trainer) GetMetrics() *TrainingMetrics {
	return t.Metrics
}

// GetState returns current training state
func (t *Trainer) GetState() *TrainingState {
	return t.State
}

// SaveCheckpoint saves the current training state
func (t *Trainer) SaveCheckpoint(filepath string) error {
	// This would save the model, optimizer state, and training state
	// Implementation depends on your serialization requirements
	return fmt.Errorf("checkpoint saving not implemented yet")
}

// LoadCheckpoint loads a training state from checkpoint
func (t *Trainer) LoadCheckpoint(filepath string) error {
	// This would load the model, optimizer state, and training state
	// Implementation depends on your serialization requirements
	return fmt.Errorf("checkpoint loading not implemented yet")
}

// Release releases GPU resources used by the trainer
func (t *Trainer) Release() {
	// Release accumulated gradients
	for _, grad := range t.State.AccumulatedGrads {
		grad.ReleaseGPU()
	}

	// Release optimizer state
	if t.State.OptimizerInstance != nil {
		t.State.OptimizerInstance.ReleaseGPU()
	}

	// Release memory pool and cache
	if t.State.MemoryPool != nil {
		t.State.MemoryPool.ReleaseAll()
	}

	if t.State.TensorCache != nil {
		t.State.TensorCache.Clear()
	}
}

func ZeroTensorGPU(t *tensor.Tensor) error {
	if t == nil {
		return fmt.Errorf("cannot zero a nil tensor")
	}
	if err := t.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to ensure tensor is on GPU before zeroing: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_tensor_fill(
		C.GPUPtr(t.GPUPtr()),
		C.long(len(t.Data)), // Size in number of float32 elements
		C.float(0.0),        // Value to fill with
		C.DevicePtr(t.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU tensor fill (zeroing) failed (code %d): %s", retCode, errMsg)
	}
	return nil
}
