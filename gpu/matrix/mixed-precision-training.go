package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"math"
	"time"

	"github.com/tsawler/go-nngpu/gpu/optimizer"
	"github.com/tsawler/go-nngpu/tensor"
)

// LearningRateScheduler is a placeholder for demo purposes
type LearningRateScheduler struct {
	step int
}

// Step updates the learning rate scheduler
func (s *LearningRateScheduler) Step() {
	if s != nil {
		s.step++
	}
}

// ClearGPUCache clears GPU cache (placeholder for demo)
func ClearGPUCache() {
	// In real implementation, would clear GPU cache
}

// MixedPrecisionTrainingConfig extends TrainingConfig with mixed precision settings
type MixedPrecisionTrainingConfig struct {
	TrainingConfig
	MixedPrecision *MixedPrecisionConfig
}

// AutocastMode defines when to use float16 vs float32
type AutocastMode int

const (
	AutocastDisabled AutocastMode = iota
	AutocastForward                // Use float16 for forward pass only
	AutocastFull                   // Use float16 for forward and backward (with scaling)
)

// MixedPrecisionTrainingLoop implements automatic mixed precision training
type MixedPrecisionTrainingLoop struct {
	config    *MixedPrecisionTrainingConfig
	trainer   *MixedPrecisionTrainer
	optimizer optimizer.Optimizer
	scheduler *LearningRateScheduler
	
	// Training state
	currentEpoch    int
	currentStep     int64
	totalLoss       float32
	
	// Performance metrics
	trainTime       time.Duration
	forwardTime     time.Duration
	backwardTime    time.Duration
	optimizerTime   time.Duration
	
	// Mixed precision stats
	overflowCount   int64
	skippedSteps    int64
	lossScaleUpdates int64
}

// NewMixedPrecisionTrainingLoop creates a new mixed precision training loop
func NewMixedPrecisionTrainingLoop(config *MixedPrecisionTrainingConfig, opt optimizer.Optimizer) (*MixedPrecisionTrainingLoop, error) {
	if config == nil {
		return nil, fmt.Errorf("training config cannot be nil")
	}
	
	if config.MixedPrecision == nil {
		config.MixedPrecision = DefaultMixedPrecisionConfig()
	}

	// Create mixed precision trainer
	trainer, err := NewMixedPrecisionTrainer(config.MixedPrecision)
	if err != nil {
		return nil, fmt.Errorf("failed to create mixed precision trainer: %w", err)
	}

	// Create learning rate scheduler if configured
	var scheduler *LearningRateScheduler
	if config.SchedulerType != NoScheduler {
		// Note: In real implementation, would create scheduler based on config
		// For demo purposes, keeping simple
		scheduler = nil
	}

	return &MixedPrecisionTrainingLoop{
		config:    config,
		trainer:   trainer,
		optimizer: opt,
		scheduler: scheduler,
	}, nil
}

// TrainEpoch trains for one epoch with mixed precision
func (loop *MixedPrecisionTrainingLoop) TrainEpoch(
	inputs []*tensor.Tensor,
	targets []*tensor.Tensor,
	weights []*tensor.Tensor,
	forwardFunc func(*tensor.Tensor, []*tensor.Tensor) (*tensor.Tensor, error),
	lossFunc func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error),
	backwardFunc func(*tensor.Tensor, []*tensor.Tensor) ([]*tensor.Tensor, error),
) error {
	
	epochStart := time.Now()
	loop.currentEpoch++
	epochLoss := float32(0.0)
	processedSamples := 0

	fmt.Printf("Epoch %d/%d - Mixed Precision Training\n", loop.currentEpoch, loop.config.EpochCount)
	fmt.Printf("Loss Scale: %.0f, Autocast: %s\n", 
		loop.trainer.GetCurrentLossScale(),
		loop.getAutocastStatus())

	// Process batches
	for batchIdx := 0; batchIdx < len(inputs); batchIdx += loop.config.BatchSize {
		batchStart := time.Now()
		
		// Get batch
		batchEnd := batchIdx + loop.config.BatchSize
		if batchEnd > len(inputs) {
			batchEnd = len(inputs)
		}
		
		batchInputs := inputs[batchIdx:batchEnd]
		batchTargets := targets[batchIdx:batchEnd]
		
		// Process batch with mixed precision
		batchLoss, err := loop.processBatch(batchInputs, batchTargets, weights, forwardFunc, lossFunc, backwardFunc)
		if err != nil {
			return fmt.Errorf("failed to process batch %d: %w", batchIdx/loop.config.BatchSize, err)
		}
		
		epochLoss += batchLoss
		processedSamples += len(batchInputs)
		
		// Update learning rate scheduler
		if loop.scheduler != nil {
			loop.scheduler.Step()
		}
		
		loop.currentStep++
		
		// Print batch progress
		if batchIdx%100 == 0 || batchIdx+loop.config.BatchSize >= len(inputs) {
			batchTime := time.Since(batchStart)
			avgLoss := epochLoss / float32(processedSamples)
			throughput := float64(len(batchInputs)) / batchTime.Seconds()
			
			fmt.Printf("  Batch %d/%d - Loss: %.6f, LR: %.6f, Throughput: %.1f samples/s\n",
				batchIdx/loop.config.BatchSize+1,
				(len(inputs)+loop.config.BatchSize-1)/loop.config.BatchSize,
				avgLoss,
				loop.optimizer.GetLearningRate(),
				throughput)
			
			if loop.trainer.config.Enabled {
				fmt.Printf("    MP Stats - Loss Scale: %.0f, Overflows: %d, Skipped: %d\n",
					loop.trainer.GetCurrentLossScale(),
					loop.overflowCount,
					loop.skippedSteps)
			}
		}
		
		// Memory cleanup
		if loop.config.ClearCacheFreq > 0 && (batchIdx/loop.config.BatchSize)%loop.config.ClearCacheFreq == 0 {
			ClearGPUCache()
		}
	}
	
	loop.trainTime += time.Since(epochStart)
	avgLoss := epochLoss / float32(len(inputs))
	
	fmt.Printf("Epoch %d completed - Avg Loss: %.6f, Time: %.2fs\n", 
		loop.currentEpoch, avgLoss, time.Since(epochStart).Seconds())
	
	return nil
}

// processBatch handles a single batch with mixed precision
func (loop *MixedPrecisionTrainingLoop) processBatch(
	inputs []*tensor.Tensor,
	targets []*tensor.Tensor,
	weights []*tensor.Tensor,
	forwardFunc func(*tensor.Tensor, []*tensor.Tensor) (*tensor.Tensor, error),
	lossFunc func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error),
	backwardFunc func(*tensor.Tensor, []*tensor.Tensor) ([]*tensor.Tensor, error),
) (float32, error) {
	
	totalLoss := float32(0.0)
	
	// Zero gradients
	var accumulatedGrads []*tensor.Tensor
	
	// Process each sample in the batch
	for _, input := range inputs {
		// Forward pass with autocast
		forwardStart := time.Now()
		
		var predictions *tensor.Tensor
		var err error
		
		if loop.trainer.config.Enabled {
			// Use mixed precision forward pass
			predictions, err = loop.mixedPrecisionForward(input, weights, forwardFunc)
		} else {
			// Standard float32 forward pass
			predictions, err = forwardFunc(input, weights)
		}
		
		if err != nil {
			return 0, fmt.Errorf("forward pass failed: %w", err)
		}
		
		loop.forwardTime += time.Since(forwardStart)
		
		// Compute loss
		targetIdx := 0 // Assuming one target per input for simplicity
		if targetIdx < len(targets) {
			loss, err := lossFunc(predictions, targets[targetIdx])
			if err != nil {
				return 0, fmt.Errorf("loss computation failed: %w", err)
			}
			
			// Scale loss for mixed precision
			if loop.trainer.config.Enabled {
				scaledLoss, err := loop.trainer.ScaleGradients(loss)
				if err != nil {
					return 0, fmt.Errorf("loss scaling failed: %w", err)
				}
				loss = scaledLoss
			}
			
			if err := loss.RetrieveCPU(); err != nil {
				return 0, fmt.Errorf("failed to move loss to CPU: %w", err)
			}
			totalLoss += loss.Data[0]
		}
		
		// Backward pass
		backwardStart := time.Now()
		grads, err := backwardFunc(predictions, weights)
		if err != nil {
			return 0, fmt.Errorf("backward pass failed: %w", err)
		}
		
		// Unscale gradients if using mixed precision
		if loop.trainer.config.Enabled {
			for i, grad := range grads {
				unscaledGrad, err := loop.trainer.UnscaleGradients(grad)
				if err != nil {
					return 0, fmt.Errorf("gradient unscaling failed: %w", err)
				}
				grads[i] = unscaledGrad
			}
			
			// Check for overflow
			if loop.trainer.GetOverflowStatus() {
				loop.overflowCount++
			}
		}
		
		loop.backwardTime += time.Since(backwardStart)
		
		// Accumulate gradients
		if accumulatedGrads == nil {
			accumulatedGrads = grads
		} else {
			for i, grad := range grads {
				if err := accumulatedGrads[i].RetrieveCPU(); err != nil {
					return 0, fmt.Errorf("failed to move accumulated grad to CPU: %w", err)
				}
				if err := grad.RetrieveCPU(); err != nil {
					return 0, fmt.Errorf("failed to move grad to CPU: %w", err)
				}
				
				for j := range accumulatedGrads[i].Data {
					accumulatedGrads[i].Data[j] += grad.Data[j]
				}
			}
		}
	}
	
	// Optimizer step
	optimizerStart := time.Now()
	
	// Check if we should skip this step due to gradient overflow
	if loop.trainer.config.Enabled && loop.trainer.ShouldSkipStep() {
		loop.skippedSteps++
		fmt.Printf("    Skipping optimizer step due to gradient overflow\n")
	} else {
		// Average gradients
		if len(inputs) > 1 {
			for _, grad := range accumulatedGrads {
				for i := range grad.Data {
					grad.Data[i] /= float32(len(inputs))
				}
			}
		}
		
		// Apply gradient clipping if configured
		if loop.config.GradientClipping > 0 {
			loop.clipGradients(accumulatedGrads, loop.config.GradientClipping)
		}
		
		// Optimizer step
		if err := loop.optimizer.Step(weights, accumulatedGrads); err != nil {
			return 0, fmt.Errorf("optimizer step failed: %w", err)
		}
	}
	
	// Update loss scale
	if loop.trainer.config.Enabled {
		loop.trainer.UpdateLossScale()
		loop.lossScaleUpdates++
	}
	
	loop.optimizerTime += time.Since(optimizerStart)
	
	return totalLoss / float32(len(inputs)), nil
}

// mixedPrecisionForward performs forward pass with automatic mixed precision
func (loop *MixedPrecisionTrainingLoop) mixedPrecisionForward(
	input *tensor.Tensor,
	weights []*tensor.Tensor,
	forwardFunc func(*tensor.Tensor, []*tensor.Tensor) (*tensor.Tensor, error),
) (*tensor.Tensor, error) {
	
	// Convert input to float16 for computation
	inputF16, err := loop.trainer.ConvertTensorToFloat16(input)
	if err != nil {
		return nil, fmt.Errorf("failed to convert input to float16: %w", err)
	}
	
	// Convert weights to float16
	weightsF16 := make([]*tensor.Tensor, len(weights))
	for i, weight := range weights {
		weightF16, err := loop.trainer.ConvertTensorToFloat16(weight)
		if err != nil {
			return nil, fmt.Errorf("failed to convert weight %d to float16: %w", i, err)
		}
		weightsF16[i] = weightF16
	}
	
	// Perform forward pass in float16
	return forwardFunc(inputF16, weightsF16)
}

// clipGradients applies gradient clipping
func (loop *MixedPrecisionTrainingLoop) clipGradients(grads []*tensor.Tensor, maxNorm float32) {
	// Compute total gradient norm
	totalNorm := float32(0.0)
	for _, grad := range grads {
		if err := grad.RetrieveCPU(); err != nil {
			continue
		}
		for _, val := range grad.Data {
			totalNorm += val * val
		}
	}
	totalNorm = float32(math.Sqrt(float64(totalNorm)))
	
	// Apply clipping if necessary
	if totalNorm > maxNorm {
		clipCoeff := maxNorm / totalNorm
		for _, grad := range grads {
			if err := grad.RetrieveCPU(); err != nil {
				continue
			}
			for i := range grad.Data {
				grad.Data[i] *= clipCoeff
			}
		}
	}
}

// getAutocastStatus returns a string describing the current autocast mode
func (loop *MixedPrecisionTrainingLoop) getAutocastStatus() string {
	if !loop.trainer.config.Enabled {
		return "Disabled"
	}
	return "Enabled (Forward + Backward)"
}

// GetTrainingStats returns comprehensive training statistics
func (loop *MixedPrecisionTrainingLoop) GetTrainingStats() map[string]interface{} {
	stats := map[string]interface{}{
		"current_epoch":     loop.currentEpoch,
		"current_step":      loop.currentStep,
		"total_train_time":  loop.trainTime.Seconds(),
		"forward_time":      loop.forwardTime.Seconds(),
		"backward_time":     loop.backwardTime.Seconds(),
		"optimizer_time":    loop.optimizerTime.Seconds(),
	}
	
	if loop.trainer.config.Enabled {
		stats["mixed_precision_enabled"] = true
		stats["current_loss_scale"] = loop.trainer.GetCurrentLossScale()
		stats["overflow_count"] = loop.overflowCount
		stats["skipped_steps"] = loop.skippedSteps
		stats["loss_scale_updates"] = loop.lossScaleUpdates
		stats["overflow_rate"] = float64(loop.overflowCount) / float64(loop.currentStep)
	} else {
		stats["mixed_precision_enabled"] = false
	}
	
	return stats
}

// Cleanup releases resources
func (loop *MixedPrecisionTrainingLoop) Cleanup() {
	if loop.trainer != nil {
		loop.trainer.Cleanup()
	}
}