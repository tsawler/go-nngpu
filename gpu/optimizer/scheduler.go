package optimizer

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"math"
	"runtime"
)

// LRScheduler represents a learning rate scheduler interface
type LRScheduler interface {
	Step(step int64) error
	GetLR() float32
	SetOptimizer(opt Optimizer)
}

// ExponentialDecayScheduler implements exponential decay learning rate scheduling
type ExponentialDecayScheduler struct {
	optimizer  Optimizer
	initialLR  float32
	decayRate  float32
	decaySteps int64
	currentLR  float32
}

// NewExponentialDecayScheduler creates a new exponential decay scheduler
func NewExponentialDecayScheduler(initialLR, decayRate float32, decaySteps int64) *ExponentialDecayScheduler {
	return &ExponentialDecayScheduler{
		initialLR:  initialLR,
		decayRate:  decayRate,
		decaySteps: decaySteps,
		currentLR:  initialLR,
	}
}

// Step updates the learning rate based on the current step
func (s *ExponentialDecayScheduler) Step(step int64) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_lr_exponential_decay(
		(*C.float)(&s.currentLR),
		C.float(s.initialLR),
		C.float(s.decayRate),
		C.long(step),
		C.long(s.decaySteps),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("exponential decay step failed (code %d): %s", retCode, errMsg)
	}

	if s.optimizer != nil {
		s.optimizer.SetLearningRate(s.currentLR)
	}

	return nil
}

// GetLR returns the current learning rate
func (s *ExponentialDecayScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *ExponentialDecayScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// StepDecayScheduler implements step decay learning rate scheduling
type StepDecayScheduler struct {
	optimizer Optimizer
	initialLR float32
	gamma     float32
	stepSize  int64
	currentLR float32
}

// NewStepDecayScheduler creates a new step decay scheduler
func NewStepDecayScheduler(initialLR, gamma float32, stepSize int64) *StepDecayScheduler {
	return &StepDecayScheduler{
		initialLR: initialLR,
		gamma:     gamma,
		stepSize:  stepSize,
		currentLR: initialLR,
	}
}

// Step updates the learning rate based on the current step
func (s *StepDecayScheduler) Step(step int64) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_lr_step_decay(
		(*C.float)(&s.currentLR),
		C.float(s.initialLR),
		C.float(s.gamma),
		C.long(step),
		C.long(s.stepSize),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("step decay step failed (code %d): %s", retCode, errMsg)
	}

	if s.optimizer != nil {
		s.optimizer.SetLearningRate(s.currentLR)
	}

	return nil
}

// GetLR returns the current learning rate
func (s *StepDecayScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *StepDecayScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// CosineAnnealingScheduler implements cosine annealing learning rate scheduling
type CosineAnnealingScheduler struct {
	optimizer  Optimizer
	initialLR  float32
	minLR      float32
	totalSteps int64
	currentLR  float32
}

// NewCosineAnnealingScheduler creates a new cosine annealing scheduler
func NewCosineAnnealingScheduler(initialLR, minLR float32, totalSteps int64) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{
		initialLR:  initialLR,
		minLR:      minLR,
		totalSteps: totalSteps,
		currentLR:  initialLR,
	}
}

// Step updates the learning rate based on the current step
func (s *CosineAnnealingScheduler) Step(step int64) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_lr_cosine_annealing(
		(*C.float)(&s.currentLR),
		C.float(s.initialLR),
		C.float(s.minLR),
		C.long(step),
		C.long(s.totalSteps),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("cosine annealing step failed (code %d): %s", retCode, errMsg)
	}

	if s.optimizer != nil {
		s.optimizer.SetLearningRate(s.currentLR)
	}

	return nil
}

// GetLR returns the current learning rate
func (s *CosineAnnealingScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *CosineAnnealingScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// PolynomialDecayScheduler implements polynomial decay learning rate scheduling
type PolynomialDecayScheduler struct {
	optimizer  Optimizer
	initialLR  float32
	finalLR    float32
	totalSteps int64
	power      float32
	currentLR  float32
}

// NewPolynomialDecayScheduler creates a new polynomial decay scheduler
func NewPolynomialDecayScheduler(initialLR, finalLR float32, totalSteps int64, power float32) *PolynomialDecayScheduler {
	return &PolynomialDecayScheduler{
		initialLR:  initialLR,
		finalLR:    finalLR,
		totalSteps: totalSteps,
		power:      power,
		currentLR:  initialLR,
	}
}

// Step updates the learning rate based on the current step
func (s *PolynomialDecayScheduler) Step(step int64) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_lr_polynomial_decay(
		(*C.float)(&s.currentLR),
		C.float(s.initialLR),
		C.float(s.finalLR),
		C.long(step),
		C.long(s.totalSteps),
		C.float(s.power),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("polynomial decay step failed (code %d): %s", retCode, errMsg)
	}

	if s.optimizer != nil {
		s.optimizer.SetLearningRate(s.currentLR)
	}

	return nil
}

// GetLR returns the current learning rate
func (s *PolynomialDecayScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *PolynomialDecayScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// WarmupScheduler implements learning rate warmup
type WarmupScheduler struct {
	optimizer     Optimizer
	targetLR      float32
	warmupSteps   int64
	warmupType    int // 0: linear, 1: exponential
	currentLR     float32
	baseScheduler LRScheduler // Optional base scheduler to use after warmup
}

// NewWarmupScheduler creates a new warmup scheduler
func NewWarmupScheduler(targetLR float32, warmupSteps int64, warmupType int) *WarmupScheduler {
	return &WarmupScheduler{
		targetLR:    targetLR,
		warmupSteps: warmupSteps,
		warmupType:  warmupType,
		currentLR:   0.0,
	}
}

// SetBaseScheduler sets a base scheduler to use after warmup completes
func (s *WarmupScheduler) SetBaseScheduler(scheduler LRScheduler) {
	s.baseScheduler = scheduler
}

// Step updates the learning rate based on the current step
func (s *WarmupScheduler) Step(step int64) error {
	if step < s.warmupSteps {
		// Apply warmup
		runtime.LockOSThread()
		var cErr C.CError
		retCode := C.perform_lr_warmup(
			(*C.float)(&s.currentLR),
			C.float(s.targetLR),
			C.long(step),
			C.long(s.warmupSteps),
			C.int(s.warmupType),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return fmt.Errorf("warmup step failed (code %d): %s", retCode, errMsg)
		}

		if s.optimizer != nil {
			s.optimizer.SetLearningRate(s.currentLR)
		}
	} else {
		// Use base scheduler if available, otherwise keep target LR
		if s.baseScheduler != nil {
			err := s.baseScheduler.Step(step - s.warmupSteps)
			if err != nil {
				return err
			}
			// Update our current LR to match the base scheduler
			s.currentLR = s.baseScheduler.GetLR()
			if s.optimizer != nil {
				s.optimizer.SetLearningRate(s.currentLR)
			}
		} else {
			s.currentLR = s.targetLR
			if s.optimizer != nil {
				s.optimizer.SetLearningRate(s.currentLR)
			}
		}
	}

	return nil
}

// GetLR returns the current learning rate
func (s *WarmupScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *WarmupScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
	if s.baseScheduler != nil {
		s.baseScheduler.SetOptimizer(opt)
	}
}

// OneCycleLRScheduler implements the one-cycle learning rate policy
type OneCycleLRScheduler struct {
	optimizer      Optimizer
	maxLR          float32
	totalSteps     int64
	pctStart       float32 // Percentage of cycle spent increasing LR
	annealStrategy string  // "cos" or "linear"
	currentLR      float32
}

// NewOneCycleLRScheduler creates a new one-cycle LR scheduler
func NewOneCycleLRScheduler(maxLR float32, totalSteps int64, pctStart float32, annealStrategy string) *OneCycleLRScheduler {
	return &OneCycleLRScheduler{
		maxLR:          maxLR,
		totalSteps:     totalSteps,
		pctStart:       pctStart,
		annealStrategy: annealStrategy,
		currentLR:      maxLR / 25.0, // Start with a low LR
	}
}

// Step updates the learning rate based on the current step
func (s *OneCycleLRScheduler) Step(step int64) error {
	if step >= s.totalSteps {
		step = s.totalSteps - 1
	}

	// progress := float32(step) / float32(s.totalSteps)
	peakStep := int64(float32(s.totalSteps) * s.pctStart)

	if step <= peakStep {
		// Increasing phase
		stepProgress := float32(step) / float32(peakStep)
		s.currentLR = (s.maxLR / 25.0) + stepProgress*(s.maxLR-s.maxLR/25.0)
	} else {
		// Decreasing phase
		remainingSteps := s.totalSteps - peakStep
		stepProgress := float32(step-peakStep) / float32(remainingSteps)

		if s.annealStrategy == "cos" {
			// Cosine annealing
			s.currentLR = (s.maxLR / 25.0) + (s.maxLR-s.maxLR/25.0)*0.5*(1.0+float32(math.Cos(math.Pi*float64(stepProgress))))
		} else {
			// Linear annealing
			s.currentLR = s.maxLR - stepProgress*(s.maxLR-s.maxLR/25.0)
		}
	}

	if s.optimizer != nil {
		s.optimizer.SetLearningRate(s.currentLR)
	}

	return nil
}

// GetLR returns the current learning rate
func (s *OneCycleLRScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *OneCycleLRScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// ReduceLROnPlateauScheduler reduces learning rate when a metric has stopped improving
type ReduceLROnPlateauScheduler struct {
	optimizer     Optimizer
	factor        float32 // Factor by which LR is reduced
	patience      int64   // Number of epochs with no improvement after which LR is reduced
	threshold     float32 // Threshold for measuring the new optimum
	cooldown      int64   // Number of epochs to wait before resuming normal operation
	minLR         float32 // Lower bound on LR
	mode          string  // "min" or "max"
	currentLR     float32
	bestMetric    float32
	waitCount     int64
	cooldownCount int64
}

// NewReduceLROnPlateauScheduler creates a new ReduceLROnPlateau scheduler
func NewReduceLROnPlateauScheduler(initialLR, factor float32, patience int64, threshold, minLR float32, mode string) *ReduceLROnPlateauScheduler {
	var bestMetric float32
	if mode == "min" {
		bestMetric = float32(math.Inf(1)) // positive infinity
	} else {
		bestMetric = float32(math.Inf(-1)) // negative infinity
	}

	return &ReduceLROnPlateauScheduler{
		factor:        factor,
		patience:      patience,
		threshold:     threshold,
		minLR:         minLR,
		mode:          mode,
		currentLR:     initialLR,
		bestMetric:    bestMetric,
		waitCount:     0,
		cooldownCount: 0,
	}
}

// Step updates the learning rate based on the metric value
func (s *ReduceLROnPlateauScheduler) StepWithMetric(metric float32) error {
	if s.cooldownCount > 0 {
		s.cooldownCount--
		return nil
	}

	improved := false
	if s.mode == "min" {
		improved = metric < s.bestMetric-s.threshold
	} else {
		improved = metric > s.bestMetric+s.threshold
	}

	if improved {
		s.bestMetric = metric
		s.waitCount = 0
	} else {
		s.waitCount++
		if s.waitCount >= s.patience {
			newLR := s.currentLR * s.factor
			if newLR >= s.minLR {
				s.currentLR = newLR
				if s.optimizer != nil {
					s.optimizer.SetLearningRate(s.currentLR)
				}
			}
			s.waitCount = 0
			s.cooldownCount = s.cooldown
		}
	}

	return nil
}

// Step is required by the interface but does nothing for this scheduler
func (s *ReduceLROnPlateauScheduler) Step(step int64) error {
	// This scheduler doesn't use step count, only metric values
	return nil
}

// GetLR returns the current learning rate
func (s *ReduceLROnPlateauScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *ReduceLROnPlateauScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// CyclicLRScheduler implements cyclical learning rates
type CyclicLRScheduler struct {
	optimizer    Optimizer
	baseLR       float32
	maxLR        float32
	stepSizeUp   int64   // Number of training iterations in the increasing half of a cycle
	stepSizeDown int64   // Number of training iterations in the decreasing half of a cycle
	mode         string  // "triangular", "triangular2", or "exp_range"
	gamma        float32 // Constant in 'exp_range' scaling function
	currentLR    float32
	cycleCount   int64
}

// NewCyclicLRScheduler creates a new cyclical LR scheduler
func NewCyclicLRScheduler(baseLR, maxLR float32, stepSizeUp, stepSizeDown int64, mode string, gamma float32) *CyclicLRScheduler {
	return &CyclicLRScheduler{
		baseLR:       baseLR,
		maxLR:        maxLR,
		stepSizeUp:   stepSizeUp,
		stepSizeDown: stepSizeDown,
		mode:         mode,
		gamma:        gamma,
		currentLR:    baseLR,
		cycleCount:   0,
	}
}

// Step updates the learning rate based on the current step
func (s *CyclicLRScheduler) Step(step int64) error {
	cycleSize := s.stepSizeUp + s.stepSizeDown
	cycle := step / cycleSize
	x := step % cycleSize

	if x <= s.stepSizeUp {
		// Increasing phase
		progress := float32(x) / float32(s.stepSizeUp)
		s.currentLR = s.baseLR + (s.maxLR-s.baseLR)*progress
	} else {
		// Decreasing phase
		progress := float32(x-s.stepSizeUp) / float32(s.stepSizeDown)
		s.currentLR = s.maxLR - (s.maxLR-s.baseLR)*progress
	}

	// Apply mode-specific scaling
	switch s.mode {
	case "triangular2":
		s.currentLR = s.baseLR + (s.currentLR-s.baseLR)/float32(math.Pow(2.0, float64(cycle)))
	case "exp_range":
		s.currentLR = s.baseLR + (s.currentLR-s.baseLR)*float32(math.Pow(float64(s.gamma), float64(step)))
	}

	if s.optimizer != nil {
		s.optimizer.SetLearningRate(s.currentLR)
	}

	return nil
}

// GetLR returns the current learning rate
func (s *CyclicLRScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer to update
func (s *CyclicLRScheduler) SetOptimizer(opt Optimizer) {
	s.optimizer = opt
}

// ChainedScheduler allows chaining multiple schedulers together
type ChainedScheduler struct {
	schedulers   []LRScheduler
	milestones   []int64 // Step counts where scheduler switches
	currentIndex int
	currentLR    float32
}

// NewChainedScheduler creates a new chained scheduler
func NewChainedScheduler(schedulers []LRScheduler, milestones []int64) *ChainedScheduler {
	if len(schedulers) != len(milestones)+1 {
		panic("Number of schedulers must be one more than number of milestones")
	}

	return &ChainedScheduler{
		schedulers:   schedulers,
		milestones:   milestones,
		currentIndex: 0,
		currentLR:    schedulers[0].GetLR(),
	}
}

// Step updates the learning rate using the appropriate scheduler
func (s *ChainedScheduler) Step(step int64) error {
	// Find the appropriate scheduler for this step
	newIndex := 0
	for i, milestone := range s.milestones {
		if step >= milestone {
			newIndex = i + 1
		} else {
			break
		}
	}

	if newIndex != s.currentIndex {
		s.currentIndex = newIndex
	}

	// Calculate relative step for the current scheduler
	relativeStep := step
	if s.currentIndex > 0 {
		relativeStep = step - s.milestones[s.currentIndex-1]
	}

	err := s.schedulers[s.currentIndex].Step(relativeStep)
	if err != nil {
		return err
	}

	s.currentLR = s.schedulers[s.currentIndex].GetLR()
	return nil
}

// GetLR returns the current learning rate
func (s *ChainedScheduler) GetLR() float32 {
	return s.currentLR
}

// SetOptimizer sets the optimizer for all schedulers
func (s *ChainedScheduler) SetOptimizer(opt Optimizer) {
	for _, scheduler := range s.schedulers {
		scheduler.SetOptimizer(opt)
	}
}

// Helper function to create a common warmup + cosine annealing scheduler
func NewWarmupCosineScheduler(initialLR, maxLR, minLR float32, warmupSteps, totalSteps int64) LRScheduler {
	// Cosine scheduler should run for the remaining steps after warmup
	cosineSteps := totalSteps - warmupSteps
	cosineScheduler := NewCosineAnnealingScheduler(maxLR, minLR, cosineSteps)
	warmupScheduler := NewWarmupScheduler(maxLR, warmupSteps, 0) // Linear warmup
	warmupScheduler.SetBaseScheduler(cosineScheduler)
	return warmupScheduler
}

// Helper function to create a common warmup + exponential decay scheduler
func NewWarmupExponentialScheduler(initialLR, maxLR, decayRate float32, warmupSteps, decaySteps int64) LRScheduler {
	expScheduler := NewExponentialDecayScheduler(maxLR, decayRate, decaySteps)
	warmupScheduler := NewWarmupScheduler(maxLR, warmupSteps, 0) // Linear warmup
	warmupScheduler.SetBaseScheduler(expScheduler)
	return warmupScheduler
}
