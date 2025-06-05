package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"
	"unsafe"

	"github.com/tsawler/gometal/tensor"
)

// MemoryEfficientAutodiff provides memory-efficient automatic differentiation with
// advanced memory management, gradient compression, and selective computation

// CheckpointingStrategy defines different checkpointing strategies
type CheckpointingStrategy int

const (
	NoCheckpointing CheckpointingStrategy = iota
	UniformCheckpointing
	AdaptiveCheckpointing
	MemoryAwareCheckpointing
)

// GradientCompressionMethod defines gradient compression methods
type GradientCompressionMethod int

const (
	NoCompression GradientCompressionMethod = iota
	TopKSparsification
	RandomSparsification
	Quantization
	ErrorFeedback
)

// MemoryEfficientConfig configures memory-efficient autodiff
type MemoryEfficientConfig struct {
	MaxMemoryUsage        int64
	CheckpointingStrategy CheckpointingStrategy
	CheckpointingRatio    float32 // Fraction of layers to checkpoint
	GradientCompression   GradientCompressionMethod
	CompressionRatio      float32 // Compression ratio (0.0-1.0)
	EnableGradientScaling bool
	EnableMixedPrecision  bool
	EnableInPlaceOps      bool
	EnableTensorFusion    bool
	MemoryBudget          int64 // Target memory budget
	SwapThreshold         int64 // Threshold for CPU-GPU swapping
}

// MemoryEfficientAutodiffEngine manages memory-efficient autodiff
type MemoryEfficientAutodiffEngine struct {
	config             *MemoryEfficientConfig
	checkpointManager  *CheckpointManager
	gradientCompressor *GradientCompressor
	memoryManager      *AutodiffMemoryManager
	swapManager        *CPUGPUSwapManager
	compressionStats   *CompressionStats
	mutex              sync.RWMutex
}

// CheckpointData stores checkpointed computation information
type CheckpointData struct {
	forwardFunction func() (*GradientTensor, error)
	inputTensors    []*GradientTensor
	savedTensors    []*tensor.Tensor
	metadata        map[string]interface{}
	memoryFootprint int64
	computationCost float32
	lastAccessTime  time.Time
}

// GradientCompressor handles gradient compression
type GradientCompressor struct {
	method            GradientCompressionMethod
	compressionRatio  float32
	errorFeedback     map[*GradientTensor]*tensor.Tensor
	compressionBuffer *CompressionBuffer
	quantizationBits  int
	mutex             sync.RWMutex
}

// CompressionBuffer manages compression workspace
type CompressionBuffer struct {
	workspace      *tensor.Tensor
	indices        []int32
	values         []float32
	scalingFactors []float32
	maxSize        int64
}

// AutodiffMemoryManager handles memory allocation and deallocation
type AutodiffMemoryManager struct {
	memoryPool        *GPUMemoryPool
	tensorCache       *TensorCache
	allocationTracker map[*tensor.Tensor]*AllocationInfo
	memoryBudget      int64
	currentUsage      int64
	gcThreshold       int64
	mutex             sync.RWMutex
}

// AllocationInfo tracks tensor allocation information
type AllocationInfo struct {
	size         int64
	allocatedAt  time.Time
	lastAccessed time.Time
	accessCount  int64
	priority     int // 0=highest, higher numbers = lower priority
	canEvict     bool
}

// CPUGPUSwapManager handles swapping between CPU and GPU memory
type CPUGPUSwapManager struct {
	swapThreshold  int64
	swappedTensors map[*tensor.Tensor]*SwapInfo
	swapBuffer     *SwapBuffer
	prefetchQueue  *PrefetchQueue
	swapStats      *SwapStats
	mutex          sync.RWMutex
}

// SwapInfo tracks swapped tensor information
type SwapInfo struct {
	cpuCopy        *tensor.Tensor
	originalGPUPtr unsafe.Pointer
	swappedAt      time.Time
	accessPattern  []time.Time
	priority       float32
}

// SwapBuffer manages the swap buffer
type SwapBuffer struct {
	buffer     []byte
	maxSize    int64
	currentPos int64
}

// PrefetchQueue manages prefetching of tensors from CPU to GPU
type PrefetchQueue struct {
	queue      []*tensor.Tensor
	priorities []float32
	maxSize    int
}

// SwapStats tracks swapping statistics
type SwapStats struct {
	TotalSwapOuts   int64
	TotalSwapIns    int64
	SwapOutBytes    int64
	SwapInBytes     int64
	PrefetchHits    int64
	PrefetchMisses  int64
	AverageSwapTime time.Duration
}

// CompressionStats tracks compression statistics
type CompressionStats struct {
	TotalCompressed   int64
	TotalDecompressed int64
	CompressionRatio  float32
	CompressionTime   time.Duration
	DecompressionTime time.Duration
	AccuracyLoss      float32
	mutex             sync.RWMutex
}

// Global memory-efficient autodiff engine
var globalMemoryEfficientEngine *MemoryEfficientAutodiffEngine

func init() {
	config := &MemoryEfficientConfig{
		MaxMemoryUsage:        2 * 1024 * 1024 * 1024, // 2GB
		CheckpointingStrategy: AdaptiveCheckpointing,
		CheckpointingRatio:    0.5,
		GradientCompression:   TopKSparsification,
		CompressionRatio:      0.01, // Keep top 1%
		EnableGradientScaling: true,
		EnableMixedPrecision:  false,
		EnableInPlaceOps:      true,
		EnableTensorFusion:    true,
		MemoryBudget:          1024 * 1024 * 1024, // 1GB
		SwapThreshold:         512 * 1024 * 1024,  // 512MB
	}
	globalMemoryEfficientEngine = NewMemoryEfficientAutodiffEngine(config)
}

// NewMemoryEfficientAutodiffEngine creates a new memory-efficient autodiff engine
func NewMemoryEfficientAutodiffEngine(config *MemoryEfficientConfig) *MemoryEfficientAutodiffEngine {
	engine := &MemoryEfficientAutodiffEngine{
		config:           config,
		compressionStats: &CompressionStats{},
	}

	// Initialize checkpoint manager
	engine.checkpointManager = &CheckpointManager{
		checkpoints:        make(map[*GradientTensor]*CheckpointData),
		strategy:           config.CheckpointingStrategy,
		checkpointingRatio: config.CheckpointingRatio,
		memoryBudget:       config.MemoryBudget,
	}

	// Initialize gradient compressor
	engine.gradientCompressor = &GradientCompressor{
		method:           config.GradientCompression,
		compressionRatio: config.CompressionRatio,
		errorFeedback:    make(map[*GradientTensor]*tensor.Tensor),
		quantizationBits: 8, // Default to 8-bit quantization
	}

	// Initialize compression buffer
	engine.gradientCompressor.compressionBuffer = &CompressionBuffer{
		maxSize: 100 * 1024 * 1024, // 100MB
	}

	// Initialize memory manager
	memoryPool, err := NewGPUMemoryPool(config.MaxMemoryUsage)
	if err != nil {
		// Log error but continue with nil memory pool
		// In production, you might want to handle this differently
		memoryPool = nil
	}
	engine.memoryManager = &AutodiffMemoryManager{
		memoryPool:        memoryPool,
		tensorCache:       NewTensorCache(1000),
		allocationTracker: make(map[*tensor.Tensor]*AllocationInfo),
		memoryBudget:      config.MemoryBudget,
		gcThreshold:       config.MemoryBudget / 2,
	}

	// Initialize swap manager
	engine.swapManager = &CPUGPUSwapManager{
		swapThreshold:  config.SwapThreshold,
		swappedTensors: make(map[*tensor.Tensor]*SwapInfo),
		swapBuffer: &SwapBuffer{
			maxSize: 1024 * 1024 * 1024, // 1GB swap buffer
		},
		prefetchQueue: &PrefetchQueue{
			maxSize: 100,
		},
		swapStats: &SwapStats{},
	}

	return engine
}

// EnableMemoryEfficientAutodiff enables memory-efficient autodiff globally
func EnableMemoryEfficientAutodiff(config *MemoryEfficientConfig) {
	globalMemoryEfficientEngine = NewMemoryEfficientAutodiffEngine(config)
}

// DisableMemoryEfficientAutodiff disables memory-efficient autodiff
func DisableMemoryEfficientAutodiff() {
	globalMemoryEfficientEngine = nil
}

// CheckpointedOperation creates a checkpointed operation
func CheckpointedOperation(forwardFn func([]*GradientTensor) (*GradientTensor, error), inputs []*GradientTensor) (*GradientTensor, error) {
	if globalMemoryEfficientEngine == nil {
		// Fall back to regular operation
		return forwardFn(inputs)
	}

	return globalMemoryEfficientEngine.createCheckpointedOperation(forwardFn, inputs)
}

// createCheckpointedOperation creates a checkpointed operation
func (engine *MemoryEfficientAutodiffEngine) createCheckpointedOperation(forwardFn func([]*GradientTensor) (*GradientTensor, error), inputs []*GradientTensor) (*GradientTensor, error) {
	// Decide whether to checkpoint based on strategy
	shouldCheckpoint := engine.shouldCheckpoint(inputs)

	if !shouldCheckpoint {
		return forwardFn(inputs)
	}

	// Create checkpoint
	checkpoint := &CheckpointData{
		forwardFunction: func() (*GradientTensor, error) {
			return forwardFn(inputs)
		},
		inputTensors:    inputs,
		memoryFootprint: engine.calculateMemoryFootprint(inputs),
		computationCost: engine.estimateComputationCost(inputs),
		lastAccessTime:  time.Now(),
	}

	// Run forward pass without saving intermediate tensors
	SetGradientMode(NoGrad)
	result, err := forwardFn(inputs)
	SetGradientMode(Grad)

	if err != nil {
		return nil, err
	}

	// Set up gradient function for recomputation
	result.GradFn = &GradientFunction{
		OpType: OpReshape, // Placeholder
		Inputs: inputs,
		BackwardFn: func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
			return engine.recomputeAndBackward(checkpoint, gradOutput)
		},
	}

	// Store checkpoint
	engine.checkpointManager.mutex.Lock()
	engine.checkpointManager.checkpoints[result] = checkpoint
	engine.checkpointManager.mutex.Unlock()

	return result, nil
}

// shouldCheckpoint determines if an operation should be checkpointed
func (engine *MemoryEfficientAutodiffEngine) shouldCheckpoint(inputs []*GradientTensor) bool {
	engine.checkpointManager.mutex.RLock()
	defer engine.checkpointManager.mutex.RUnlock()

	switch engine.checkpointManager.strategy {
	case NoCheckpointing:
		return false

	case UniformCheckpointing:
		// Checkpoint every N operations based on ratio
		return len(engine.checkpointManager.checkpoints)%int(1.0/engine.checkpointManager.checkpointingRatio) == 0

	case AdaptiveCheckpointing:
		// Checkpoint based on memory usage and computation cost
		memoryFootprint := engine.calculateMemoryFootprint(inputs)
		return engine.checkpointManager.currentMemoryUsage+memoryFootprint > engine.checkpointManager.memoryBudget

	case MemoryAwareCheckpointing:
		// Checkpoint when approaching memory limit
		currentUsage := engine.memoryManager.getCurrentUsage()
		return currentUsage > engine.config.MemoryBudget*8/10 // 80% threshold

	default:
		return false
	}
}

// calculateMemoryFootprint estimates the memory footprint of tensors
func (engine *MemoryEfficientAutodiffEngine) calculateMemoryFootprint(tensors []*GradientTensor) int64 {
	totalSize := int64(0)
	for _, gt := range tensors {
		if gt.Tensor != nil {
			totalSize += int64(len(gt.Tensor.Data) * 4) // float32 = 4 bytes
		}
		if gt.Gradient != nil {
			totalSize += int64(len(gt.Gradient.Data) * 4)
		}
	}
	return totalSize
}

// estimateComputationCost estimates the computational cost of operations
func (engine *MemoryEfficientAutodiffEngine) estimateComputationCost(tensors []*GradientTensor) float32 {
	totalOps := float32(0)
	for _, gt := range tensors {
		if gt.Tensor != nil {
			// Simple heuristic: cost proportional to tensor size
			totalOps += float32(len(gt.Tensor.Data))
		}
	}
	return totalOps
}

// recomputeAndBackward recomputes forward pass and performs backward pass
func (engine *MemoryEfficientAutodiffEngine) recomputeAndBackward(checkpoint *CheckpointData, gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// Recompute forward pass
	SetGradientMode(Grad)
	result, err := checkpoint.forwardFunction()
	if err != nil {
		return nil, fmt.Errorf("recomputation failed: %w", err)
	}

	// Perform backward pass
	err = result.BackwardWithGradient(gradOutput)
	if err != nil {
		return nil, fmt.Errorf("backward pass failed: %w", err)
	}

	// Collect input gradients
	gradients := make([]*tensor.Tensor, len(checkpoint.inputTensors))
	for i, input := range checkpoint.inputTensors {
		if input.Gradient != nil {
			gradients[i] = input.Gradient
		}
	}

	// Update checkpoint access time
	checkpoint.lastAccessTime = time.Now()

	return gradients, nil
}

// CompressGradients compresses gradients using the configured method
func CompressGradients(gradients []*GradientTensor) error {
	if globalMemoryEfficientEngine == nil {
		return nil // No compression if engine not enabled
	}

	return globalMemoryEfficientEngine.compressGradients(gradients)
}

// compressGradients compresses gradients
func (engine *MemoryEfficientAutodiffEngine) compressGradients(gradients []*GradientTensor) error {
	compressor := engine.gradientCompressor
	compressor.mutex.Lock()
	defer compressor.mutex.Unlock()

	for _, gt := range gradients {
		if gt.Gradient == nil {
			continue
		}

		start := time.Now()

		switch compressor.method {
		case TopKSparsification:
			err := engine.applyTopKSparsification(gt)
			if err != nil {
				return fmt.Errorf("TopK sparsification failed: %w", err)
			}

		case RandomSparsification:
			err := engine.applyRandomSparsification(gt)
			if err != nil {
				return fmt.Errorf("random sparsification failed: %w", err)
			}

		case Quantization:
			err := engine.applyQuantization(gt)
			if err != nil {
				return fmt.Errorf("quantization failed: %w", err)
			}

		case ErrorFeedback:
			err := engine.applyErrorFeedback(gt)
			if err != nil {
				return fmt.Errorf("error feedback failed: %w", err)
			}
		}

		elapsed := time.Since(start)
		engine.compressionStats.mutex.Lock()
		engine.compressionStats.TotalCompressed++
		engine.compressionStats.CompressionTime += elapsed
		engine.compressionStats.mutex.Unlock()
	}

	return nil
}

// applyTopKSparsification applies Top-K sparsification
func (engine *MemoryEfficientAutodiffEngine) applyTopKSparsification(gt *GradientTensor) error {
	if err := gt.Gradient.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve gradient: %w", err)
	}

	k := int(float32(len(gt.Gradient.Data)) * engine.gradientCompressor.compressionRatio)
	if k == 0 {
		k = 1
	}

	// Find top-k elements by magnitude
	type IndexValue struct {
		index int
		value float32
		abs   float32
	}

	values := make([]IndexValue, len(gt.Gradient.Data))
	for i, val := range gt.Gradient.Data {
		values[i] = IndexValue{
			index: i,
			value: val,
			abs:   abs(val),
		}
	}

	// Sort by absolute value (descending)
	sort.Slice(values, func(i, j int) bool {
		return values[i].abs > values[j].abs
	})

	// Zero out all but top-k elements
	for i := range gt.Gradient.Data {
		gt.Gradient.Data[i] = 0.0
	}

	for i := 0; i < k && i < len(values); i++ {
		gt.Gradient.Data[values[i].index] = values[i].value
	}

	return nil
}

// applyRandomSparsification applies random sparsification
func (engine *MemoryEfficientAutodiffEngine) applyRandomSparsification(gt *GradientTensor) error {
	if err := gt.Gradient.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve gradient: %w", err)
	}

	keepRatio := engine.gradientCompressor.compressionRatio

	for i := range gt.Gradient.Data {
		if rand.Float32() > keepRatio {
			gt.Gradient.Data[i] = 0.0
		} else {
			// Scale up remaining values to maintain expected magnitude
			gt.Gradient.Data[i] /= keepRatio
		}
	}

	return nil
}

// applyQuantization applies gradient quantization
func (engine *MemoryEfficientAutodiffEngine) applyQuantization(gt *GradientTensor) error {
	if err := gt.Gradient.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve gradient: %w", err)
	}

	bits := engine.gradientCompressor.quantizationBits
	levels := (1 << bits) - 1 // 2^bits - 1

	// Find min and max values
	minVal, maxVal := gt.Gradient.Data[0], gt.Gradient.Data[0]
	for _, val := range gt.Gradient.Data {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}

	// Quantize
	scale := (maxVal - minVal) / float32(levels)
	if scale == 0 {
		return nil // All values are the same
	}

	for i, val := range gt.Gradient.Data {
		// Quantize to integer level
		level := int((val - minVal) / scale)
		if level < 0 {
			level = 0
		}
		if level > levels {
			level = levels
		}

		// Dequantize back to float
		gt.Gradient.Data[i] = minVal + float32(level)*scale
	}

	return nil
}

// applyErrorFeedback applies error feedback compression
func (engine *MemoryEfficientAutodiffEngine) applyErrorFeedback(gt *GradientTensor) error {
	compressor := engine.gradientCompressor

	if err := gt.Gradient.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve gradient: %w", err)
	}

	// Get or create error feedback tensor
	errorTensor, exists := compressor.errorFeedback[gt]
	if !exists {
		errorData := make([]float32, len(gt.Gradient.Data))
		var err error
		errorTensor, err = tensor.NewTensor(gt.Gradient.Shape, errorData)
		if err != nil {
			return fmt.Errorf("failed to create error tensor: %w", err)
		}
		compressor.errorFeedback[gt] = errorTensor
	}

	// Add accumulated error to gradient
	for i := range gt.Gradient.Data {
		gt.Gradient.Data[i] += errorTensor.Data[i]
	}

	// Apply compression (Top-K for simplicity)
	originalGrad := make([]float32, len(gt.Gradient.Data))
	copy(originalGrad, gt.Gradient.Data)

	err := engine.applyTopKSparsification(gt)
	if err != nil {
		return err
	}

	// Compute and accumulate error
	for i := range gt.Gradient.Data {
		errorTensor.Data[i] = originalGrad[i] - gt.Gradient.Data[i]
	}

	return nil
}

// SwapTensorToCPU swaps a tensor from GPU to CPU memory
func SwapTensorToCPU(t *tensor.Tensor) error {
	if globalMemoryEfficientEngine == nil {
		return nil
	}

	return globalMemoryEfficientEngine.swapManager.swapToCPU(t)
}

// SwapTensorToGPU swaps a tensor from CPU to GPU memory
func SwapTensorToGPU(t *tensor.Tensor) error {
	if globalMemoryEfficientEngine == nil {
		return nil
	}

	return globalMemoryEfficientEngine.swapManager.swapToGPU(t)
}

// swapToCPU swaps a tensor from GPU to CPU
func (sm *CPUGPUSwapManager) swapToCPU(t *tensor.Tensor) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	if _, exists := sm.swappedTensors[t]; exists {
		return nil // Already swapped
	}

	// Create CPU copy
	if err := t.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve tensor to CPU: %w", err)
	}

	cpuData := make([]float32, len(t.Data))
	copy(cpuData, t.Data)

	cpuCopy, err := tensor.NewTensor(t.Shape, cpuData)
	if err != nil {
		return fmt.Errorf("failed to create CPU copy: %w", err)
	}

	// Store swap info
	swapInfo := &SwapInfo{
		cpuCopy:        cpuCopy,
		originalGPUPtr: t.GPUPtr(),
		swappedAt:      time.Now(),
		accessPattern:  []time.Time{time.Now()},
		priority:       1.0,
	}

	sm.swappedTensors[t] = swapInfo

	// Release GPU memory
	t.ReleaseGPU()

	sm.swapStats.TotalSwapOuts++
	sm.swapStats.SwapOutBytes += int64(len(t.Data) * 4)

	return nil
}

// swapToGPU swaps a tensor from CPU to GPU
func (sm *CPUGPUSwapManager) swapToGPU(t *tensor.Tensor) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	swapInfo, exists := sm.swappedTensors[t]
	if !exists {
		return nil // Not swapped
	}

	// Copy data back to original tensor
	copy(t.Data, swapInfo.cpuCopy.Data)

	// Move back to GPU
	if err := t.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move tensor back to GPU: %w", err)
	}

	// Update access pattern
	swapInfo.accessPattern = append(swapInfo.accessPattern, time.Now())

	// Clean up
	delete(sm.swappedTensors, t)

	sm.swapStats.TotalSwapIns++
	sm.swapStats.SwapInBytes += int64(len(t.Data) * 4)

	return nil
}

// OptimizeMemoryUsage performs global memory optimization
func OptimizeMemoryUsage() error {
	if globalMemoryEfficientEngine == nil {
		return nil
	}

	return globalMemoryEfficientEngine.optimizeMemoryUsage()
}

// optimizeMemoryUsage performs memory optimization
func (engine *MemoryEfficientAutodiffEngine) optimizeMemoryUsage() error {
	// 1. Garbage collect unused tensors
	err := engine.garbageCollectTensors()
	if err != nil {
		return fmt.Errorf("garbage collection failed: %w", err)
	}

	// 2. Compress gradients if enabled
	if engine.config.GradientCompression != NoCompression {
		err = engine.compressAllGradients()
		if err != nil {
			return fmt.Errorf("gradient compression failed: %w", err)
		}
	}

	// 3. Swap tensors if memory pressure is high
	currentUsage := engine.memoryManager.getCurrentUsage()
	if currentUsage > engine.config.SwapThreshold {
		err = engine.performSwapping()
		if err != nil {
			return fmt.Errorf("swapping failed: %w", err)
		}
	}

	// 4. Clear old checkpoints
	err = engine.cleanupCheckpoints()
	if err != nil {
		return fmt.Errorf("checkpoint cleanup failed: %w", err)
	}

	return nil
}

// garbageCollectTensors performs garbage collection of unused tensors
func (engine *MemoryEfficientAutodiffEngine) garbageCollectTensors() error {
	manager := engine.memoryManager
	manager.mutex.Lock()
	defer manager.mutex.Unlock()

	now := time.Now()
	gcThreshold := 5 * time.Minute // GC tensors not accessed for 5 minutes

	var toRelease []*tensor.Tensor

	for t, info := range manager.allocationTracker {
		if info.canEvict && now.Sub(info.lastAccessed) > gcThreshold {
			toRelease = append(toRelease, t)
		}
	}

	for _, t := range toRelease {
		t.ReleaseGPU()
		delete(manager.allocationTracker, t)
		manager.currentUsage -= manager.allocationTracker[t].size
	}

	return nil
}

// compressAllGradients compresses all gradients in the computation graph
func (engine *MemoryEfficientAutodiffEngine) compressAllGradients() error {
	var gradients []*GradientTensor

	for _, node := range globalGraph.nodes {
		if node.Gradient != nil {
			gradients = append(gradients, node)
		}
	}

	return engine.compressGradients(gradients)
}

// performSwapping performs intelligent swapping based on access patterns
func (engine *MemoryEfficientAutodiffEngine) performSwapping() error {
	manager := engine.memoryManager
	swapManager := engine.swapManager

	manager.mutex.RLock()

	// Find candidates for swapping (least recently used, low priority)
	type SwapCandidate struct {
		tensor   *tensor.Tensor
		priority float32
		size     int64
	}

	var candidates []SwapCandidate
	now := time.Now()

	for t, info := range manager.allocationTracker {
		if info.canEvict {
			timeSinceAccess := now.Sub(info.lastAccessed)
			priority := float32(timeSinceAccess.Minutes()) / float32(info.accessCount+1)

			candidates = append(candidates, SwapCandidate{
				tensor:   t,
				priority: priority,
				size:     info.size,
			})
		}
	}

	manager.mutex.RUnlock()

	// Sort by priority (highest priority = best candidate for swapping)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].priority > candidates[j].priority
	})

	// Swap until we're under the threshold
	currentUsage := manager.getCurrentUsage()
	targetReduction := currentUsage - engine.config.SwapThreshold
	swapped := int64(0)

	for _, candidate := range candidates {
		if swapped >= targetReduction {
			break
		}

		err := swapManager.swapToCPU(candidate.tensor)
		if err != nil {
			continue // Skip on error
		}

		swapped += candidate.size
	}

	return nil
}

// cleanupCheckpoints removes old checkpoints
func (engine *MemoryEfficientAutodiffEngine) cleanupCheckpoints() error {
	cm := engine.checkpointManager
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	now := time.Now()
	cleanupThreshold := 10 * time.Minute // Remove checkpoints older than 10 minutes

	var toDelete []*GradientTensor

	for gt, checkpoint := range cm.checkpoints {
		if now.Sub(checkpoint.lastAccessTime) > cleanupThreshold {
			toDelete = append(toDelete, gt)
		}
	}
	for _, gt := range toDelete {
		delete(cm.checkpoints, gt)
	}

	return nil
}

// getCurrentUsage returns the current memory usage
func (manager *AutodiffMemoryManager) getCurrentUsage() int64 {
	manager.mutex.RLock()
	defer manager.mutex.RUnlock()
	return manager.currentUsage
}

// TrackTensorAllocation tracks a tensor allocation
func TrackTensorAllocation(t *tensor.Tensor, canEvict bool) {
	if globalMemoryEfficientEngine == nil {
		return
	}

	globalMemoryEfficientEngine.memoryManager.trackAllocation(t, canEvict)
}

// trackAllocation tracks a tensor allocation
func (manager *AutodiffMemoryManager) trackAllocation(t *tensor.Tensor, canEvict bool) {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()

	size := int64(len(t.Data) * 4) // float32 = 4 bytes

	info := &AllocationInfo{
		size:         size,
		allocatedAt:  time.Now(),
		lastAccessed: time.Now(),
		accessCount:  1,
		priority:     1, // Normal priority
		canEvict:     canEvict,
	}

	manager.allocationTracker[t] = info
	manager.currentUsage += size
}

// UpdateTensorAccess updates tensor access information
func UpdateTensorAccess(t *tensor.Tensor) {
	if globalMemoryEfficientEngine == nil {
		return
	}

	globalMemoryEfficientEngine.memoryManager.updateAccess(t)
}

// updateAccess updates tensor access information
func (manager *AutodiffMemoryManager) updateAccess(t *tensor.Tensor) {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()

	if info, exists := manager.allocationTracker[t]; exists {
		info.lastAccessed = time.Now()
		info.accessCount++
	}
}

// GetMemoryEfficientStats returns comprehensive statistics
func GetMemoryEfficientStats() map[string]interface{} {
	if globalMemoryEfficientEngine == nil {
		return map[string]interface{}{"enabled": false}
	}

	engine := globalMemoryEfficientEngine
	stats := make(map[string]interface{})

	stats["enabled"] = true
	stats["config"] = engine.config

	// Checkpoint statistics
	engine.checkpointManager.mutex.RLock()
	stats["checkpoint_count"] = len(engine.checkpointManager.checkpoints)
	stats["checkpoint_memory_usage"] = engine.checkpointManager.currentMemoryUsage
	engine.checkpointManager.mutex.RUnlock()

	// Compression statistics
	engine.compressionStats.mutex.RLock()
	stats["compression_stats"] = map[string]interface{}{
		"total_compressed":   engine.compressionStats.TotalCompressed,
		"total_decompressed": engine.compressionStats.TotalDecompressed,
		"compression_ratio":  engine.compressionStats.CompressionRatio,
		"compression_time":   engine.compressionStats.CompressionTime,
		"decompression_time": engine.compressionStats.DecompressionTime,
		"accuracy_loss":      engine.compressionStats.AccuracyLoss,
	}
	engine.compressionStats.mutex.RUnlock()

	// Memory management statistics
	engine.memoryManager.mutex.RLock()
	stats["memory_stats"] = map[string]interface{}{
		"current_usage":   engine.memoryManager.currentUsage,
		"memory_budget":   engine.memoryManager.memoryBudget,
		"tracked_tensors": len(engine.memoryManager.allocationTracker),
		"gc_threshold":    engine.memoryManager.gcThreshold,
	}
	engine.memoryManager.mutex.RUnlock()

	// Swap statistics
	engine.swapManager.mutex.RLock()
	stats["swap_stats"] = map[string]interface{}{
		"swapped_tensors":   len(engine.swapManager.swappedTensors),
		"total_swap_outs":   engine.swapManager.swapStats.TotalSwapOuts,
		"total_swap_ins":    engine.swapManager.swapStats.TotalSwapIns,
		"swap_out_bytes":    engine.swapManager.swapStats.SwapOutBytes,
		"swap_in_bytes":     engine.swapManager.swapStats.SwapInBytes,
		"prefetch_hits":     engine.swapManager.swapStats.PrefetchHits,
		"prefetch_misses":   engine.swapManager.swapStats.PrefetchMisses,
		"average_swap_time": engine.swapManager.swapStats.AverageSwapTime,
	}
	engine.swapManager.mutex.RUnlock()

	return stats
}

// MemoryEfficientGradContext provides a context for memory-efficient gradient computation
type MemoryEfficientGradContext struct {
	originalConfig     *MemoryEfficientConfig
	temporaryConfig    *MemoryEfficientConfig
	compressionEnabled bool
	swappingEnabled    bool
	checkpointingLevel int
}

// WithMemoryEfficientContext executes a function with a specific memory-efficient context
func WithMemoryEfficientContext(config *MemoryEfficientConfig, fn func() error) error {
	if globalMemoryEfficientEngine == nil {
		return fn() // Execute without memory efficiency if not enabled
	}

	// Save original config
	originalConfig := globalMemoryEfficientEngine.config

	// Apply temporary config
	globalMemoryEfficientEngine.config = config

	defer func() {
		// Restore original config
		globalMemoryEfficientEngine.config = originalConfig
	}()

	return fn()
}

// MixedPrecisionContext provides mixed precision training context
type MixedPrecisionContext struct {
	enabled       bool
	lossScaling   float32
	scalingFactor float32
	skipSteps     int
	currentStep   int
}

// WithMixedPrecision executes a function with mixed precision enabled
func WithMixedPrecision(lossScaling float32, fn func() error) error {
	// This would implement mixed precision training
	// For now, just execute the function normally
	return fn()
}

// AdaptiveMemoryManager automatically adjusts memory settings based on usage patterns
type AdaptiveMemoryManager struct {
	memoryUsageHistory []int64
	performanceHistory []float32
	adaptationEnabled  bool
	adaptationInterval time.Duration
	lastAdaptation     time.Time
	learningRate       float32
	mutex              sync.RWMutex
}

// NewAdaptiveMemoryManager creates a new adaptive memory manager
func NewAdaptiveMemoryManager() *AdaptiveMemoryManager {
	return &AdaptiveMemoryManager{
		memoryUsageHistory: make([]int64, 0, 100),
		performanceHistory: make([]float32, 0, 100),
		adaptationEnabled:  true,
		adaptationInterval: 1 * time.Minute,
		learningRate:       0.1,
	}
}

// AdaptMemorySettings adapts memory settings based on usage patterns
func (amm *AdaptiveMemoryManager) AdaptMemorySettings() {
	amm.mutex.Lock()
	defer amm.mutex.Unlock()

	if !amm.adaptationEnabled || time.Since(amm.lastAdaptation) < amm.adaptationInterval {
		return
	}

	if len(amm.memoryUsageHistory) < 10 {
		return // Need more data points
	}

	// Analyze memory usage trend
	recentUsage := amm.memoryUsageHistory[len(amm.memoryUsageHistory)-10:]
	avgUsage := amm.calculateAverage(recentUsage)

	// Analyze performance trend
	recentPerformance := amm.performanceHistory[len(amm.performanceHistory)-10:]
	avgPerformance := amm.calculateAverageFloat32(recentPerformance)

	// Adapt settings based on patterns
	if globalMemoryEfficientEngine != nil {
		config := globalMemoryEfficientEngine.config

		// Adjust compression ratio based on memory pressure
		if avgUsage > config.MemoryBudget*8/10 { // Above 80%
			// Increase compression
			newRatio := config.CompressionRatio * (1 + amm.learningRate)
			if newRatio < 0.99 {
				config.CompressionRatio = newRatio
			}
		} else if avgUsage < config.MemoryBudget*3/10 { // Below 30%
			// Decrease compression for better accuracy
			newRatio := config.CompressionRatio * (1 - amm.learningRate)
			if newRatio > 0.001 {
				config.CompressionRatio = newRatio
			}
		}

		// Adjust checkpointing ratio based on performance
		if avgPerformance < 0.8 { // Poor performance
			// Reduce checkpointing
			newRatio := config.CheckpointingRatio * (1 - amm.learningRate)
			if newRatio > 0.1 {
				config.CheckpointingRatio = newRatio
			}
		}

		// Adjust swap threshold based on swap activity
		swapStats := globalMemoryEfficientEngine.swapManager.swapStats
		if swapStats.TotalSwapOuts > 100 { // High swap activity
			// Increase swap threshold to reduce swapping
			config.SwapThreshold = int64(float32(config.SwapThreshold) * (1 + amm.learningRate))
		}
	}

	amm.lastAdaptation = time.Now()
}

// RecordMemoryUsage records current memory usage
func (amm *AdaptiveMemoryManager) RecordMemoryUsage(usage int64) {
	amm.mutex.Lock()
	defer amm.mutex.Unlock()

	amm.memoryUsageHistory = append(amm.memoryUsageHistory, usage)

	// Keep only recent history
	if len(amm.memoryUsageHistory) > 100 {
		amm.memoryUsageHistory = amm.memoryUsageHistory[1:]
	}
}

// RecordPerformance records performance metric
func (amm *AdaptiveMemoryManager) RecordPerformance(performance float32) {
	amm.mutex.Lock()
	defer amm.mutex.Unlock()

	amm.performanceHistory = append(amm.performanceHistory, performance)

	// Keep only recent history
	if len(amm.performanceHistory) > 100 {
		amm.performanceHistory = amm.performanceHistory[1:]
	}
}

// calculateAverage calculates the average of int64 slice
func (amm *AdaptiveMemoryManager) calculateAverage(data []int64) int64 {
	if len(data) == 0 {
		return 0
	}

	sum := int64(0)
	for _, val := range data {
		sum += val
	}

	return sum / int64(len(data))
}

// calculateAverageFloat32 calculates the average of float32 slice
func (amm *AdaptiveMemoryManager) calculateAverageFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}

	sum := float32(0)
	for _, val := range data {
		sum += val
	}

	return sum / float32(len(data))
}

// Utility functions for memory-efficient operations

// InPlaceOperation performs an operation in-place to save memory
func InPlaceOperation(tensor *GradientTensor, operation func(*tensor.Tensor) error) error {
	if globalMemoryEfficientEngine == nil || !globalMemoryEfficientEngine.config.EnableInPlaceOps {
		// Fall back to regular operation
		return operation(tensor.Tensor)
	}

	// Perform in-place operation
	return operation(tensor.Tensor)
}

// TensorFusion fuses multiple small tensors into larger ones for efficiency
func TensorFusion(tensors []*GradientTensor) ([]*GradientTensor, error) {
	if globalMemoryEfficientEngine == nil || !globalMemoryEfficientEngine.config.EnableTensorFusion {
		return tensors, nil // No fusion
	}

	// Group tensors by similar shapes for fusion
	groups := make(map[string][]*GradientTensor)

	for _, t := range tensors {
		key := fmt.Sprintf("%v", t.Tensor.Shape)
		groups[key] = append(groups[key], t)
	}

	var fusedTensors []*GradientTensor

	for _, group := range groups {
		if len(group) <= 1 {
			fusedTensors = append(fusedTensors, group...)
			continue
		}

		// Fuse tensors in this group
		fused, err := fuseTensorGroup(group)
		if err != nil {
			// Fall back to original tensors on error
			fusedTensors = append(fusedTensors, group...)
		} else {
			fusedTensors = append(fusedTensors, fused)
		}
	}

	return fusedTensors, nil
}

// fuseTensorGroup fuses a group of tensors with similar shapes
func fuseTensorGroup(group []*GradientTensor) (*GradientTensor, error) {
	if len(group) == 0 {
		return nil, fmt.Errorf("empty tensor group")
	}

	// Calculate total size for fused tensor
	totalSize := 0
	for _, t := range group {
		totalSize += len(t.Tensor.Data)
	}

	// Create fused tensor
	fusedData := make([]float32, totalSize)
	offset := 0

	for _, t := range group {
		if err := t.Tensor.RetrieveCPU(); err != nil {
			return nil, fmt.Errorf("failed to retrieve tensor data: %w", err)
		}

		copy(fusedData[offset:offset+len(t.Tensor.Data)], t.Tensor.Data)
		offset += len(t.Tensor.Data)
	}

	fusedTensor, err := tensor.NewTensor([]int{totalSize}, fusedData)
	if err != nil {
		return nil, fmt.Errorf("failed to create fused tensor: %w", err)
	}

	// Create gradient tensor
	requiresGrad := false
	for _, t := range group {
		if t.RequiresGrad {
			requiresGrad = true
			break
		}
	}

	fusedGT := &GradientTensor{
		Tensor:       fusedTensor,
		RequiresGrad: requiresGrad,
		IsLeaf:       false,
	}

	// Set up unfusion for backward pass
	if requiresGrad && globalGraph.gradMode == Grad {
		fusedGT.GradFn = &GradientFunction{
			OpType: OpReshape,
			Inputs: group,
			BackwardFn: func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
				return unfuseGradients(gradOutput, group)
			},
		}
		globalGraph.addNode(fusedGT)
	}

	return fusedGT, nil
}

// unfuseGradients splits fused gradients back to individual tensors
func unfuseGradients(fusedGrad *tensor.Tensor, originalGroup []*GradientTensor) ([]*tensor.Tensor, error) {
	if err := fusedGrad.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve fused gradient: %w", err)
	}

	gradients := make([]*tensor.Tensor, len(originalGroup))
	offset := 0

	for i, t := range originalGroup {
		size := len(t.Tensor.Data)
		gradData := make([]float32, size)
		copy(gradData, fusedGrad.Data[offset:offset+size])

		grad, err := tensor.NewTensor(t.Tensor.Shape, gradData)
		if err != nil {
			return nil, fmt.Errorf("failed to create unfused gradient: %w", err)
		}

		gradients[i] = grad
		offset += size
	}

	return gradients, nil
}

// Helper functions for random number generation (simple implementation)
type simpleRand struct {
	seed uint64
}

func (r *simpleRand) Float32() float32 {
	r.seed = r.seed*1103515245 + 12345
	return float32(r.seed&0x7fffffff) / 0x7fffffff
}

// var rand = &simpleRand{seed: uint64(time.Now().UnixNano())}

// PrintMemoryEfficientStats prints detailed memory efficiency statistics
func PrintMemoryEfficientStats() {
	stats := GetMemoryEfficientStats()

	fmt.Println("=== Memory-Efficient Autodiff Statistics ===")

	if enabled, ok := stats["enabled"].(bool); !ok || !enabled {
		fmt.Println("Memory-efficient autodiff is disabled")
		return
	}

	fmt.Println("Memory-efficient autodiff is enabled")

	if checkpointCount, ok := stats["checkpoint_count"].(int); ok {
		fmt.Printf("Active checkpoints: %d\n", checkpointCount)
	}

	if compressionStats, ok := stats["compression_stats"].(map[string]interface{}); ok {
		fmt.Println("\nCompression Statistics:")
		if totalCompressed, ok := compressionStats["total_compressed"].(int64); ok {
			fmt.Printf("  Total compressed: %d\n", totalCompressed)
		}
		if compressionRatio, ok := compressionStats["compression_ratio"].(float32); ok {
			fmt.Printf("  Compression ratio: %.4f\n", compressionRatio)
		}
		if compressionTime, ok := compressionStats["compression_time"].(time.Duration); ok {
			fmt.Printf("  Compression time: %v\n", compressionTime)
		}
	}

	if memoryStats, ok := stats["memory_stats"].(map[string]interface{}); ok {
		fmt.Println("\nMemory Statistics:")
		if currentUsage, ok := memoryStats["current_usage"].(int64); ok {
			fmt.Printf("  Current usage: %s\n", formatBytes(currentUsage))
		}
		if memoryBudget, ok := memoryStats["memory_budget"].(int64); ok {
			fmt.Printf("  Memory budget: %s\n", formatBytes(memoryBudget))
		}
		if trackedTensors, ok := memoryStats["tracked_tensors"].(int); ok {
			fmt.Printf("  Tracked tensors: %d\n", trackedTensors)
		}
	}

	if swapStats, ok := stats["swap_stats"].(map[string]interface{}); ok {
		fmt.Println("\nSwap Statistics:")
		if swappedTensors, ok := swapStats["swapped_tensors"].(int); ok {
			fmt.Printf("  Currently swapped: %d tensors\n", swappedTensors)
		}
		if totalSwapOuts, ok := swapStats["total_swap_outs"].(int64); ok {
			fmt.Printf("  Total swap outs: %d\n", totalSwapOuts)
		}
		if totalSwapIns, ok := swapStats["total_swap_ins"].(int64); ok {
			fmt.Printf("  Total swap ins: %d\n", totalSwapIns)
		}
	}
}
