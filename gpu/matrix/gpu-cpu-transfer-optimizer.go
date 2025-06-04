package matrix

import (
	"fmt"
	"sync"
	"time"

	"github.com/tsawler/go-nngpu/tensor"
)

// Phase 8C: GPU-CPU Transfer Optimizer
// This component optimizes data transfers between CPU and GPU memory

// GPUCPUTransferOptimizer manages efficient data transfers between CPU and GPU
type GPUCPUTransferOptimizer struct {
	tensorCache      map[*tensor.Tensor]*TransferCacheEntry
	transferStats    TransferStatistics
	cachingEnabled   bool
	asyncTransfers   bool
	batchSize        int
	mutex            sync.RWMutex
	bandwidthMonitor *MemoryBandwidthMonitor
}

// TransferCacheEntry tracks tensor transfer state
type TransferCacheEntry struct {
	Tensor         *tensor.Tensor
	GPUValid       bool
	CPUValid       bool
	LastGPUAccess  time.Time
	LastCPUAccess  time.Time
	TransferCount  int
	IsPinned       bool // Whether tensor uses pinned memory
	LastOperation  string
}

// TransferStatistics tracks transfer performance metrics
type TransferStatistics struct {
	TotalTransfers      int64
	TotalBytesTransferred int64
	CPUToGPUTransfers   int64
	GPUToCPUTransfers   int64
	CacheHits           int64
	CacheMisses         int64
	AverageTransferTime time.Duration
	PinnedMemoryUsed    int64
}

// NewGPUCPUTransferOptimizer creates a new transfer optimizer
func NewGPUCPUTransferOptimizer() *GPUCPUTransferOptimizer {
	return &GPUCPUTransferOptimizer{
		tensorCache:      make(map[*tensor.Tensor]*TransferCacheEntry),
		cachingEnabled:   true,
		asyncTransfers:   true,
		batchSize:        16,
		bandwidthMonitor: GetGlobalBandwidthMonitor(),
	}
}

// OptimizeTransfer optimizes a tensor transfer between CPU and GPU
func (opt *GPUCPUTransferOptimizer) OptimizeTransfer(t *tensor.Tensor, toGPU bool, operation string) error {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()

	// Get or create cache entry
	entry, exists := opt.tensorCache[t]
	if !exists {
		entry = &TransferCacheEntry{
			Tensor:        t,
			GPUValid:      false,
			CPUValid:      true,
			LastOperation: operation,
		}
		opt.tensorCache[t] = entry
	}

	// Check if transfer is needed
	if toGPU && entry.GPUValid {
		opt.transferStats.CacheHits++
		entry.LastGPUAccess = time.Now()
		return nil // Already on GPU
	} else if !toGPU && entry.CPUValid {
		opt.transferStats.CacheHits++
		entry.LastCPUAccess = time.Now()
		return nil // Already on CPU
	}

	opt.transferStats.CacheMisses++

	// Perform the transfer
	start := time.Now()
	var err error
	
	if toGPU {
		err = opt.transferToGPU(t, entry)
		if err == nil {
			entry.GPUValid = true
			entry.LastGPUAccess = time.Now()
			opt.transferStats.CPUToGPUTransfers++
		}
	} else {
		err = opt.transferToCPU(t, entry)
		if err == nil {
			entry.CPUValid = true
			entry.LastCPUAccess = time.Now()
			opt.transferStats.GPUToCPUTransfers++
		}
	}

	if err != nil {
		return err
	}

	// Update statistics
	transferTime := time.Since(start)
	opt.updateTransferStats(t, transferTime)
	entry.TransferCount++
	entry.LastOperation = operation

	return nil
}

// transferToGPU performs optimized CPU to GPU transfer
func (opt *GPUCPUTransferOptimizer) transferToGPU(t *tensor.Tensor, entry *TransferCacheEntry) error {
	// In a real implementation, this would:
	// 1. Use pinned memory for faster transfers
	// 2. Use async transfers when possible
	// 3. Batch small transfers together
	
	// For now, use the tensor's built-in GPU transfer
	err := t.EnsureGPU()
	if err != nil {
		return fmt.Errorf("failed to transfer tensor to GPU: %w", err)
	}

	return nil
}

// transferToCPU performs optimized GPU to CPU transfer
func (opt *GPUCPUTransferOptimizer) transferToCPU(t *tensor.Tensor, entry *TransferCacheEntry) error {
	// In a real implementation, this would optimize the transfer
	// For now, mark as CPU valid (tensor data is already in CPU memory)
	entry.CPUValid = true
	return nil
}

// updateTransferStats updates transfer statistics
func (opt *GPUCPUTransferOptimizer) updateTransferStats(t *tensor.Tensor, transferTime time.Duration) {
	tensorSize := int64(len(t.Data) * 4) // 4 bytes per float32
	
	opt.transferStats.TotalTransfers++
	opt.transferStats.TotalBytesTransferred += tensorSize
	
	// Update average transfer time
	if opt.transferStats.AverageTransferTime == 0 {
		opt.transferStats.AverageTransferTime = transferTime
	} else {
		// Running average
		opt.transferStats.AverageTransferTime = (opt.transferStats.AverageTransferTime + transferTime) / 2
	}

	// Record bandwidth if monitor is available
	if opt.bandwidthMonitor != nil {
		opt.bandwidthMonitor.RecordTransfer(tensorSize, transferTime)
	}
}

// ShouldTransferToGPU determines if a tensor should be transferred to GPU
func (opt *GPUCPUTransferOptimizer) ShouldTransferToGPU(t *tensor.Tensor) bool {
	opt.mutex.RLock()
	defer opt.mutex.RUnlock()

	entry, exists := opt.tensorCache[t]
	if !exists {
		// New tensor, should transfer if it will be used on GPU
		return true
	}

	// Already on GPU and valid
	if entry.GPUValid {
		return false
	}

	// Transfer if not on GPU
	return true
}

// MarkGPUValid marks a tensor as having valid data on GPU
func (opt *GPUCPUTransferOptimizer) MarkGPUValid(t *tensor.Tensor, operation string) {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()

	entry, exists := opt.tensorCache[t]
	if !exists {
		entry = &TransferCacheEntry{
			Tensor:        t,
			LastOperation: operation,
		}
		opt.tensorCache[t] = entry
	}

	entry.GPUValid = true
	entry.LastGPUAccess = time.Now()
	entry.LastOperation = operation
}

// MarkCPUValid marks a tensor as having valid data on CPU
func (opt *GPUCPUTransferOptimizer) MarkCPUValid(t *tensor.Tensor, operation string) {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()

	entry, exists := opt.tensorCache[t]
	if !exists {
		entry = &TransferCacheEntry{
			Tensor:        t,
			LastOperation: operation,
		}
		opt.tensorCache[t] = entry
	}

	entry.CPUValid = true
	entry.LastCPUAccess = time.Now()
	entry.LastOperation = operation
}

// InvalidateGPU marks GPU data as invalid (e.g., after CPU modification)
func (opt *GPUCPUTransferOptimizer) InvalidateGPU(t *tensor.Tensor) {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()

	if entry, exists := opt.tensorCache[t]; exists {
		entry.GPUValid = false
	}
}

// InvalidateCPU marks CPU data as invalid (e.g., after GPU computation)
func (opt *GPUCPUTransferOptimizer) InvalidateCPU(t *tensor.Tensor) {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()

	if entry, exists := opt.tensorCache[t]; exists {
		entry.CPUValid = false
	}
}

// GetTransferStats returns current transfer statistics
func (opt *GPUCPUTransferOptimizer) GetTransferStats() TransferStatistics {
	opt.mutex.RLock()
	defer opt.mutex.RUnlock()
	
	return opt.transferStats
}

// ClearCache clears the tensor transfer cache
func (opt *GPUCPUTransferOptimizer) ClearCache() {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()
	
	opt.tensorCache = make(map[*tensor.Tensor]*TransferCacheEntry)
}

// EnableCaching enables or disables transfer caching
func (opt *GPUCPUTransferOptimizer) EnableCaching(enable bool) {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()
	
	opt.cachingEnabled = enable
}

// SetBatchSize sets the batch size for batched transfers
func (opt *GPUCPUTransferOptimizer) SetBatchSize(size int) {
	opt.mutex.Lock()
	defer opt.mutex.Unlock()
	
	if size > 0 {
		opt.batchSize = size
	}
}

// GetCacheInfo returns information about cached tensors
func (opt *GPUCPUTransferOptimizer) GetCacheInfo() map[string]interface{} {
	opt.mutex.RLock()
	defer opt.mutex.RUnlock()

	info := map[string]interface{}{
		"cache_size":       len(opt.tensorCache),
		"caching_enabled":  opt.cachingEnabled,
		"async_transfers":  opt.asyncTransfers,
		"batch_size":       opt.batchSize,
		"total_transfers":  opt.transferStats.TotalTransfers,
		"cache_hit_rate":   float64(0),
	}

	totalAccesses := opt.transferStats.CacheHits + opt.transferStats.CacheMisses
	if totalAccesses > 0 {
		info["cache_hit_rate"] = float64(opt.transferStats.CacheHits) / float64(totalAccesses) * 100
	}

	return info
}

// BatchedTransfer represents a batch of tensor transfers
type BatchedTransfer struct {
	Tensors   []*tensor.Tensor
	ToGPU     bool
	Operation string
}

// OptimizeBatchTransfer optimizes a batch of tensor transfers
func (opt *GPUCPUTransferOptimizer) OptimizeBatchTransfer(batch *BatchedTransfer) error {
	// In a real implementation, this would:
	// 1. Sort tensors by size for optimal packing
	// 2. Use a single large transfer when possible
	// 3. Pipeline transfers with computation
	
	for _, t := range batch.Tensors {
		if err := opt.OptimizeTransfer(t, batch.ToGPU, batch.Operation); err != nil {
			return err
		}
	}
	
	return nil
}

// Global transfer optimizer instance
var globalTransferOptimizer *GPUCPUTransferOptimizer
var transferOptimizerOnce sync.Once

// GetGlobalTransferOptimizer returns the global transfer optimizer instance
func GetGlobalTransferOptimizer() *GPUCPUTransferOptimizer {
	transferOptimizerOnce.Do(func() {
		globalTransferOptimizer = NewGPUCPUTransferOptimizer()
	})
	return globalTransferOptimizer
}

// OptimizeTensorTransfer optimizes a single tensor transfer using the global optimizer
func OptimizeTensorTransfer(t *tensor.Tensor, toGPU bool, operation string) error {
	optimizer := GetGlobalTransferOptimizer()
	return optimizer.OptimizeTransfer(t, toGPU, operation)
}

// OptimizeBatchedTransfer optimizes a batch of tensor transfers using the global optimizer
func OptimizeBatchedTransfer(tensors []*tensor.Tensor, toGPU bool, operation string) error {
	optimizer := GetGlobalTransferOptimizer()
	batch := &BatchedTransfer{
		Tensors:   tensors,
		ToGPU:     toGPU,
		Operation: operation,
	}
	return optimizer.OptimizeBatchTransfer(batch)
}