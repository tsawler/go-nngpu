package matrix

import (
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/tsawler/go-nngpu/tensor"
)

// Phase 8C: Buffer Reuse System for Intermediate Tensors
// This system minimizes memory allocations by reusing GPU buffers for temporary computations

// BufferReuseManager manages reusable buffers for intermediate tensor operations
type BufferReuseManager struct {
	bufferPool     map[string][]*ReusableBuffer // Pool of available buffers by size category
	activeBuffers  map[*tensor.Tensor]*ReusableBuffer // Currently active buffers
	bufferStats    map[string]*BufferStats      // Statistics for each buffer category
	mutex          sync.RWMutex
	maxPoolSize    int                          // Maximum number of buffers per category
	cleanupTicker  *time.Ticker                 // Periodic cleanup
	memoryPool     *GPUMemoryPool               // Underlying memory pool
}

// ReusableBuffer represents a GPU buffer that can be reused for multiple operations
type ReusableBuffer struct {
	GPUPtr        unsafe.Pointer    // GPU buffer pointer
	Size          int               // Buffer size in bytes
	Shape         []int             // Current tensor shape using this buffer
	LastUsed      time.Time         // Last usage time for cleanup
	UsageCount    int               // Number of times reused
	Category      string            // Buffer size category (small, medium, large, etc.)
	IsActive      bool              // Whether buffer is currently in use
	OriginalTensor *tensor.Tensor   // Original tensor if this buffer was allocated for one
}

// BufferStats tracks usage statistics for buffer categories
type BufferStats struct {
	TotalAllocations int64         // Total allocations in this category
	TotalReuses      int64         // Total reuses
	TotalHits        int64         // Cache hits
	TotalMisses      int64         // Cache misses
	AverageLifetime  time.Duration // Average buffer lifetime
	PeakUsage        int           // Peak number of buffers in use
	CurrentUsage     int           // Current number of buffers in use
}

// NewBufferReuseManager creates a new buffer reuse manager
func NewBufferReuseManager(memoryPool *GPUMemoryPool) *BufferReuseManager {
	manager := &BufferReuseManager{
		bufferPool:    make(map[string][]*ReusableBuffer),
		activeBuffers: make(map[*tensor.Tensor]*ReusableBuffer),
		bufferStats:   make(map[string]*BufferStats),
		maxPoolSize:   32, // Maximum 32 buffers per category
		memoryPool:    memoryPool,
	}
	
	// Start periodic cleanup
	manager.cleanupTicker = time.NewTicker(30 * time.Second)
	go manager.periodicCleanup()
	
	return manager
}

// GetBuffer gets a reusable buffer for a tensor operation
func (brm *BufferReuseManager) GetBuffer(shape []int, operation string) (*ReusableBuffer, error) {
	brm.mutex.Lock()
	defer brm.mutex.Unlock()
	
	// Calculate buffer size and category
	size := calculateBufferSize(shape)
	category := categorizeBufferSize(size)
	
	// Initialize stats if needed
	if _, exists := brm.bufferStats[category]; !exists {
		brm.bufferStats[category] = &BufferStats{}
	}
	stats := brm.bufferStats[category]
	
	// Try to find a reusable buffer
	if buffers, exists := brm.bufferPool[category]; exists && len(buffers) > 0 {
		// Find the best fit buffer
		bestBuffer := brm.findBestFitBuffer(buffers, size)
		if bestBuffer != nil {
			// Remove from pool and return
			brm.removeBufferFromPool(category, bestBuffer)
			bestBuffer.Shape = shape
			bestBuffer.LastUsed = time.Now()
			bestBuffer.UsageCount++
			bestBuffer.IsActive = true
			
			stats.TotalReuses++
			stats.TotalHits++
			stats.CurrentUsage++
			
			return bestBuffer, nil
		}
	}
	
	// No suitable buffer found, allocate new one
	gpuPtr, err := brm.memoryPool.Allocate(int64(size))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU buffer: %w", err)
	}
	
	buffer := &ReusableBuffer{
		GPUPtr:     gpuPtr,
		Size:       size,
		Shape:      shape,
		LastUsed:   time.Now(),
		UsageCount: 1,
		Category:   category,
		IsActive:   true,
	}
	
	stats.TotalAllocations++
	stats.TotalMisses++
	stats.CurrentUsage++
	
	if stats.CurrentUsage > stats.PeakUsage {
		stats.PeakUsage = stats.CurrentUsage
	}
	
	return buffer, nil
}

// ReturnBuffer returns a buffer to the pool for reuse
func (brm *BufferReuseManager) ReturnBuffer(buffer *ReusableBuffer) {
	brm.mutex.Lock()
	defer brm.mutex.Unlock()
	
	if !buffer.IsActive {
		return // Buffer already returned
	}
	
	buffer.IsActive = false
	buffer.LastUsed = time.Now()
	
	// Update stats
	if stats, exists := brm.bufferStats[buffer.Category]; exists {
		stats.CurrentUsage--
	}
	
	// Add to pool if there's room
	if len(brm.bufferPool[buffer.Category]) < brm.maxPoolSize {
		brm.bufferPool[buffer.Category] = append(brm.bufferPool[buffer.Category], buffer)
	} else {
		// Pool is full, release the buffer
		brm.memoryPool.Release(buffer.GPUPtr)
	}
}

// GetTensorBuffer gets a buffer specifically for a tensor and tracks it
func (brm *BufferReuseManager) GetTensorBuffer(tensor *tensor.Tensor, operation string) (*ReusableBuffer, error) {
	buffer, err := brm.GetBuffer(tensor.Shape, operation)
	if err != nil {
		return nil, err
	}
	
	brm.mutex.Lock()
	buffer.OriginalTensor = tensor
	brm.activeBuffers[tensor] = buffer
	brm.mutex.Unlock()
	
	return buffer, nil
}

// ReleaseTensorBuffer releases a buffer associated with a tensor
func (brm *BufferReuseManager) ReleaseTensorBuffer(tensor *tensor.Tensor) {
	brm.mutex.Lock()
	defer brm.mutex.Unlock()
	
	if buffer, exists := brm.activeBuffers[tensor]; exists {
		delete(brm.activeBuffers, tensor)
		buffer.OriginalTensor = nil
		brm.ReturnBuffer(buffer)
	}
}

// findBestFitBuffer finds the best fitting buffer from available buffers
func (brm *BufferReuseManager) findBestFitBuffer(buffers []*ReusableBuffer, targetSize int) *ReusableBuffer {
	var bestBuffer *ReusableBuffer
	bestFit := -1
	
	for _, buffer := range buffers {
		if buffer.Size >= targetSize {
			fit := buffer.Size - targetSize
			if bestFit == -1 || fit < bestFit {
				bestBuffer = buffer
				bestFit = fit
			}
		}
	}
	
	return bestBuffer
}

// removeBufferFromPool removes a buffer from the pool
func (brm *BufferReuseManager) removeBufferFromPool(category string, targetBuffer *ReusableBuffer) {
	buffers := brm.bufferPool[category]
	for i, buffer := range buffers {
		if buffer == targetBuffer {
			// Remove buffer from slice
			brm.bufferPool[category] = append(buffers[:i], buffers[i+1:]...)
			break
		}
	}
}

// periodicCleanup performs periodic cleanup of unused buffers
func (brm *BufferReuseManager) periodicCleanup() {
	for range brm.cleanupTicker.C {
		brm.cleanup()
	}
}

// cleanup removes old unused buffers
func (brm *BufferReuseManager) cleanup() {
	brm.mutex.Lock()
	defer brm.mutex.Unlock()
	
	cutoffTime := time.Now().Add(-5 * time.Minute) // Remove buffers unused for 5 minutes
	
	for category, buffers := range brm.bufferPool {
		var keepBuffers []*ReusableBuffer
		
		for _, buffer := range buffers {
			if buffer.LastUsed.After(cutoffTime) {
				keepBuffers = append(keepBuffers, buffer)
			} else {
				// Release old buffer
				brm.memoryPool.Release(buffer.GPUPtr)
			}
		}
		
		brm.bufferPool[category] = keepBuffers
	}
}

// GetStats returns buffer reuse statistics
func (brm *BufferReuseManager) GetStats() map[string]*BufferStats {
	brm.mutex.RLock()
	defer brm.mutex.RUnlock()
	
	// Return copy of stats
	statsCopy := make(map[string]*BufferStats)
	for category, stats := range brm.bufferStats {
		statsCopy[category] = &BufferStats{
			TotalAllocations: stats.TotalAllocations,
			TotalReuses:      stats.TotalReuses,
			TotalHits:        stats.TotalHits,
			TotalMisses:      stats.TotalMisses,
			AverageLifetime:  stats.AverageLifetime,
			PeakUsage:        stats.PeakUsage,
			CurrentUsage:     stats.CurrentUsage,
		}
	}
	
	return statsCopy
}

// Close stops the buffer reuse manager and releases all resources
func (brm *BufferReuseManager) Close() {
	if brm.cleanupTicker != nil {
		brm.cleanupTicker.Stop()
	}
	
	brm.mutex.Lock()
	defer brm.mutex.Unlock()
	
	// Release all buffers
	for _, buffers := range brm.bufferPool {
		for _, buffer := range buffers {
			brm.memoryPool.Release(buffer.GPUPtr)
		}
	}
	
	// Release active buffers
	for _, buffer := range brm.activeBuffers {
		brm.memoryPool.Release(buffer.GPUPtr)
	}
	
	brm.bufferPool = make(map[string][]*ReusableBuffer)
	brm.activeBuffers = make(map[*tensor.Tensor]*ReusableBuffer)
}

// IntermediateTensorManager manages intermediate tensors used in complex operations
type IntermediateTensorManager struct {
	bufferManager *BufferReuseManager
	tensors       []*tensor.Tensor    // Active intermediate tensors
	buffers       []*ReusableBuffer   // Associated buffers
	mutex         sync.Mutex
}

// NewIntermediateTensorManager creates a new intermediate tensor manager
func NewIntermediateTensorManager(bufferManager *BufferReuseManager) *IntermediateTensorManager {
	return &IntermediateTensorManager{
		bufferManager: bufferManager,
		tensors:       make([]*tensor.Tensor, 0, 16),
		buffers:       make([]*ReusableBuffer, 0, 16),
	}
}

// CreateIntermediateTensor creates a temporary tensor for intermediate computations
func (itm *IntermediateTensorManager) CreateIntermediateTensor(shape []int, operation string) (*tensor.Tensor, error) {
	// Get a reusable buffer
	buffer, err := itm.bufferManager.GetBuffer(shape, operation)
	if err != nil {
		return nil, err
	}
	
	// Create tensor data with appropriate size
	size := calculateBufferSize(shape)
	data := make([]float32, size/4) // Divide by 4 for float32 size
	
	// Create tensor with the allocated data
	tensor, err := tensor.NewTensor(shape, data)
	if err != nil {
		itm.bufferManager.ReturnBuffer(buffer)
		return nil, err
	}
	
	// TODO: Note: In a real implementation, we would associate the GPU buffer with the tensor
	// For now, we'll just track the association
	
	// Track the tensor and buffer
	itm.mutex.Lock()
	itm.tensors = append(itm.tensors, tensor)
	itm.buffers = append(itm.buffers, buffer)
	itm.mutex.Unlock()
	
	return tensor, nil
}

// ReleaseAllIntermediateTensors releases all intermediate tensors
func (itm *IntermediateTensorManager) ReleaseAllIntermediateTensors() {
	itm.mutex.Lock()
	defer itm.mutex.Unlock()
	
	// Return all buffers to the pool
	for i, buffer := range itm.buffers {
		itm.bufferManager.ReturnBuffer(buffer)
		if i < len(itm.tensors) {
			itm.tensors[i].ReleaseGPU() // Release GPU resources
		}
	}
	
	// Clear tracking arrays
	itm.tensors = itm.tensors[:0]
	itm.buffers = itm.buffers[:0]
}

// GetIntermediateTensorCount returns the number of active intermediate tensors
func (itm *IntermediateTensorManager) GetIntermediateTensorCount() int {
	itm.mutex.Lock()
	defer itm.mutex.Unlock()
	return len(itm.tensors)
}

// Utility functions

// calculateBufferSize calculates the GPU buffer size needed for a tensor shape
func calculateBufferSize(shape []int) int {
	size := 4 // Size of float32 in bytes
	for _, dim := range shape {
		size *= dim
	}
	
	// Align to 64-byte boundary for optimal GPU access
	alignment := 64
	return ((size + alignment - 1) / alignment) * alignment
}

// categorizeBufferSize categorizes buffer sizes for efficient reuse
func categorizeBufferSize(size int) string {
	switch {
	case size <= 1024:        // 1KB
		return "tiny"
	case size <= 16*1024:     // 16KB
		return "small"
	case size <= 256*1024:    // 256KB
		return "medium"
	case size <= 4*1024*1024: // 4MB
		return "large"
	case size <= 64*1024*1024: // 64MB
		return "xlarge"
	default:
		return "huge"
	}
}

// Global buffer reuse manager
var globalBufferReuseManager *BufferReuseManager
var bufferReuseManagerOnce sync.Once

// InitializeGlobalBufferReuseManager initializes the global buffer reuse manager
func InitializeGlobalBufferReuseManager(memoryPool *GPUMemoryPool) {
	bufferReuseManagerOnce.Do(func() {
		globalBufferReuseManager = NewBufferReuseManager(memoryPool)
	})
}

// GetGlobalBufferReuseManager returns the global buffer reuse manager
func GetGlobalBufferReuseManager() *BufferReuseManager {
	if globalBufferReuseManager == nil {
		// Try to initialize from the global memory optimization suite
		if suite := GetGlobalMemoryOptimizationSuite(); suite != nil && suite.bufferReuseManager != nil {
			globalBufferReuseManager = suite.bufferReuseManager
		}
	}
	return globalBufferReuseManager
}

// CreateOptimizedIntermediateTensor creates an optimized intermediate tensor
func CreateOptimizedIntermediateTensor(shape []int, operation string) (*tensor.Tensor, error) {
	if globalBufferReuseManager == nil {
		// Fall back to regular tensor creation
		data := make([]float32, calculateBufferSize(shape)/4)
		return tensor.NewTensor(shape, data)
	}
	
	// Use the global intermediate tensor manager
	manager := NewIntermediateTensorManager(globalBufferReuseManager)
	return manager.CreateIntermediateTensor(shape, operation)
}

// OperationScope represents a scope for managing intermediate tensors in an operation
type OperationScope struct {
	manager     *IntermediateTensorManager
	isActive    bool
	operation   string
	startTime   time.Time
}

// NewOperationScope creates a new operation scope for managing intermediate tensors
func NewOperationScope(operation string) *OperationScope {
	bufferManager := GetGlobalBufferReuseManager()
	if bufferManager == nil {
		return &OperationScope{isActive: false}
	}
	
	return &OperationScope{
		manager:   NewIntermediateTensorManager(bufferManager),
		isActive:  true,
		operation: operation,
		startTime: time.Now(),
	}
}

// CreateTensor creates an intermediate tensor within this scope
func (os *OperationScope) CreateTensor(shape []int) (*tensor.Tensor, error) {
	if !os.isActive {
		// Fall back to regular tensor creation
		data := make([]float32, calculateBufferSize(shape)/4)
		return tensor.NewTensor(shape, data)
	}
	
	return os.manager.CreateIntermediateTensor(shape, os.operation)
}

// Close releases all intermediate tensors in this scope
func (os *OperationScope) Close() {
	if os.isActive {
		os.manager.ReleaseAllIntermediateTensors()
		os.isActive = false
	}
}

// GetTensorCount returns the number of active intermediate tensors in this scope
func (os *OperationScope) GetTensorCount() int {
	if !os.isActive {
		return 0
	}
	return os.manager.GetIntermediateTensorCount()
}

// GetDuration returns the duration this scope has been active
func (os *OperationScope) GetDuration() time.Duration {
	return time.Since(os.startTime)
}