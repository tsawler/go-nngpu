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
	"sync"
	"time"
	"unsafe"

	"github.com/tsawler/gometal/tensor"
)

// Phase 8C: Memory Bandwidth Optimization
// This file implements optimizations to reduce memory bandwidth usage and improve GPU performance

// AllocateAlignedBuffer allocates an aligned GPU buffer using the optimized C functions
func AllocateAlignedBuffer(device unsafe.Pointer, size int, alignment int) (unsafe.Pointer, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	var buffer C.GPUPtr
	result := C.allocate_aligned_gpu_buffer(C.long(size), C.long(alignment), &buffer, C.DevicePtr(device), &cErr)
	if result != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("failed to allocate aligned GPU buffer: %s", errMsg)
	}

	return unsafe.Pointer(buffer), nil
}

// ReleaseOptimizedBuffer releases a GPU buffer using optimized C functions
func ReleaseOptimizedBuffer(buffer unsafe.Pointer) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	result := C.release_optimized_gpu_buffer(C.GPUPtr(buffer), &cErr)
	if result != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("failed to release GPU buffer: %s", errMsg)
	}
	return nil
}

// CoalescedMemoryCopy performs optimized memory copy with coalescing
func CoalescedMemoryCopy(device unsafe.Pointer, srcPtr, dstPtr unsafe.Pointer, size int, srcStride, dstStride int) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	result := C.coalesced_memory_copy(C.GPUPtr(srcPtr), C.GPUPtr(dstPtr), C.long(size),
		C.long(srcStride), C.long(dstStride), C.DevicePtr(device), &cErr)
	if result != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("coalesced memory copy failed: %s", errMsg)
	}
	return nil
}

// PrefetchGPUData prefetches data to GPU cache for optimal access patterns
func PrefetchGPUData(device unsafe.Pointer, bufferPtr unsafe.Pointer, size, offset int) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	result := C.prefetch_gpu_data(C.GPUPtr(bufferPtr), C.long(size), C.long(offset), C.DevicePtr(device), &cErr)
	if result != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU data prefetch failed: %s", errMsg)
	}
	return nil
}

// FlushGPUCache flushes GPU cache to ensure optimal access patterns
func FlushGPUCache(device unsafe.Pointer) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	result := C.flush_gpu_cache(C.DevicePtr(device), &cErr)
	if result != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU cache flush failed: %s", errMsg)
	}
	return nil
}

// Memory optimization utilities

// OptimizedTensorBuffer manages efficient buffer allocation for tensors
type OptimizedTensorBuffer struct {
	tempBuffers []unsafe.Pointer // Temporary buffers for intermediate results
	bufferSizes []int            // Sizes of temporary buffers
	device      unsafe.Pointer   // Metal device
	mutex       sync.Mutex
}

// NewOptimizedTensorBuffer creates a new optimized tensor buffer manager
func NewOptimizedTensorBuffer(device unsafe.Pointer) *OptimizedTensorBuffer {
	return &OptimizedTensorBuffer{
		device:      device,
		tempBuffers: make([]unsafe.Pointer, 0, 16), // Pre-allocate for common cases
		bufferSizes: make([]int, 0, 16),
	}
}

// GetTempBuffer allocates a temporary buffer for intermediate computations
func (otb *OptimizedTensorBuffer) GetTempBuffer(size int) (unsafe.Pointer, error) {
	otb.mutex.Lock()
	defer otb.mutex.Unlock()

	// Try to reuse an existing temporary buffer
	for i, bufSize := range otb.bufferSizes {
		if bufSize >= size && otb.tempBuffers[i] != nil {
			buffer := otb.tempBuffers[i]
			otb.tempBuffers[i] = nil // Mark as used
			return buffer, nil
		}
	}

	// Allocate new temporary buffer with alignment
	buffer, err := AllocateAlignedBuffer(otb.device, size*4, 256) // size*4 for float32
	if err != nil {
		return nil, err
	}

	// Track the buffer
	otb.tempBuffers = append(otb.tempBuffers, nil) // Will be set when returned
	otb.bufferSizes = append(otb.bufferSizes, size)

	return buffer, nil
}

// ReturnTempBuffer returns a temporary buffer for reuse
func (otb *OptimizedTensorBuffer) ReturnTempBuffer(buffer unsafe.Pointer) {
	otb.mutex.Lock()
	defer otb.mutex.Unlock()

	// Find an empty slot to store the buffer
	for i, buf := range otb.tempBuffers {
		if buf == nil {
			otb.tempBuffers[i] = buffer
			return
		}
	}

	// No empty slot, release buffer
	ReleaseOptimizedBuffer(buffer)
}

// ReleaseAll releases all temporary buffers
func (otb *OptimizedTensorBuffer) ReleaseAll() {
	otb.mutex.Lock()
	defer otb.mutex.Unlock()

	for i, buffer := range otb.tempBuffers {
		if buffer != nil {
			ReleaseOptimizedBuffer(buffer)
			otb.tempBuffers[i] = nil
		}
	}
}

// SimpleMemoryCoalescingOptimizer optimizes memory access patterns for GPU efficiency
// Note: A more advanced version is available in memory-coalescing-optimizer.go
type SimpleMemoryCoalescingOptimizer struct {
	preferredAlignment int // Preferred memory alignment in bytes
	cacheLineSize      int // GPU cache line size
}

// NewSimpleMemoryCoalescingOptimizer creates a new simple memory coalescing optimizer
func NewSimpleMemoryCoalescingOptimizer() *SimpleMemoryCoalescingOptimizer {
	return &SimpleMemoryCoalescingOptimizer{
		preferredAlignment: 128, // 128-byte alignment for optimal coalescing
		cacheLineSize:      64,  // Common GPU cache line size
	}
}

// OptimizeTensorLayout reorganizes tensor data for optimal GPU access patterns
func (mco *SimpleMemoryCoalescingOptimizer) OptimizeTensorLayout(tensor *tensor.Tensor, operation string) (*tensor.Tensor, error) {
	switch operation {
	case "conv2d":
		return mco.optimizeForConvolution(tensor)
	case "matmul":
		return mco.optimizeForMatrixMultiplication(tensor)
	case "elementwise":
		return mco.optimizeForElementwise(tensor)
	default:
		return tensor, nil // No optimization for unknown operations
	}
}

// optimizeForConvolution reorganizes data for convolution operations
func (mco *SimpleMemoryCoalescingOptimizer) optimizeForConvolution(t *tensor.Tensor) (*tensor.Tensor, error) {
	// For convolution, we want NHWC layout with channel padding for alignment
	if len(t.Shape) != 4 {
		return t, nil
	}

	batch, height, width, channels := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]

	// Pad channels to multiple of cache line size (in float32s)
	alignedChannels := ((channels + 15) / 16) * 16 // 16 float32s = 64 bytes

	if alignedChannels == channels {
		return t, nil // Already aligned
	}

	// Create new tensor with padded channels
	newShape := []int{batch, height, width, alignedChannels}
	newData := make([]float32, batch*height*width*alignedChannels)

	// Copy data with channel padding
	for b := 0; b < batch; b++ {
		for h := 0; h < height; h++ {
			for w := 0; w < width; w++ {
				srcOffset := ((b*height+h)*width + w) * channels
				dstOffset := ((b*height+h)*width + w) * alignedChannels

				copy(newData[dstOffset:dstOffset+channels], t.Data[srcOffset:srcOffset+channels])
				// Padding channels are left as zero
			}
		}
	}

	return tensor.NewTensor(newShape, newData)
}

// optimizeForMatrixMultiplication reorganizes data for matrix operations
func (mco *SimpleMemoryCoalescingOptimizer) optimizeForMatrixMultiplication(t *tensor.Tensor) (*tensor.Tensor, error) {
	if len(t.Shape) != 2 {
		return t, nil
	}

	rows, cols := t.Shape[0], t.Shape[1]

	// Pad columns to multiple of 16 for vectorized operations
	alignedCols := ((cols + 15) / 16) * 16

	if alignedCols == cols {
		return t, nil // Already aligned
	}

	// Create new tensor with padded columns
	newShape := []int{rows, alignedCols}
	newData := make([]float32, rows*alignedCols)

	// Copy data with column padding
	for r := 0; r < rows; r++ {
		srcOffset := r * cols
		dstOffset := r * alignedCols
		copy(newData[dstOffset:dstOffset+cols], t.Data[srcOffset:srcOffset+cols])
		// Padding columns are left as zero
	}

	return tensor.NewTensor(newShape, newData)
}

// optimizeForElementwise ensures data is aligned for vectorized operations
func (mco *SimpleMemoryCoalescingOptimizer) optimizeForElementwise(t *tensor.Tensor) (*tensor.Tensor, error) {
	totalSize := len(t.Data)

	// Pad to multiple of 16 for SIMD operations
	alignedSize := ((totalSize + 15) / 16) * 16

	if alignedSize == totalSize {
		return t, nil // Already aligned
	}

	// Create new tensor with padding
	newData := make([]float32, alignedSize)
	copy(newData, t.Data)
	// Padding elements are left as zero

	// Keep original shape, but note that the underlying data is larger
	return tensor.NewTensor(t.Shape, newData[:totalSize])
}

// GPU-CPU Transfer Optimizer reduces unnecessary memory transfers
// Note: A more advanced version is available in gpu-cpu-transfer-optimizer.go
type SimpleTransferOptimizer struct {
	gpuDataCache  map[*tensor.Tensor]bool   // Tracks which tensors have valid GPU data
	cpuDataCache  map[*tensor.Tensor]bool   // Tracks which tensors have valid CPU data
	lastOperation map[*tensor.Tensor]string // Last operation on each tensor
	mutex         sync.RWMutex
}

// NewSimpleTransferOptimizer creates a new simple transfer optimizer
func NewSimpleTransferOptimizer() *SimpleTransferOptimizer {
	return &SimpleTransferOptimizer{
		gpuDataCache:  make(map[*tensor.Tensor]bool),
		cpuDataCache:  make(map[*tensor.Tensor]bool),
		lastOperation: make(map[*tensor.Tensor]string),
	}
}

// MarkGPUValid marks a tensor as having valid GPU data
func (to *SimpleTransferOptimizer) MarkGPUValid(tensor *tensor.Tensor, operation string) {
	to.mutex.Lock()
	defer to.mutex.Unlock()

	to.gpuDataCache[tensor] = true
	to.lastOperation[tensor] = operation
}

// MarkCPUValid marks a tensor as having valid CPU data
func (to *SimpleTransferOptimizer) MarkCPUValid(tensor *tensor.Tensor) {
	to.mutex.Lock()
	defer to.mutex.Unlock()

	to.cpuDataCache[tensor] = true
}

// MarkGPUInvalid marks a tensor's GPU data as invalid
func (to *SimpleTransferOptimizer) MarkGPUInvalid(tensor *tensor.Tensor) {
	to.mutex.Lock()
	defer to.mutex.Unlock()

	to.gpuDataCache[tensor] = false
}

// MarkCPUInvalid marks a tensor's CPU data as invalid
func (to *SimpleTransferOptimizer) MarkCPUInvalid(tensor *tensor.Tensor) {
	to.mutex.Lock()
	defer to.mutex.Unlock()

	to.cpuDataCache[tensor] = false
}

// ShouldTransferToGPU determines if a tensor needs to be transferred to GPU
func (to *SimpleTransferOptimizer) ShouldTransferToGPU(tensor *tensor.Tensor) bool {
	to.mutex.RLock()
	defer to.mutex.RUnlock()

	gpuValid, exists := to.gpuDataCache[tensor]
	return !exists || !gpuValid
}

// ShouldTransferToCPU determines if a tensor needs to be transferred to CPU
func (to *SimpleTransferOptimizer) ShouldTransferToCPU(tensor *tensor.Tensor) bool {
	to.mutex.RLock()
	defer to.mutex.RUnlock()

	cpuValid, exists := to.cpuDataCache[tensor]
	return !exists || !cpuValid
}

// CleanupTensor removes tracking for a tensor
func (to *SimpleTransferOptimizer) CleanupTensor(tensor *tensor.Tensor) {
	to.mutex.Lock()
	defer to.mutex.Unlock()

	delete(to.gpuDataCache, tensor)
	delete(to.cpuDataCache, tensor)
	delete(to.lastOperation, tensor)
}

// Global instances for simple optimizers (fallback versions)
var globalSimpleMemoryOptimizer *SimpleMemoryCoalescingOptimizer
var globalSimpleTransferOptimizer *SimpleTransferOptimizer
var simpleMemoryOptimizerOnce sync.Once

// InitializeMemoryOptimizers initializes global memory optimizers
func InitializeMemoryOptimizers() {
	simpleMemoryOptimizerOnce.Do(func() {
		globalSimpleMemoryOptimizer = NewSimpleMemoryCoalescingOptimizer()
		globalSimpleTransferOptimizer = NewSimpleTransferOptimizer()
	})
}

// GetGlobalSimpleMemoryOptimizer returns the global simple memory coalescing optimizer
func GetGlobalSimpleMemoryOptimizer() *SimpleMemoryCoalescingOptimizer {
	return globalSimpleMemoryOptimizer
}

// GetGlobalSimpleTransferOptimizer returns the global simple transfer optimizer
func GetGlobalSimpleTransferOptimizer() *SimpleTransferOptimizer {
	return globalSimpleTransferOptimizer
}

// Utility functions

// nextPowerOf2 returns the next power of 2 greater than or equal to n
func nextPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}

	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++

	return n
}

// OptimizedTensorAlloc allocates a tensor using the optimized memory pool
func OptimizedTensorAlloc(shape []int, operation string) (*tensor.Tensor, error) {
	// Initialize optimizers if needed
	InitializeMemoryOptimizers()

	// Calculate size
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	data := make([]float32, size)
	tensor, err := tensor.NewTensor(shape, data)
	if err != nil {
		return nil, err
	}

	// Optimize layout if beneficial
	if globalSimpleMemoryOptimizer != nil {
		optimized, err := globalSimpleMemoryOptimizer.OptimizeTensorLayout(tensor, operation)
		if err != nil {
			return tensor, nil // Fall back to original if optimization fails
		}
		tensor = optimized
	}

	return tensor, nil
}

// Memory bandwidth monitoring and optimization hints
type MemoryBandwidthMonitor struct {
	transferTimes    []time.Duration
	transferSizes    []int64
	lastTransferTime time.Time
	totalTransfers   int64
	totalBytes       int64
	mutex            sync.Mutex
}

// NewMemoryBandwidthMonitor creates a new bandwidth monitor
func NewMemoryBandwidthMonitor() *MemoryBandwidthMonitor {
	return &MemoryBandwidthMonitor{
		transferTimes: make([]time.Duration, 0, 1000),
		transferSizes: make([]int64, 0, 1000),
	}
}

// RecordTransfer records a memory transfer for bandwidth analysis
func (mbm *MemoryBandwidthMonitor) RecordTransfer(bytes int64, duration time.Duration) {
	mbm.mutex.Lock()
	defer mbm.mutex.Unlock()

	mbm.transferTimes = append(mbm.transferTimes, duration)
	mbm.transferSizes = append(mbm.transferSizes, bytes)
	mbm.totalTransfers++
	mbm.totalBytes += bytes
	mbm.lastTransferTime = time.Now()

	// Keep only recent transfers (last 1000)
	if len(mbm.transferTimes) > 1000 {
		mbm.transferTimes = mbm.transferTimes[100:] // Remove oldest 100
		mbm.transferSizes = mbm.transferSizes[100:]
	}
}

// GetBandwidthStats returns current bandwidth statistics
func (mbm *MemoryBandwidthMonitor) GetBandwidthStats() (avgBandwidth float64, peakBandwidth float64, totalTransfers int64) {
	mbm.mutex.Lock()
	defer mbm.mutex.Unlock()

	if len(mbm.transferTimes) == 0 {
		return 0, 0, mbm.totalTransfers
	}

	totalTime := time.Duration(0)
	totalSize := int64(0)
	maxBandwidth := 0.0

	for i, duration := range mbm.transferTimes {
		if duration > 0 {
			bandwidth := float64(mbm.transferSizes[i]) / duration.Seconds()
			if bandwidth > maxBandwidth {
				maxBandwidth = bandwidth
			}
			totalTime += duration
			totalSize += mbm.transferSizes[i]
		}
	}

	avgBandwidth = 0
	if totalTime > 0 {
		avgBandwidth = float64(totalSize) / totalTime.Seconds()
	}

	return avgBandwidth, maxBandwidth, mbm.totalTransfers
}

// Global bandwidth monitor
var globalBandwidthMonitor *MemoryBandwidthMonitor
var bandwidthMonitorOnce sync.Once

// GetGlobalBandwidthMonitor returns the global bandwidth monitor
func GetGlobalBandwidthMonitor() *MemoryBandwidthMonitor {
	bandwidthMonitorOnce.Do(func() {
		globalBandwidthMonitor = NewMemoryBandwidthMonitor()
	})
	return globalBandwidthMonitor
}
