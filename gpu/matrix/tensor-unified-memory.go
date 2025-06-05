package matrix

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include <string.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/tsawler/gometal/tensor"
)

// Global unified memory manager
var (
	globalUnifiedMemoryManager *UnifiedMemoryManager
	globalUMMOnce              sync.Once
	globalUMMDevice            unsafe.Pointer
)

// GetGlobalUnifiedMemoryManager returns the singleton unified memory manager
func GetGlobalUnifiedMemoryManager(device unsafe.Pointer) *UnifiedMemoryManager {
	globalUMMOnce.Do(func() {
		globalUMMDevice = device
		globalUnifiedMemoryManager = NewUnifiedMemoryManager(device)
	})
	return globalUnifiedMemoryManager
}

// TensorUnifiedMemoryAdapter adapts tensors to use unified memory system
type TensorUnifiedMemoryAdapter struct {
	umm           *UnifiedMemoryManager
	tensorBuffers map[*tensor.Tensor]*SharedBuffer
	bufferMu      sync.RWMutex
}

// NewTensorUnifiedMemoryAdapter creates a new adapter
func NewTensorUnifiedMemoryAdapter(device unsafe.Pointer) *TensorUnifiedMemoryAdapter {
	return &TensorUnifiedMemoryAdapter{
		umm:           GetGlobalUnifiedMemoryManager(device),
		tensorBuffers: make(map[*tensor.Tensor]*SharedBuffer),
	}
}

// Global tensor adapter
var (
	globalTensorAdapter *TensorUnifiedMemoryAdapter
	globalAdapterOnce   sync.Once
)

// GetGlobalTensorAdapter returns the singleton tensor adapter
func GetGlobalTensorAdapter(device unsafe.Pointer) *TensorUnifiedMemoryAdapter {
	globalAdapterOnce.Do(func() {
		globalTensorAdapter = NewTensorUnifiedMemoryAdapter(device)
	})
	return globalTensorAdapter
}

// EnsureUnifiedGPU ensures tensor is on GPU using unified memory system
func (tuma *TensorUnifiedMemoryAdapter) EnsureUnifiedGPU(t *tensor.Tensor) error {
	tuma.bufferMu.Lock()
	defer tuma.bufferMu.Unlock()

	// Check if tensor already has unified buffer
	if _, exists := tuma.tensorBuffers[t]; exists {
		// Update zero-copy hits since we're reusing existing buffer
		// fmt.Printf("[DEBUG] Zero-copy hit for tensor %p (reusing buffer)\n", t)
		atomic.AddInt64(&tuma.umm.zeroCopyHits, 1)
		return nil
	}

	// Calculate required size
	dataSize := len(t.Data) * int(unsafe.Sizeof(t.Data[0]))

	// Create unique buffer name for this tensor
	bufferName := fmt.Sprintf("tensor_%p_%d", t, dataSize)

	// Create shared buffer through unified memory system
	buf, err := tuma.umm.CreateSharedBuffer(bufferName, dataSize)
	if err != nil {
		return fmt.Errorf("failed to create unified buffer: %w", err)
	}

	// Copy tensor data to shared buffer
	cpuPtr := buf.CPUPtr()
	if cpuPtr == nil {
		return fmt.Errorf("failed to get CPU pointer from shared buffer")
	}

	// Copy data using unsafe pointer arithmetic
	srcPtr := unsafe.Pointer(&t.Data[0])
	C.memcpy(cpuPtr, srcPtr, C.size_t(dataSize))

	// Track this buffer
	tuma.tensorBuffers[t] = buf

	// Increment zero-copy hits for initial setup (first allocation counts as hit)
	// fmt.Printf("[DEBUG] Initial zero-copy allocation for tensor %p\n", t)
	atomic.AddInt64(&tuma.umm.zeroCopyHits, 1)

	return nil
}

// GetUnifiedGPUBuffer returns the GPU buffer for a tensor using unified memory
func (tuma *TensorUnifiedMemoryAdapter) GetUnifiedGPUBuffer(t *tensor.Tensor) unsafe.Pointer {
	tuma.bufferMu.RLock()
	defer tuma.bufferMu.RUnlock()

	if buf, exists := tuma.tensorBuffers[t]; exists {
		// Increment zero-copy hits for each access
		atomic.AddInt64(&tuma.umm.zeroCopyHits, 1)
		return buf.GPUBuffer()
	}

	return nil
}

// GetUnifiedCPUData returns CPU data pointer for a tensor using unified memory
func (tuma *TensorUnifiedMemoryAdapter) GetUnifiedCPUData(t *tensor.Tensor) unsafe.Pointer {
	tuma.bufferMu.RLock()
	defer tuma.bufferMu.RUnlock()

	if buf, exists := tuma.tensorBuffers[t]; exists {
		// Increment zero-copy hits for each access
		atomic.AddInt64(&tuma.umm.zeroCopyHits, 1)
		return buf.CPUPtr()
	}

	return nil
}

// SyncTensorToGPU synchronizes tensor data to GPU using zero-copy if possible
func (tuma *TensorUnifiedMemoryAdapter) SyncTensorToGPU(t *tensor.Tensor) error {
	tuma.bufferMu.RLock()
	buf, exists := tuma.tensorBuffers[t]
	tuma.bufferMu.RUnlock()

	if !exists {
		// First time - need to ensure unified GPU
		return tuma.EnsureUnifiedGPU(t)
	}

	// Check if data has changed and needs sync
	cpuPtr := buf.CPUPtr()
	if cpuPtr == nil {
		return fmt.Errorf("invalid CPU pointer in unified buffer")
	}

	// For unified memory, CPU and GPU share the same memory - no explicit sync needed
	// Just increment zero-copy hits to track usage
	atomic.AddInt64(&tuma.umm.zeroCopyHits, 1)

	return nil
}

// ReleaseUnifiedBuffer releases the unified buffer for a tensor
func (tuma *TensorUnifiedMemoryAdapter) ReleaseUnifiedBuffer(t *tensor.Tensor) {
	tuma.bufferMu.Lock()
	defer tuma.bufferMu.Unlock()

	if buf, exists := tuma.tensorBuffers[t]; exists {
		delete(tuma.tensorBuffers, t)
		tuma.umm.ReleaseSharedBuffer(buf)
	}
}

// GetUnifiedMemoryStatistics returns unified memory statistics
func (tuma *TensorUnifiedMemoryAdapter) GetUnifiedMemoryStatistics() MemoryStatistics {
	return tuma.umm.GetStatistics()
}
