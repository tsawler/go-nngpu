package tensor

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework Accelerate
#cgo CFLAGS: -x objective-c -ObjC -fobjc-arc
#include <stdlib.h>
#include "../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// globalMemoryPool is a package-level memory pool for tracking all tensor allocations
var globalMemoryPool *GlobalMemoryPool
var globalMemoryPoolOnce sync.Once

// GlobalMemoryPool tracks all GPU memory allocations made by tensors
type GlobalMemoryPool struct {
	allocations map[unsafe.Pointer]int64
	totalUsage  int64
	peakUsage   int64
	mutex       sync.RWMutex
}

// initGlobalMemoryPool initializes the global memory pool
func initGlobalMemoryPool() {
	globalMemoryPool = &GlobalMemoryPool{
		allocations: make(map[unsafe.Pointer]int64),
	}
}

// getGlobalMemoryPool returns the singleton global memory pool
func getGlobalMemoryPool() *GlobalMemoryPool {
	globalMemoryPoolOnce.Do(initGlobalMemoryPool)
	return globalMemoryPool
}

// trackAllocation records a new allocation
func (p *GlobalMemoryPool) trackAllocation(ptr unsafe.Pointer, size int64) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	p.allocations[ptr] = size
	p.totalUsage += size
	if p.totalUsage > p.peakUsage {
		p.peakUsage = p.totalUsage
	}
}

// trackDeallocation records a deallocation
func (p *GlobalMemoryPool) trackDeallocation(ptr unsafe.Pointer) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	if size, exists := p.allocations[ptr]; exists {
		delete(p.allocations, ptr)
		p.totalUsage -= size
	}
}

// GetStats returns current memory usage statistics
func (p *GlobalMemoryPool) GetStats() (currentUsage, peakUsage int64, numAllocations int) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	return p.totalUsage, p.peakUsage, len(p.allocations)
}

// GetGlobalMemoryStats returns global memory statistics for all tensors
func GetGlobalMemoryStats() (currentUsage, peakUsage int64, numAllocations int) {
	pool := getGlobalMemoryPool()
	return pool.GetStats()
}

// Tensor represents a multi-dimensional array of float32, potentially backed by a Metal GPU buffer.
type Tensor struct {
	Shape     []int          // Dimensions of the tensor (e.g., [rows, cols] for a matrix)
	Data      []float32      // CPU-side data
	gpuPtr    unsafe.Pointer // Pointer to the underlying C-allocated MTLBuffer (opaque to Go)
	isOnGPU   bool
	isOwner   bool           // Does this Tensor own the GPU buffer memory? (for deallocation)
	devicePtr unsafe.Pointer // Opaque pointer to the MTLDevice used
}

// NewTensor creates a new CPU-backed Tensor.
func NewTensor(shape []int, data []float32) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if len(data) != size {
		return nil, fmt.Errorf("data length (%d) does not match shape dimensions (%d)", len(data), size)
	}
	t := &Tensor{
		Shape:   shape,
		Data:    data,
		isOnGPU: false,
		isOwner: true, // New tensors own their CPU data
	}
	// TODO: It's common practice to set a finalizer to ensure GPU resources are cleaned up
	// even if ReleaseGPU() isn't called explicitly. However, for a production library,
	// explicit management is preferred for predictable behavior. For now, rely on defer.
	// runtime.SetFinalizer(t, (*Tensor).finalizer)
	return t, nil
}

// EnsureGPU ensures the tensor's data is on the GPU.
// If it's already on GPU, it does nothing. If on CPU, it transfers it.
func (t *Tensor) EnsureGPU() error {
	if t.isOnGPU {
		return nil // Already on GPU
	}

	// Lock OS thread to prevent Go GC from moving memory while C is using it
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Call CGO function to get a GPU buffer
	cData := (*C.float)(unsafe.Pointer(&t.Data[0]))
	cLen := C.long(len(t.Data) * int(unsafe.Sizeof(t.Data[0]))) // Length in bytes
	// cDevicePtr := (*C.void)(nil) // This was problematic, device should be managed once globally

	var cGPUPtr C.GPUPtr
	var cDevice C.DevicePtr // Use a temporary for the device pointer returned
	var cErr C.CError
	retCode := C.create_gpu_buffer(cData, cLen, &cGPUPtr, &cDevice, &cErr)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message) // Free memory from C
		}
		return fmt.Errorf("failed to create GPU buffer (code %d): %s", retCode, errMsg)
	}

	t.gpuPtr = unsafe.Pointer(cGPUPtr)
	t.devicePtr = unsafe.Pointer(cDevice) // Store the device pointer
	t.isOnGPU = true
	
	// Track the allocation in the global memory pool
	pool := getGlobalMemoryPool()
	pool.trackAllocation(t.gpuPtr, int64(cLen))
	
	return nil
}

// RetrieveCPU ensures the tensor's data is on the CPU.
// If it's already on CPU, it does nothing. If on GPU, it transfers it.
func (t *Tensor) RetrieveCPU() error {
	if !t.isOnGPU {
		return nil // Already on CPU
	}

	// Lock OS thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if t.gpuPtr == nil {
		return fmt.Errorf("tensor is marked as on GPU but has no GPU pointer")
	}

	cData := (*C.float)(unsafe.Pointer(&t.Data[0]))             // Pointer to where data should be copied
	cLen := C.long(len(t.Data) * int(unsafe.Sizeof(t.Data[0]))) // Length in bytes

	var cErr C.CError
	retCode := C.retrieve_gpu_buffer_data(C.GPUPtr(t.gpuPtr), cData, cLen, &cErr)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("failed to retrieve GPU buffer data (code %d): %s", retCode, errMsg)
	}

	t.isOnGPU = false // Data is now on CPU
	return nil
}

// ReleaseGPU releases the GPU-side buffer.
// Call this when the GPU data is no longer needed.
// It checks if the tensor is the owner of the GPU buffer.
func (t *Tensor) ReleaseGPU() {
	if t.isOnGPU && t.gpuPtr != nil && t.isOwner {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		
		// Track the deallocation in the global memory pool
		pool := getGlobalMemoryPool()
		pool.trackDeallocation(t.gpuPtr)
		
		C.release_gpu_buffer(C.GPUPtr(t.gpuPtr))
		t.gpuPtr = nil
		t.isOnGPU = false
	}
}

// GPUPtr returns the unsafe.Pointer to the GPU buffer.
// This is an internal helper for CGO calls within the go-nngpu module.
func (t *Tensor) GPUPtr() unsafe.Pointer {
	return t.gpuPtr
}

// DevicePtr returns the unsafe.Pointer to the Metal device.
// This is an internal helper for CGO calls within the go-nngpu module.
func (t *Tensor) DevicePtr() unsafe.Pointer {
	return t.devicePtr
}
