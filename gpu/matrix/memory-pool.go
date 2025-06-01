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

	"github.com/tsawler/go-nngpu/tensor"
	_ "github.com/tsawler/go-nngpu/internal/cgo"
)

// GPUMemoryPool manages GPU memory allocation and reuse
type GPUMemoryPool struct {
	maxMemory     int64
	currentUsage  int64
	allocatedPtrs map[unsafe.Pointer]*MemoryBlock
	freeBlocks    map[int64][]*MemoryBlock // Size -> list of free blocks
	mutex         sync.RWMutex
	stats         MemoryStats
}

// MemoryBlock represents a block of GPU memory
type MemoryBlock struct {
	ptr       unsafe.Pointer
	size      int64
	allocated time.Time
	lastUsed  time.Time
	inUse     bool
	refCount  int
}

// MemoryStats tracks memory pool statistics
type MemoryStats struct {
	TotalAllocated     int64
	TotalFreed        int64
	PeakUsage         int64
	AllocationCount   int64
	FreeCount         int64
	CacheHits         int64
	CacheMisses       int64
	FragmentationRatio float32
}

// TensorCache provides caching for frequently used tensors
type TensorCache struct {
	cache     map[string]*CachedTensor
	lru       *LRUList
	maxSize   int
	mutex     sync.RWMutex
	hits      int64
	misses    int64
}

// CachedTensor represents a cached tensor with metadata
type CachedTensor struct {
	tensor    *tensor.Tensor
	shape     []int
	size      int64
	created   time.Time
	accessed  time.Time
	useCount  int64
	lruNode   *LRUNode
}

// LRUList implements a doubly-linked list for LRU cache
type LRUList struct {
	head *LRUNode
	tail *LRUNode
	size int
}

// LRUNode represents a node in the LRU list
type LRUNode struct {
	key   string
	prev  *LRUNode
	next  *LRUNode
}

// NewGPUMemoryPool creates a new GPU memory pool
func NewGPUMemoryPool(maxMemory int64) (*GPUMemoryPool, error) {
	pool := &GPUMemoryPool{
		maxMemory:     maxMemory,
		allocatedPtrs: make(map[unsafe.Pointer]*MemoryBlock),
		freeBlocks:    make(map[int64][]*MemoryBlock),
	}
	
	// Initialize memory pool on GPU side
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	
	var cErr C.CError
	retCode := C.initialize_memory_pool(
		C.long(maxMemory),
		&cErr,
	)
	
	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("failed to initialize GPU memory pool (code %d): %s", retCode, errMsg)
	}
	
	return pool, nil
}

// Allocate allocates memory from the pool
func (p *GPUMemoryPool) Allocate(size int64) (unsafe.Pointer, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// Check if we have a free block of the right size
	if blocks, exists := p.freeBlocks[size]; exists && len(blocks) > 0 {
		// Reuse existing block
		block := blocks[len(blocks)-1]
		p.freeBlocks[size] = blocks[:len(blocks)-1]
		
		block.inUse = true
		block.lastUsed = time.Now()
		block.refCount = 1
		
		p.stats.CacheHits++
		return block.ptr, nil
	}
	
	// Check memory limit
	if p.maxMemory > 0 && p.currentUsage+size > p.maxMemory {
		// Try to free some memory
		err := p.garbageCollect()
		if err != nil {
			return nil, fmt.Errorf("garbage collection failed: %w", err)
		}
		
		// Check again after GC
		if p.currentUsage+size > p.maxMemory {
			return nil, fmt.Errorf("out of GPU memory: requested %d bytes, available %d bytes", 
				size, p.maxMemory-p.currentUsage)
		}
	}
	
	// Allocate new block
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	
	var ptr C.GPUPtr
	var cErr C.CError
	retCode := C.allocate_gpu_memory(
		C.long(size),
		&ptr,
		&cErr,
	)
	
	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU memory allocation failed (code %d): %s", retCode, errMsg)
	}
	
	block := &MemoryBlock{
		ptr:       unsafe.Pointer(ptr),
		size:      size,
		allocated: time.Now(),
		lastUsed:  time.Now(),
		inUse:     true,
		refCount:  1,
	}
	
	p.allocatedPtrs[block.ptr] = block
	p.currentUsage += size
	p.stats.TotalAllocated += size
	p.stats.AllocationCount++
	p.stats.CacheMisses++
	
	if p.currentUsage > p.stats.PeakUsage {
		p.stats.PeakUsage = p.currentUsage
	}
	
	return block.ptr, nil
}

// Free returns memory to the pool for reuse
func (p *GPUMemoryPool) Free(ptr unsafe.Pointer) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	block, exists := p.allocatedPtrs[ptr]
	if !exists {
		return fmt.Errorf("attempting to free unallocated pointer")
	}
	
	block.refCount--
	if block.refCount > 0 {
		return nil // Still has references
	}
	
	block.inUse = false
	block.lastUsed = time.Now()
	
	// Add to free blocks for reuse
	p.freeBlocks[block.size] = append(p.freeBlocks[block.size], block)
	p.stats.FreeCount++
	
	return nil
}

// Release permanently frees memory and releases GPU resources
func (p *GPUMemoryPool) Release(ptr unsafe.Pointer) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	block, exists := p.allocatedPtrs[ptr]
	if !exists {
		return fmt.Errorf("attempting to release unallocated pointer")
	}
	
	// Remove from free blocks if present
	if blocks, exists := p.freeBlocks[block.size]; exists {
		for i, freeBlock := range blocks {
			if freeBlock.ptr == ptr {
				p.freeBlocks[block.size] = append(blocks[:i], blocks[i+1:]...)
				break
			}
		}
	}
	
	// Free GPU memory
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	
	var cErr C.CError
	retCode := C.free_gpu_memory(
		C.GPUPtr(ptr),
		&cErr,
	)
	
	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU memory release failed (code %d): %s", retCode, errMsg)
	}
	
	delete(p.allocatedPtrs, ptr)
	p.currentUsage -= block.size
	p.stats.TotalFreed += block.size
	
	return nil
}

// garbageCollect frees unused memory blocks
func (p *GPUMemoryPool) garbageCollect() error {
	now := time.Now()
	threshold := 5 * time.Minute // Free blocks unused for 5 minutes
	
	var toRelease []unsafe.Pointer
	
	for size, blocks := range p.freeBlocks {
		var keepBlocks []*MemoryBlock
		
		for _, block := range blocks {
			if !block.inUse && now.Sub(block.lastUsed) > threshold {
				toRelease = append(toRelease, block.ptr)
			} else {
				keepBlocks = append(keepBlocks, block)
			}
		}
		
		p.freeBlocks[size] = keepBlocks
	}
	
	// Release old blocks
	for _, ptr := range toRelease {
		err := p.Release(ptr)
		if err != nil {
			return fmt.Errorf("failed to release memory during GC: %w", err)
		}
	}
	
	// Update fragmentation ratio
	p.updateFragmentationRatio()
	
	return nil
}

// updateFragmentationRatio calculates memory fragmentation
func (p *GPUMemoryPool) updateFragmentationRatio() {
	if p.currentUsage == 0 {
		p.stats.FragmentationRatio = 0.0
		return
	}
	
	// Count free blocks
	freeBlockCount := 0
	freeMemory := int64(0)
	
	for _, blocks := range p.freeBlocks {
		freeBlockCount += len(blocks)
		for _, block := range blocks {
			freeMemory += block.size
		}
	}
	
	if freeBlockCount > 1 && freeMemory > 0 {
		// Simple fragmentation metric: ratio of free blocks to total memory
		p.stats.FragmentationRatio = float32(freeBlockCount) / float32(p.currentUsage/1024) // per KB
	} else {
		p.stats.FragmentationRatio = 0.0
	}
}

// GetUsage returns current memory usage
func (p *GPUMemoryPool) GetUsage() int64 {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.currentUsage
}

// GetStats returns memory pool statistics
func (p *GPUMemoryPool) GetStats() MemoryStats {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.stats
}

// ReleaseAll releases all memory pool resources
func (p *GPUMemoryPool) ReleaseAll() {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// Release all allocated blocks
	for ptr := range p.allocatedPtrs {
		p.releaseUnsafe(ptr)
	}
	
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	
	var cErr C.CError
	C.cleanup_memory_pool(&cErr)
}

// releaseUnsafe releases memory without locking (internal use)
func (p *GPUMemoryPool) releaseUnsafe(ptr unsafe.Pointer) error {
	block, exists := p.allocatedPtrs[ptr]
	if !exists {
		return fmt.Errorf("attempting to release unallocated pointer")
	}
	
	// Remove from free blocks if present
	if blocks, exists := p.freeBlocks[block.size]; exists {
		for i, freeBlock := range blocks {
			if freeBlock.ptr == ptr {
				p.freeBlocks[block.size] = append(blocks[:i], blocks[i+1:]...)
				break
			}
		}
	}
	
	// Free GPU memory
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	
	var cErr C.CError
	retCode := C.free_gpu_memory(
		C.GPUPtr(ptr),
		&cErr,
	)
	
	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU memory release failed (code %d): %s", retCode, errMsg)
	}
	
	delete(p.allocatedPtrs, ptr)
	p.currentUsage -= block.size
	p.stats.TotalFreed += block.size
	
	return nil
}

// NewTensorCache creates a new tensor cache
func NewTensorCache(maxSize int) *TensorCache {
	return &TensorCache{
		cache:   make(map[string]*CachedTensor),
		lru:     NewLRUList(),
		maxSize: maxSize,
	}
}

// Get retrieves a tensor from cache or creates a new one
func (tc *TensorCache) Get(key string, shape []int, createFn func() (*tensor.Tensor, error)) (*tensor.Tensor, error) {
	tc.mutex.Lock()
	defer tc.mutex.Unlock()
	
	if cached, exists := tc.cache[key]; exists {
		// Check if shape matches
		if len(cached.shape) == len(shape) {
			match := true
			for i, dim := range shape {
				if dim != cached.shape[i] {
					match = false
					break
				}
			}
			
			if match {
				cached.accessed = time.Now()
				cached.useCount++
				tc.lru.MoveToFront(cached.lruNode)
				tc.hits++
				return cached.tensor, nil
			}
		}
		
		// Shape mismatch, remove from cache
		tc.removeUnsafe(key)
	}
	
	// Cache miss, create new tensor
	tc.misses++
	tensor, err := createFn()
	if err != nil {
		return nil, err
	}
	
	// Add to cache if space available
	if len(tc.cache) < tc.maxSize {
		tc.putUnsafe(key, tensor, shape)
	} else {
		// Evict LRU item
		tc.evictLRU()
		tc.putUnsafe(key, tensor, shape)
	}
	
	return tensor, nil
}

// Put adds a tensor to the cache
func (tc *TensorCache) Put(key string, tensor *tensor.Tensor, shape []int) {
	tc.mutex.Lock()
	defer tc.mutex.Unlock()
	tc.putUnsafe(key, tensor, shape)
}

// putUnsafe adds a tensor to cache without locking
func (tc *TensorCache) putUnsafe(key string, tensorObj *tensor.Tensor, shape []int) {
	if len(tc.cache) >= tc.maxSize {
		tc.evictLRU()
	}
	
	size := int64(len(tensorObj.Data) * 4) // float32 = 4 bytes
	
	cached := &CachedTensor{
		tensor:   tensorObj,
		shape:    append([]int{}, shape...),
		size:     size,
		created:  time.Now(),
		accessed: time.Now(),
		useCount: 1,
	}
	
	node := tc.lru.AddToFront(key)
	cached.lruNode = node
	
	tc.cache[key] = cached
}

// evictLRU removes the least recently used item
func (tc *TensorCache) evictLRU() {
	if tc.lru.size == 0 {
		return
	}
	
	key := tc.lru.RemoveLast()
	if cached, exists := tc.cache[key]; exists {
		cached.tensor.ReleaseGPU()
		delete(tc.cache, key)
	}
}

// removeUnsafe removes an item from cache without locking
func (tc *TensorCache) removeUnsafe(key string) {
	if cached, exists := tc.cache[key]; exists {
		tc.lru.Remove(cached.lruNode)
		cached.tensor.ReleaseGPU()
		delete(tc.cache, key)
	}
}

// Clear removes all items from cache
func (tc *TensorCache) Clear() {
	tc.mutex.Lock()
	defer tc.mutex.Unlock()
	
	for _, cached := range tc.cache {
		cached.tensor.ReleaseGPU()
	}
	
	tc.cache = make(map[string]*CachedTensor)
	tc.lru = NewLRUList()
}

// GetStats returns cache statistics
func (tc *TensorCache) GetStats() (hits, misses int64, hitRatio float32) {
	tc.mutex.RLock()
	defer tc.mutex.RUnlock()
	
	total := tc.hits + tc.misses
	if total > 0 {
		hitRatio = float32(tc.hits) / float32(total)
	}
	
	return tc.hits, tc.misses, hitRatio
}

// NewLRUList creates a new LRU list
func NewLRUList() *LRUList {
	head := &LRUNode{}
	tail := &LRUNode{}
	head.next = tail
	tail.prev = head
	
	return &LRUList{
		head: head,
		tail: tail,
		size: 0,
	}
}

// AddToFront adds a node to the front of the list
func (lru *LRUList) AddToFront(key string) *LRUNode {
	node := &LRUNode{key: key}
	
	node.next = lru.head.next
	node.prev = lru.head
	lru.head.next.prev = node
	lru.head.next = node
	
	lru.size++
	return node
}

// Remove removes a node from the list
func (lru *LRUList) Remove(node *LRUNode) {
	node.prev.next = node.next
	node.next.prev = node.prev
	lru.size--
}

// MoveToFront moves a node to the front
func (lru *LRUList) MoveToFront(node *LRUNode) {
	lru.Remove(node)
	lru.head.next.prev = node
	node.next = lru.head.next
	node.prev = lru.head
	lru.head.next = node
	lru.size++
}

// RemoveLast removes and returns the key of the last node
func (lru *LRUList) RemoveLast() string {
	if lru.size == 0 {
		return ""
	}
	
	last := lru.tail.prev
	key := last.key
	lru.Remove(last)
	return key
}

// MemoryProfiler provides memory usage profiling
type MemoryProfiler struct {
	samples    []MemorySample
	interval   time.Duration
	maxSamples int
	mutex      sync.RWMutex
	active     bool
	stopChan   chan bool
}

// MemorySample represents a memory usage sample
type MemorySample struct {
	Timestamp   time.Time
	GPUUsage    int64
	CPUUsage    int64
	CacheHits   int64
	CacheMisses int64
}

// NewMemoryProfiler creates a new memory profiler
func NewMemoryProfiler(interval time.Duration, maxSamples int) *MemoryProfiler {
	return &MemoryProfiler{
		samples:    make([]MemorySample, 0, maxSamples),
		interval:   interval,
		maxSamples: maxSamples,
		stopChan:   make(chan bool),
	}
}

// Start begins memory profiling
func (mp *MemoryProfiler) Start(memPool *GPUMemoryPool, cache *TensorCache) {
	mp.mutex.Lock()
	mp.active = true
	mp.mutex.Unlock()
	
	go func() {
		ticker := time.NewTicker(mp.interval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				mp.collectSample(memPool, cache)
			case <-mp.stopChan:
				return
			}
		}
	}()
}

// Stop ends memory profiling
func (mp *MemoryProfiler) Stop() {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	if mp.active {
		mp.active = false
		mp.stopChan <- true
	}
}

// collectSample collects a memory usage sample
func (mp *MemoryProfiler) collectSample(memPool *GPUMemoryPool, cache *TensorCache) {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	hits, misses, _ := cache.GetStats()
	
	sample := MemorySample{
		Timestamp:   time.Now(),
		GPUUsage:    memPool.GetUsage(),
		CPUUsage:    int64(m.Alloc),
		CacheHits:   hits,
		CacheMisses: misses,
	}
	
	if len(mp.samples) >= mp.maxSamples {
		// Remove oldest sample
		copy(mp.samples, mp.samples[1:])
		mp.samples[len(mp.samples)-1] = sample
	} else {
		mp.samples = append(mp.samples, sample)
	}
}

// GetSamples returns all collected samples
func (mp *MemoryProfiler) GetSamples() []MemorySample {
	mp.mutex.RLock()
	defer mp.mutex.RUnlock()
	
	result := make([]MemorySample, len(mp.samples))
	copy(result, mp.samples)
	return result
}

// GetPeakUsage returns peak GPU and CPU usage
func (mp *MemoryProfiler) GetPeakUsage() (gpuPeak, cpuPeak int64) {
	mp.mutex.RLock()
	defer mp.mutex.RUnlock()
	
	for _, sample := range mp.samples {
		if sample.GPUUsage > gpuPeak {
			gpuPeak = sample.GPUUsage
		}
		if sample.CPUUsage > cpuPeak {
			cpuPeak = sample.CPUUsage
		}
	}
	
	return gpuPeak, cpuPeak
}