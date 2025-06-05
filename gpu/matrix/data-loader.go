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
	"sync"
	"time"

	"github.com/tsawler/gometal/tensor"
)

// Enhanced data loading implementations with memory efficiency and async loading

// AsyncDataLoader provides asynchronous data loading with prefetching
type AsyncDataLoader struct {
	dataset       Dataset
	config        DataLoaderConfig
	batchQueue    chan *BatchData
	prefetchQueue chan *PrefetchTask
	workers       []*DataWorker
	currentEpoch  int
	currentBatch  int
	indices       []int
	stats         DataLoaderStats
	mutex         sync.RWMutex
	stopChan      chan struct{}
	running       bool
}

// DataWorker handles async data loading
type DataWorker struct {
	id         int
	dataset    Dataset
	memPool    *GPUMemoryPool
	cache      *TensorCache
	active     bool
	loadChan   chan *LoadRequest
	resultChan chan *BatchData
	loader     *AsyncDataLoader   // Add this field
	taskChan   chan *PrefetchTask // Add this field
	stopChan   chan struct{}      // Add this field
}

// LoadRequest represents a data loading request
type LoadRequest struct {
	batchIdx   int
	indices    []int
	resultChan chan *PrefetchedBatch
}

// PrefetchedBatch represents a prefetched batch
type PrefetchedBatch struct {
	inputs   *tensor.Tensor
	targets  *tensor.Tensor
	batchIdx int
	error    error
}

// BatchData represents a loaded batch with metadata
type BatchData struct {
	inputs     *tensor.Tensor
	targets    *tensor.Tensor
	batchIndex int
	loadTime   time.Duration
	fromCache  bool
	metadata   map[string]interface{}
}

// PrefetchTask represents a prefetching task
type PrefetchTask struct {
	batchIndex int
	priority   int
	requiredBy time.Time
	retryCount int
}

// DataLoader interface for efficient data loading
type DataLoader interface {
	GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
	BatchCount() int
	Shuffle() error
	Reset() error
	SetBatchSize(batchSize int)
	GetDatasetSize() int
}

// DataLoaderConfig contains configuration for data loader
type DataLoaderConfig struct {
	BatchSize      int
	Shuffle        bool
	DropLast       bool
	NumWorkers     int
	PrefetchFactor int
	CacheSize      int
	MaxMemoryUsage int64
	MemoryPool     *GPUMemoryPool
	TensorCache    *TensorCache
}

// StreamingDataLoader provides streaming data loading for large datasets
type StreamingDataLoader struct {
	config          DataLoaderConfig
	streamProviders []StreamProvider
	bufferPool      *BufferPool
	currentStream   int
	stats           DataLoaderStats
	mutex           sync.RWMutex
}

// DataLoaderStats tracks data loading performance
type DataLoaderStats struct {
	TotalBatches      int64
	CacheHits         int64
	CacheMisses       int64
	LoadTime          time.Duration
	AverageLoadTime   time.Duration
	PrefetchHits      int64
	MemoryUsage       int64
	WorkerUtilization []float32
}

// StreamProvider interface for streaming data sources
type StreamProvider interface {
	NextBatch() (*BatchData, error)
	Reset() error
	EstimatedBatches() int
	GetStreamInfo() StreamInfo
}

// StreamInfo provides information about a data stream
type StreamInfo struct {
	Source        string
	Format        string
	Compression   string
	EstimatedSize int64
	LastModified  time.Time
}

// BufferPool manages reusable tensor buffers
type BufferPool struct {
	buffers    map[string][]*tensor.Tensor
	maxBuffers int
	mutex      sync.RWMutex
}

// DistributedDataLoader handles distributed data loading across multiple GPUs/nodes
type DistributedDataLoader struct {
	config      DataLoaderConfig
	nodeRank    int
	worldSize   int
	localLoader DataLoader
	coordinator *DistributedCoordinator
	stats       DataLoaderStats
}

// DistributedCoordinator coordinates data loading across distributed nodes
type DistributedCoordinator struct {
	nodes        []NodeInfo
	currentNode  int
	shardMapping map[int][]int // node -> shard indices
	mutex        sync.RWMutex
}

// NodeInfo represents information about a distributed node
type NodeInfo struct {
	NodeID    int
	Address   string
	GPUCount  int
	Available bool
	LastSeen  time.Time
}

// Dataset interface for data sources
type Dataset interface {
	GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error)
	Len() int
	GetShape() ([]int, []int) // input shape, target shape
}

// NewAsyncDataLoader creates a new asynchronous data loader
func NewAsyncDataLoader(dataset Dataset, config DataLoaderConfig) (*AsyncDataLoader, error) {
	if dataset == nil {
		return nil, fmt.Errorf("dataset cannot be nil")
	}

	loader := &AsyncDataLoader{
		dataset:       dataset,
		config:        config,
		batchQueue:    make(chan *BatchData, config.PrefetchFactor),
		prefetchQueue: make(chan *PrefetchTask, config.PrefetchFactor*2),
		workers:       make([]*DataWorker, config.NumWorkers),
		indices:       make([]int, dataset.Len()),
		stopChan:      make(chan struct{}),
		running:       false,
	}

	// Initialize indices
	for i := 0; i < dataset.Len(); i++ {
		loader.indices[i] = i
	}

	// Create workers
	for i := 0; i < config.NumWorkers; i++ {
		worker := &DataWorker{
			id:         i,
			loader:     loader,
			taskChan:   make(chan *PrefetchTask, 10),
			resultChan: make(chan *BatchData, 5),
			stopChan:   make(chan struct{}),
		}
		loader.workers[i] = worker
	}

	return loader, nil
}

// Start begins asynchronous data loading
func (adl *AsyncDataLoader) Start() error {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()

	if adl.running {
		return fmt.Errorf("data loader is already running")
	}

	adl.running = true

	// Start workers
	for _, worker := range adl.workers {
		go worker.run()
	}

	// Start prefetch coordinator
	go adl.runPrefetchCoordinator()

	// Start batch collector
	go adl.runBatchCollector()

	return nil
}

// Stop stops the asynchronous data loading
func (adl *AsyncDataLoader) Stop() {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()

	if !adl.running {
		return
	}

	adl.running = false
	close(adl.stopChan)

	// Stop workers
	for _, worker := range adl.workers {
		close(worker.stopChan)
	}
}

// GetBatch returns the next batch (implements DataLoader interface)
func (adl *AsyncDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	if !adl.running {
		err := adl.Start()
		if err != nil {
			return nil, nil, fmt.Errorf("failed to start async loader: %w", err)
		}
	}

	// Try to get from cache first
	if adl.config.TensorCache != nil {
		cacheKey := fmt.Sprintf("batch_%d_epoch_%d", batchIdx, adl.currentEpoch)
		if cachedBatch := adl.getCachedBatch(cacheKey); cachedBatch != nil {
			adl.updateStats(true, cachedBatch.loadTime)
			return cachedBatch.inputs, cachedBatch.targets, nil
		}
	}

	// Schedule prefetch for this batch if not already done
	adl.schedulePrefetch(batchIdx, 1)

	// Wait for batch to be ready
	select {
	case batch := <-adl.batchQueue:
		if batch.batchIndex == batchIdx {
			adl.updateStats(batch.fromCache, batch.loadTime)
			return batch.inputs, batch.targets, nil
		} else {
			// Wrong batch, put it back and try again
			adl.batchQueue <- batch
			return adl.loadBatchDirect(batchIdx)
		}
	case <-time.After(time.Second * 10):
		// Timeout, fall back to direct loading
		return adl.loadBatchDirect(batchIdx)
	}
}

// BatchCount returns the number of batches (implements DataLoader interface)
func (adl *AsyncDataLoader) BatchCount() int {
	datasetSize := adl.dataset.Len()
	batchCount := datasetSize / adl.config.BatchSize
	if adl.config.DropLast {
		return batchCount
	}
	if datasetSize%adl.config.BatchSize != 0 {
		batchCount++
	}
	return batchCount
}

// Shuffle shuffles the dataset indices (implements DataLoader interface)
func (adl *AsyncDataLoader) Shuffle() error {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()

	if adl.config.Shuffle {
		rand.Shuffle(len(adl.indices), func(i, j int) {
			adl.indices[i], adl.indices[j] = adl.indices[j], adl.indices[i]
		})
	}

	return nil
}

// Reset resets the data loader state (implements DataLoader interface)
func (adl *AsyncDataLoader) Reset() error {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()

	adl.currentBatch = 0
	adl.currentEpoch++

	// Clear batch queue
	for len(adl.batchQueue) > 0 {
		batch := <-adl.batchQueue
		batch.inputs.ReleaseGPU()
		batch.targets.ReleaseGPU()
	}

	return adl.Shuffle()
}

// SetBatchSize sets the batch size (implements DataLoader interface)
func (adl *AsyncDataLoader) SetBatchSize(batchSize int) {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()
	adl.config.BatchSize = batchSize
}

// GetDatasetSize returns the dataset size (implements DataLoader interface)
func (adl *AsyncDataLoader) GetDatasetSize() int {
	return adl.dataset.Len()
}

// GetStats returns data loader statistics
func (adl *AsyncDataLoader) GetStats() DataLoaderStats {
	adl.mutex.RLock()
	defer adl.mutex.RUnlock()
	return adl.stats
}

// Private methods for AsyncDataLoader

func (adl *AsyncDataLoader) runPrefetchCoordinator() {
	ticker := time.NewTicker(time.Millisecond * 100)
	defer ticker.Stop()

	for {
		select {
		case <-adl.stopChan:
			return
		case <-ticker.C:
			adl.coordinatePrefetching()
		case task := <-adl.prefetchQueue:
			adl.assignTaskToWorker(task)
		}
	}
}

func (adl *AsyncDataLoader) runBatchCollector() {
	for {
		select {
		case <-adl.stopChan:
			return
		default:
			// Collect results from workers
			for _, worker := range adl.workers {
				select {
				case batch := <-worker.resultChan:
					adl.batchQueue <- batch
				default:
					// No batch ready from this worker
				}
			}
			time.Sleep(time.Millisecond * 10)
		}
	}
}

func (adl *AsyncDataLoader) coordinatePrefetching() {
	// Schedule prefetching for upcoming batches
	batchCount := adl.BatchCount()

	for i := 0; i < adl.config.PrefetchFactor; i++ {
		nextBatch := (adl.currentBatch + i) % batchCount
		if !adl.isBatchInQueue(nextBatch) {
			adl.schedulePrefetch(nextBatch, 5-i) // Higher priority for closer batches
		}
	}
}

func (adl *AsyncDataLoader) schedulePrefetch(batchIdx, priority int) {
	task := &PrefetchTask{
		batchIndex: batchIdx,
		priority:   priority,
		requiredBy: time.Now().Add(time.Second * 5),
		retryCount: 0,
	}

	select {
	case adl.prefetchQueue <- task:
		// Task scheduled successfully
	default:
		// Queue full, skip this prefetch
	}
}

func (adl *AsyncDataLoader) assignTaskToWorker(task *PrefetchTask) {
	// Simple round-robin assignment
	workerID := task.batchIndex % len(adl.workers)
	worker := adl.workers[workerID]

	select {
	case worker.taskChan <- task:
		// Task assigned successfully
	default:
		// Worker busy, try next worker
		nextWorker := adl.workers[(workerID+1)%len(adl.workers)]
		select {
		case nextWorker.taskChan <- task:
			// Assigned to next worker
		default:
			// All workers busy, drop task
		}
	}
}

func (adl *AsyncDataLoader) isBatchInQueue(batchIdx int) bool {
	// Check if batch is already in queue
	tempQueue := make([]*BatchData, 0, len(adl.batchQueue))
	found := false

	// Drain queue temporarily
	for len(adl.batchQueue) > 0 {
		batch := <-adl.batchQueue
		if batch.batchIndex == batchIdx {
			found = true
		}
		tempQueue = append(tempQueue, batch)
	}

	// Restore queue
	for _, batch := range tempQueue {
		adl.batchQueue <- batch
	}

	return found
}

func (adl *AsyncDataLoader) loadBatchDirect(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	start := time.Now()

	batchSize := adl.config.BatchSize
	startIdx := batchIdx * batchSize
	endIdx := startIdx + batchSize

	if endIdx > len(adl.indices) {
		endIdx = len(adl.indices)
	}

	actualBatchSize := endIdx - startIdx
	if actualBatchSize == 0 {
		return nil, nil, fmt.Errorf("empty batch")
	}

	// Get input and target shapes
	inputShape, targetShape := adl.dataset.GetShape()

	// Create batch tensors
	batchInputShape := append([]int{actualBatchSize}, inputShape...)
	batchTargetShape := append([]int{actualBatchSize}, targetShape...)

	inputSize := actualBatchSize
	for _, dim := range inputShape {
		inputSize *= dim
	}

	targetSize := actualBatchSize
	for _, dim := range targetShape {
		targetSize *= dim
	}

	inputData := make([]float32, inputSize)
	targetData := make([]float32, targetSize)

	// Load batch data
	for i := 0; i < actualBatchSize; i++ {
		dataIdx := adl.indices[startIdx+i]

		input, target, err := adl.dataset.GetItem(dataIdx)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to get item %d: %w", dataIdx, err)
		}

		// Ensure data is on CPU for copying
		if err := input.RetrieveCPU(); err != nil {
			input.ReleaseGPU()
			target.ReleaseGPU()
			return nil, nil, fmt.Errorf("failed to retrieve input to CPU: %w", err)
		}

		if err := target.RetrieveCPU(); err != nil {
			input.ReleaseGPU()
			target.ReleaseGPU()
			return nil, nil, fmt.Errorf("failed to retrieve target to CPU: %w", err)
		}

		// Copy data
		inputOffset := i * len(input.Data)
		targetOffset := i * len(target.Data)

		copy(inputData[inputOffset:inputOffset+len(input.Data)], input.Data)
		copy(targetData[targetOffset:targetOffset+len(target.Data)], target.Data)

		// Release individual tensors
		input.ReleaseGPU()
		target.ReleaseGPU()
	}

	// Create batch tensors
	batchInputs, err := tensor.NewTensor(batchInputShape, inputData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create batch inputs: %w", err)
	}

	batchTargets, err := tensor.NewTensor(batchTargetShape, targetData)
	if err != nil {
		batchInputs.ReleaseGPU()
		return nil, nil, fmt.Errorf("failed to create batch targets: %w", err)
	}

	loadTime := time.Since(start)
	adl.updateStats(false, loadTime)

	return batchInputs, batchTargets, nil
}

func (adl *AsyncDataLoader) getCachedBatch(cacheKey string) *BatchData {
	if adl.config.TensorCache == nil {
		return nil
	}

	// Try to get from cache
	inputShape, targetShape := adl.dataset.GetShape()
	batchInputShape := append([]int{adl.config.BatchSize}, inputShape...)
	batchTargetShape := append([]int{adl.config.BatchSize}, targetShape...)

	inputs, err := adl.config.TensorCache.Get(cacheKey+"_inputs", batchInputShape, func() (*tensor.Tensor, error) {
		return nil, fmt.Errorf("not in cache")
	})
	if err != nil {
		return nil
	}

	targets, err := adl.config.TensorCache.Get(cacheKey+"_targets", batchTargetShape, func() (*tensor.Tensor, error) {
		return nil, fmt.Errorf("not in cache")
	})
	if err != nil {
		return nil
	}

	return &BatchData{
		inputs:    inputs,
		targets:   targets,
		fromCache: true,
		loadTime:  0,
	}
}

func (adl *AsyncDataLoader) updateStats(fromCache bool, loadTime time.Duration) {
	adl.mutex.Lock()
	defer adl.mutex.Unlock()

	adl.stats.TotalBatches++
	adl.stats.LoadTime += loadTime

	if adl.stats.TotalBatches > 0 {
		adl.stats.AverageLoadTime = adl.stats.LoadTime / time.Duration(adl.stats.TotalBatches)
	}

	if fromCache {
		adl.stats.CacheHits++
	} else {
		adl.stats.CacheMisses++
	}
}

// DataWorker implementation

func (dw *DataWorker) run() {
	for {
		select {
		case <-dw.stopChan:
			return
		case task := <-dw.taskChan:
			batch, err := dw.processBatch(task)
			if err != nil {
				// Retry if possible
				if task.retryCount < 3 {
					task.retryCount++
					time.Sleep(time.Millisecond * 100)
					dw.taskChan <- task
				}
				continue
			}

			// Try to send result
			select {
			case dw.resultChan <- batch:
				// Result sent successfully
			case <-time.After(time.Second):
				// Timeout, discard batch to prevent memory leak
				batch.inputs.ReleaseGPU()
				batch.targets.ReleaseGPU()
			}
		}
	}
}

func (dw *DataWorker) processBatch(task *PrefetchTask) (*BatchData, error) {
	start := time.Now()

	// Use the loader's direct batch loading method
	inputs, targets, err := dw.loader.loadBatchDirect(task.batchIndex)
	if err != nil {
		return nil, err
	}

	batch := &BatchData{
		inputs:     inputs,
		targets:    targets,
		batchIndex: task.batchIndex,
		loadTime:   time.Since(start),
		fromCache:  false,
		metadata:   make(map[string]interface{}),
	}

	return batch, nil
}

// NewStreamingDataLoader creates a new streaming data loader
func NewStreamingDataLoader(config DataLoaderConfig, providers []StreamProvider) (*StreamingDataLoader, error) {
	if len(providers) == 0 {
		return nil, fmt.Errorf("at least one stream provider is required")
	}

	loader := &StreamingDataLoader{
		config:          config,
		streamProviders: providers,
		bufferPool:      NewBufferPool(config.MaxMemoryUsage),
		currentStream:   0,
	}

	return loader, nil
}

// GetBatch returns the next batch from the stream
func (sdl *StreamingDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	provider := sdl.streamProviders[sdl.currentStream]

	batch, err := provider.NextBatch()
	if err != nil {
		// Try next stream provider
		sdl.currentStream = (sdl.currentStream + 1) % len(sdl.streamProviders)
		if sdl.currentStream == 0 {
			// All streams exhausted, reset
			for _, p := range sdl.streamProviders {
				p.Reset()
			}
		}

		provider = sdl.streamProviders[sdl.currentStream]
		batch, err = provider.NextBatch()
		if err != nil {
			return nil, nil, fmt.Errorf("all streams failed: %w", err)
		}
	}

	return batch.inputs, batch.targets, nil
}

// NewBufferPool creates a new buffer pool
func NewBufferPool(maxMemory int64) *BufferPool {
	return &BufferPool{
		buffers:    make(map[string][]*tensor.Tensor),
		maxBuffers: int(maxMemory / (1024 * 1024)), // Rough estimate
	}
}

// GetBuffer retrieves a buffer from the pool or creates a new one
func (bp *BufferPool) GetBuffer(shape []int) (*tensor.Tensor, error) {
	bp.mutex.Lock()
	defer bp.mutex.Unlock()

	key := fmt.Sprintf("%v", shape)

	if buffers, exists := bp.buffers[key]; exists && len(buffers) > 0 {
		// Reuse existing buffer
		buffer := buffers[len(buffers)-1]
		bp.buffers[key] = buffers[:len(buffers)-1]
		return buffer, nil
	}

	// Create new buffer
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	data := make([]float32, size)
	return tensor.NewTensor(shape, data)
}

// ReturnBuffer returns a buffer to the pool
func (bp *BufferPool) ReturnBuffer(buffer *tensor.Tensor) {
	bp.mutex.Lock()
	defer bp.mutex.Unlock()

	key := fmt.Sprintf("%v", buffer.Shape)

	if buffers, exists := bp.buffers[key]; exists {
		if len(buffers) < bp.maxBuffers {
			bp.buffers[key] = append(buffers, buffer)
		} else {
			// Pool full, release buffer
			buffer.ReleaseGPU()
		}
	} else {
		bp.buffers[key] = []*tensor.Tensor{buffer}
	}
}

// NewDistributedDataLoader creates a new distributed data loader
func NewDistributedDataLoader(config DataLoaderConfig, nodeRank, worldSize int, localLoader DataLoader) (*DistributedDataLoader, error) {
	if nodeRank >= worldSize || nodeRank < 0 {
		return nil, fmt.Errorf("invalid node rank %d for world size %d", nodeRank, worldSize)
	}

	coordinator := &DistributedCoordinator{
		nodes:        make([]NodeInfo, worldSize),
		currentNode:  nodeRank,
		shardMapping: make(map[int][]int),
	}

	// Initialize shard mapping (simple round-robin for now)
	totalBatches := localLoader.BatchCount()
	for i := 0; i < totalBatches; i++ {
		targetNode := i % worldSize
		coordinator.shardMapping[targetNode] = append(coordinator.shardMapping[targetNode], i)
	}

	loader := &DistributedDataLoader{
		config:      config,
		nodeRank:    nodeRank,
		worldSize:   worldSize,
		localLoader: localLoader,
		coordinator: coordinator,
	}

	return loader, nil
}

// GetBatch returns the next batch for this distributed node
func (ddl *DistributedDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	// Check if this batch belongs to this node
	myShards := ddl.coordinator.shardMapping[ddl.nodeRank]

	localBatchIdx := -1
	for i, shardIdx := range myShards {
		if shardIdx == batchIdx {
			localBatchIdx = i
			break
		}
	}

	if localBatchIdx == -1 {
		return nil, nil, fmt.Errorf("batch %d not assigned to node %d", batchIdx, ddl.nodeRank)
	}

	return ddl.localLoader.GetBatch(localBatchIdx)
}

// GetLocalBatchCount returns the number of batches for this node
func (ddl *DistributedDataLoader) GetLocalBatchCount() int {
	return len(ddl.coordinator.shardMapping[ddl.nodeRank])
}

// Utility functions

// ValidateDataLoaderConfig validates data loader configuration
func ValidateDataLoaderConfig(config DataLoaderConfig) error {
	if config.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive")
	}

	if config.NumWorkers < 0 {
		return fmt.Errorf("number of workers cannot be negative")
	}

	if config.PrefetchFactor < 0 {
		return fmt.Errorf("prefetch factor cannot be negative")
	}

	if config.MaxMemoryUsage < 0 {
		return fmt.Errorf("max memory usage cannot be negative")
	}

	return nil
}

// EstimateMemoryUsage estimates memory usage for a data loader configuration
func EstimateMemoryUsage(config DataLoaderConfig, inputShape, targetShape []int) int64 {
	inputSize := int64(config.BatchSize)
	for _, dim := range inputShape {
		inputSize *= int64(dim)
	}

	targetSize := int64(config.BatchSize)
	for _, dim := range targetShape {
		targetSize *= int64(dim)
	}

	// 4 bytes per float32
	batchSize := (inputSize + targetSize) * 4

	// Account for prefetching and caching
	totalMemory := batchSize * int64(config.PrefetchFactor)

	if config.CacheSize > 0 {
		totalMemory += batchSize * int64(config.CacheSize)
	}

	return totalMemory
}
