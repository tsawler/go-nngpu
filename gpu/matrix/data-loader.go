package matrix

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/tsawler/go-nngpu/tensor"
)

// DataLoader interface for efficient data loading
type DataLoader interface {
	GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
	BatchCount() int
	Shuffle() error
	Reset() error
	SetBatchSize(batchSize int)
	GetDatasetSize() int
}

// Dataset interface for data sources
type Dataset interface {
	GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error)
	Len() int
	GetShape() ([]int, []int) // input shape, target shape
}

// MemoryDataLoader loads data efficiently with memory optimization
type MemoryDataLoader struct {
	dataset        Dataset
	batchSize      int
	shuffle        bool
	dropLast       bool
	numWorkers     int
	prefetchFactor int
	
	// Internal state
	indices        []int
	currentEpoch   int
	batchCache     map[int]*CachedBatch
	prefetchQueue  chan *PrefetchedBatch
	memoryPool     *GPUMemoryPool
	tensorCache    *TensorCache
	
	// Async loading
	workers        []*DataWorker
	workerPool     chan *DataWorker
	loadQueue      chan *LoadRequest
	
	// Synchronization
	mutex          sync.RWMutex
	wg             sync.WaitGroup
	stopChan       chan bool
	active         bool
	
	// Statistics
	stats          DataLoaderStats
}

// CachedBatch represents a cached batch of data
type CachedBatch struct {
	inputs     *tensor.Tensor
	targets    *tensor.Tensor
	batchIdx   int
	created    time.Time
	accessed   time.Time
	useCount   int64
}

// PrefetchedBatch represents a prefetched batch
type PrefetchedBatch struct {
	inputs   *tensor.Tensor
	targets  *tensor.Tensor
	batchIdx int
	error    error
}

// DataWorker handles async data loading
type DataWorker struct {
	id         int
	dataset    Dataset
	memPool    *GPUMemoryPool
	cache      *TensorCache
	active     bool
	loadChan   chan *LoadRequest
	resultChan chan *PrefetchedBatch
}

// LoadRequest represents a data loading request
type LoadRequest struct {
	batchIdx  int
	indices   []int
	resultChan chan *PrefetchedBatch
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

// NewMemoryDataLoader creates a new memory-efficient data loader
func NewMemoryDataLoader(dataset Dataset, config DataLoaderConfig) (*MemoryDataLoader, error) {
	if dataset == nil {
		return nil, fmt.Errorf("dataset cannot be nil")
	}
	
	if config.BatchSize <= 0 {
		config.BatchSize = 32
	}
	
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	
	if config.PrefetchFactor <= 0 {
		config.PrefetchFactor = 2
	}
	
	// Create memory pool if not provided
	var memPool *GPUMemoryPool
	var err error
	if config.MemoryPool != nil {
		memPool = config.MemoryPool
	} else {
		memPool, err = NewGPUMemoryPool(config.MaxMemoryUsage)
		if err != nil {
			return nil, fmt.Errorf("failed to create memory pool: %w", err)
		}
	}
	
	// Create tensor cache if not provided
	var cache *TensorCache
	if config.TensorCache != nil {
		cache = config.TensorCache
	} else {
		cache = NewTensorCache(config.CacheSize)
	}
	
	loader := &MemoryDataLoader{
		dataset:        dataset,
		batchSize:      config.BatchSize,
		shuffle:        config.Shuffle,
		dropLast:       config.DropLast,
		numWorkers:     config.NumWorkers,
		prefetchFactor: config.PrefetchFactor,
		batchCache:     make(map[int]*CachedBatch),
		prefetchQueue:  make(chan *PrefetchedBatch, config.NumWorkers*config.PrefetchFactor),
		memoryPool:     memPool,
		tensorCache:    cache,
		workerPool:     make(chan *DataWorker, config.NumWorkers),
		loadQueue:      make(chan *LoadRequest, config.NumWorkers*2),
		stopChan:       make(chan bool),
		stats:          DataLoaderStats{
			WorkerUtilization: make([]float32, config.NumWorkers),
		},
	}
	
	// Initialize indices
	err = loader.initializeIndices()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize indices: %w", err)
	}
	
	// Start workers
	err = loader.startWorkers()
	if err != nil {
		return nil, fmt.Errorf("failed to start workers: %w", err)
	}
	
	return loader, nil
}

// DataLoaderConfig contains configuration for data loader
type DataLoaderConfig struct {
	BatchSize       int
	Shuffle         bool
	DropLast        bool
	NumWorkers      int
	PrefetchFactor  int
	CacheSize       int
	MaxMemoryUsage  int64
	MemoryPool      *GPUMemoryPool
	TensorCache     *TensorCache
}

// GetBatch returns a batch of data
func (dl *MemoryDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	startTime := time.Now()
	defer func() {
		dl.stats.LoadTime += time.Since(startTime)
		dl.stats.TotalBatches++
		if dl.stats.TotalBatches > 0 {
			dl.stats.AverageLoadTime = dl.stats.LoadTime / time.Duration(dl.stats.TotalBatches)
		}
	}()
	
	// Check cache first
	dl.mutex.RLock()
	if cached, exists := dl.batchCache[batchIdx]; exists {
		cached.accessed = time.Now()
		cached.useCount++
		dl.mutex.RUnlock()
		dl.stats.CacheHits++
		return cached.inputs, cached.targets, nil
	}
	dl.mutex.RUnlock()
	
	// Check prefetch queue
	select {
	case prefetched := <-dl.prefetchQueue:
		if prefetched.batchIdx == batchIdx {
			dl.stats.PrefetchHits++
			if prefetched.error != nil {
				return nil, nil, prefetched.error
			}
			
			// Cache the batch
			dl.cacheBatch(batchIdx, prefetched.inputs, prefetched.targets)
			return prefetched.inputs, prefetched.targets, nil
		}
		// Put it back if it's a different batch
		dl.prefetchQueue <- prefetched
	default:
		// No prefetched batch available
	}
	
	// Load synchronously
	dl.stats.CacheMisses++
	return dl.loadBatchSync(batchIdx)
}

// loadBatchSync loads a batch synchronously
func (dl *MemoryDataLoader) loadBatchSync(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	batchIndices := dl.getBatchIndices(batchIdx)
	if len(batchIndices) == 0 {
		return nil, nil, fmt.Errorf("invalid batch index: %d", batchIdx)
	}
	
	// Get sample shapes
	inputShape, targetShape := dl.dataset.GetShape()
	
	// Calculate batch shapes
	batchInputShape := append([]int{len(batchIndices)}, inputShape...)
	batchTargetShape := append([]int{len(batchIndices)}, targetShape...)
	
	// Allocate batch tensors
	inputSize := 1
	for _, dim := range batchInputShape {
		inputSize *= dim
	}
	targetSize := 1
	for _, dim := range batchTargetShape {
		targetSize *= dim
	}
	
	inputData := make([]float32, inputSize)
	targetData := make([]float32, targetSize)
	
	batchInputs, err := tensor.NewTensor(batchInputShape, inputData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create batch input tensor: %w", err)
	}
	
	batchTargets, err := tensor.NewTensor(batchTargetShape, targetData)
	if err != nil {
		batchInputs.ReleaseGPU()
		return nil, nil, fmt.Errorf("failed to create batch target tensor: %w", err)
	}
	
	// Load individual samples
	inputOffset := 0
	targetOffset := 0
	inputStride := inputSize / len(batchIndices)
	targetStride := targetSize / len(batchIndices)
	
	for _, idx := range batchIndices {
		input, target, err := dl.dataset.GetItem(idx)
		if err != nil {
			batchInputs.ReleaseGPU()
			batchTargets.ReleaseGPU()
			return nil, nil, fmt.Errorf("failed to get item %d: %w", idx, err)
		}
		
		// Copy input data
		if err := input.RetrieveCPU(); err != nil {
			input.ReleaseGPU()
			target.ReleaseGPU()
			batchInputs.ReleaseGPU()
			batchTargets.ReleaseGPU()
			return nil, nil, fmt.Errorf("failed to retrieve input to CPU: %w", err)
		}
		
		copy(batchInputs.Data[inputOffset:inputOffset+len(input.Data)], input.Data)
		inputOffset += inputStride
		
		// Copy target data
		if err := target.RetrieveCPU(); err != nil {
			input.ReleaseGPU()
			target.ReleaseGPU()
			batchInputs.ReleaseGPU()
			batchTargets.ReleaseGPU()
			return nil, nil, fmt.Errorf("failed to retrieve target to CPU: %w", err)
		}
		
		copy(batchTargets.Data[targetOffset:targetOffset+len(target.Data)], target.Data)
		targetOffset += targetStride
		
		// Release individual tensors
		input.ReleaseGPU()
		target.ReleaseGPU()
	}
	
	// Move to GPU
	if err := batchInputs.EnsureGPU(); err != nil {
		batchInputs.ReleaseGPU()
		batchTargets.ReleaseGPU()
		return nil, nil, fmt.Errorf("failed to move batch inputs to GPU: %w", err)
	}
	
	if err := batchTargets.EnsureGPU(); err != nil {
		batchInputs.ReleaseGPU()
		batchTargets.ReleaseGPU()
		return nil, nil, fmt.Errorf("failed to move batch targets to GPU: %w", err)
	}
	
	// Cache the batch
	dl.cacheBatch(batchIdx, batchInputs, batchTargets)
	
	return batchInputs, batchTargets, nil
}

// getBatchIndices returns the indices for a specific batch
func (dl *MemoryDataLoader) getBatchIndices(batchIdx int) []int {
	dl.mutex.RLock()
	defer dl.mutex.RUnlock()
	
	start := batchIdx * dl.batchSize
	end := start + dl.batchSize
	
	if start >= len(dl.indices) {
		return nil
	}
	
	if end > len(dl.indices) {
		if dl.dropLast {
			return nil
		}
		end = len(dl.indices)
	}
	
	return dl.indices[start:end]
}

// cacheBatch adds a batch to the cache
func (dl *MemoryDataLoader) cacheBatch(batchIdx int, inputs, targets *tensor.Tensor) {
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	
	// Check cache size limit
	maxCacheSize := 100 // Maximum number of cached batches
	if len(dl.batchCache) >= maxCacheSize {
		// Remove oldest cached batch
		var oldestIdx int
		var oldestTime time.Time = time.Now()
		
		for idx, cached := range dl.batchCache {
			if cached.accessed.Before(oldestTime) {
				oldestTime = cached.accessed
				oldestIdx = idx
			}
		}
		
		if oldest, exists := dl.batchCache[oldestIdx]; exists {
			oldest.inputs.ReleaseGPU()
			oldest.targets.ReleaseGPU()
			delete(dl.batchCache, oldestIdx)
		}
	}
	
	dl.batchCache[batchIdx] = &CachedBatch{
		inputs:   inputs,
		targets:  targets,
		batchIdx: batchIdx,
		created:  time.Now(),
		accessed: time.Now(),
		useCount: 1,
	}
}

// BatchCount returns the number of batches in the dataset
func (dl *MemoryDataLoader) BatchCount() int {
	dl.mutex.RLock()
	defer dl.mutex.RUnlock()
	
	datasetSize := len(dl.indices)
	if dl.dropLast {
		return datasetSize / dl.batchSize
	}
	return (datasetSize + dl.batchSize - 1) / dl.batchSize
}

// Shuffle shuffles the dataset indices
func (dl *MemoryDataLoader) Shuffle() error {
	if !dl.shuffle {
		return nil
	}
	
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	
	// Clear cache when shuffling
	for _, cached := range dl.batchCache {
		cached.inputs.ReleaseGPU()
		cached.targets.ReleaseGPU()
	}
	dl.batchCache = make(map[int]*CachedBatch)
	
	// Shuffle indices
	rand.Seed(time.Now().UnixNano())
	for i := len(dl.indices) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
	}
	
	dl.currentEpoch++
	return nil
}

// Reset resets the data loader state
func (dl *MemoryDataLoader) Reset() error {
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	
	// Clear cache
	for _, cached := range dl.batchCache {
		cached.inputs.ReleaseGPU()
		cached.targets.ReleaseGPU()
	}
	dl.batchCache = make(map[int]*CachedBatch)
	
	// Clear prefetch queue
	for len(dl.prefetchQueue) > 0 {
		prefetched := <-dl.prefetchQueue
		if prefetched.inputs != nil {
			prefetched.inputs.ReleaseGPU()
		}
		if prefetched.targets != nil {
			prefetched.targets.ReleaseGPU()
		}
	}
	
	return dl.initializeIndices()
}

// SetBatchSize sets the batch size
func (dl *MemoryDataLoader) SetBatchSize(batchSize int) {
	dl.mutex.Lock()
	defer dl.mutex.Unlock()
	
	dl.batchSize = batchSize
	
	// Clear cache since batch structure changed
	for _, cached := range dl.batchCache {
		cached.inputs.ReleaseGPU()
		cached.targets.ReleaseGPU()
	}
	dl.batchCache = make(map[int]*CachedBatch)
}

// GetDatasetSize returns the size of the dataset
func (dl *MemoryDataLoader) GetDatasetSize() int {
	return dl.dataset.Len()
}

// initializeIndices creates the initial index sequence
func (dl *MemoryDataLoader) initializeIndices() error {
	datasetSize := dl.dataset.Len()
	dl.indices = make([]int, datasetSize)
	for i := 0; i < datasetSize; i++ {
		dl.indices[i] = i
	}
	
	if dl.shuffle {
		rand.Seed(time.Now().UnixNano())
		for i := len(dl.indices) - 1; i > 0; i-- {
			j := rand.Intn(i + 1)
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		}
	}
	
	return nil
}

// startWorkers starts the background worker goroutines
func (dl *MemoryDataLoader) startWorkers() error {
	dl.workers = make([]*DataWorker, dl.numWorkers)
	
	for i := 0; i < dl.numWorkers; i++ {
		worker := &DataWorker{
			id:         i,
			dataset:    dl.dataset,
			memPool:    dl.memoryPool,
			cache:      dl.tensorCache,
			active:     true,
			loadChan:   make(chan *LoadRequest, 1),
			resultChan: make(chan *PrefetchedBatch, 1),
		}
		
		dl.workers[i] = worker
		dl.workerPool <- worker
		
		dl.wg.Add(1)
		go dl.runWorker(worker)
	}
	
	dl.active = true
	
	// Start prefetch manager
	dl.wg.Add(1)
	go dl.runPrefetchManager()
	
	return nil
}

// runWorker runs a data loading worker
func (dl *MemoryDataLoader) runWorker(worker *DataWorker) {
	defer dl.wg.Done()
	
	for {
		select {
		case req := <-dl.loadQueue:
			if req == nil {
				return
			}
			
			startTime := time.Now()
			inputs, targets, err := dl.loadBatchSync(req.batchIdx)
			loadTime := time.Since(startTime)
			
			result := &PrefetchedBatch{
				inputs:   inputs,
				targets:  targets,
				batchIdx: req.batchIdx,
				error:    err,
			}
			
			// Update worker utilization
			dl.mutex.Lock()
			if worker.id < len(dl.stats.WorkerUtilization) {
				// Simple utilization metric based on load time
				utilization := float32(loadTime.Nanoseconds()) / float32(time.Millisecond.Nanoseconds())
				dl.stats.WorkerUtilization[worker.id] = utilization
			}
			dl.mutex.Unlock()
			
			select {
			case req.resultChan <- result:
			case <-dl.stopChan:
				return
			}
			
		case <-dl.stopChan:
			return
		}
	}
}

// runPrefetchManager manages prefetching of batches
func (dl *MemoryDataLoader) runPrefetchManager() {
	defer dl.wg.Done()
	
	for {
		select {
		case <-dl.stopChan:
			return
		default:
			// Check if we need to prefetch more batches
			if len(dl.prefetchQueue) < dl.prefetchFactor {
				// Find next batch to prefetch
				batchCount := dl.BatchCount()
				for batchIdx := 0; batchIdx < batchCount; batchIdx++ {
					// Check if already cached or in prefetch queue
					dl.mutex.RLock()
					_, cached := dl.batchCache[batchIdx]
					dl.mutex.RUnlock()
					
					if !cached && !dl.isInPrefetchQueue(batchIdx) {
						// Request prefetch
						resultChan := make(chan *PrefetchedBatch, 1)
						req := &LoadRequest{
							batchIdx:   batchIdx,
							indices:    dl.getBatchIndices(batchIdx),
							resultChan: resultChan,
						}
						
						select {
						case dl.loadQueue <- req:
							// Wait for result and add to prefetch queue
							go func() {
								select {
								case result := <-resultChan:
									select {
									case dl.prefetchQueue <- result:
									case <-dl.stopChan:
										// Clean up if stopping
										if result.inputs != nil {
											result.inputs.ReleaseGPU()
										}
										if result.targets != nil {
											result.targets.ReleaseGPU()
										}
									}
								case <-dl.stopChan:
									return
								}
							}()
						default:
							// Worker queue full, try again later
						}
						break
					}
				}
			}
			
			// Sleep briefly to avoid busy waiting
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// isInPrefetchQueue checks if a batch is already in the prefetch queue
func (dl *MemoryDataLoader) isInPrefetchQueue(batchIdx int) bool {
	// This is a simplified check - in practice you might want a more efficient implementation
	tempQueue := make([]*PrefetchedBatch, 0, len(dl.prefetchQueue))
	found := false
	
	// Drain queue and check
	for len(dl.prefetchQueue) > 0 {
		batch := <-dl.prefetchQueue
		if batch.batchIdx == batchIdx {
			found = true
		}
		tempQueue = append(tempQueue, batch)
	}
	
	// Restore queue
	for _, batch := range tempQueue {
		dl.prefetchQueue <- batch
	}
	
	return found
}

// GetStats returns data loader statistics
func (dl *MemoryDataLoader) GetStats() DataLoaderStats {
	dl.mutex.RLock()
	defer dl.mutex.RUnlock()
	
	stats := dl.stats
	stats.MemoryUsage = dl.memoryPool.GetUsage()
	return stats
}

// Stop stops the data loader and cleans up resources
func (dl *MemoryDataLoader) Stop() {
	dl.mutex.Lock()
	if !dl.active {
		dl.mutex.Unlock()
		return
	}
	dl.active = false
	dl.mutex.Unlock()
	
	// Signal stop to all workers
	close(dl.stopChan)
	
	// Wait for workers to finish
	dl.wg.Wait()
	
	// Clean up cache
	for _, cached := range dl.batchCache {
		cached.inputs.ReleaseGPU()
		cached.targets.ReleaseGPU()
	}
	
	// Clean up prefetch queue
	for len(dl.prefetchQueue) > 0 {
		prefetched := <-dl.prefetchQueue
		if prefetched.inputs != nil {
			prefetched.inputs.ReleaseGPU()
		}
		if prefetched.targets != nil {
			prefetched.targets.ReleaseGPU()
		}
	}
}

// InMemoryDataset is a simple dataset that holds all data in memory
type InMemoryDataset struct {
	inputs  []*tensor.Tensor
	targets []*tensor.Tensor
	inputShape  []int
	targetShape []int
}

// NewInMemoryDataset creates a new in-memory dataset
func NewInMemoryDataset(inputs, targets []*tensor.Tensor) (*InMemoryDataset, error) {
	if len(inputs) != len(targets) {
		return nil, fmt.Errorf("inputs and targets must have the same length")
	}
	
	if len(inputs) == 0 {
		return nil, fmt.Errorf("dataset cannot be empty")
	}
	
	return &InMemoryDataset{
		inputs:      inputs,
		targets:     targets,
		inputShape:  inputs[0].Shape,
		targetShape: targets[0].Shape,
	}, nil
}

// GetItem returns a single item from the dataset
func (ds *InMemoryDataset) GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error) {
	if index < 0 || index >= len(ds.inputs) {
		return nil, nil, fmt.Errorf("index out of bounds: %d", index)
	}
	
	return ds.inputs[index], ds.targets[index], nil
}

// Len returns the size of the dataset
func (ds *InMemoryDataset) Len() int {
	return len(ds.inputs)
}

// GetShape returns the shape of input and target tensors
func (ds *InMemoryDataset) GetShape() ([]int, []int) {
	return ds.inputShape, ds.targetShape
}

// FileDataset loads data from files on demand
type FileDataset struct {
	inputPaths  []string
	targetPaths []string
	inputShape  []int
	targetShape []int
	transform   func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error)
}

// NewFileDataset creates a new file-based dataset
func NewFileDataset(inputPaths, targetPaths []string, inputShape, targetShape []int) (*FileDataset, error) {
	if len(inputPaths) != len(targetPaths) {
		return nil, fmt.Errorf("input and target paths must have the same length")
	}
	
	return &FileDataset{
		inputPaths:  inputPaths,
		targetPaths: targetPaths,
		inputShape:  inputShape,
		targetShape: targetShape,
	}, nil
}

// SetTransform sets a transformation function for the data
func (ds *FileDataset) SetTransform(transform func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error)) {
	ds.transform = transform
}

// GetItem loads and returns a single item from files
func (ds *FileDataset) GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error) {
	if index < 0 || index >= len(ds.inputPaths) {
		return nil, nil, fmt.Errorf("index out of bounds: %d", index)
	}
	
	// Load input (placeholder - implement based on your file format)
	input, err := ds.loadTensorFromFile(ds.inputPaths[index], ds.inputShape)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load input: %w", err)
	}
	
	// Load target (placeholder - implement based on your file format)
	target, err := ds.loadTensorFromFile(ds.targetPaths[index], ds.targetShape)
	if err != nil {
		input.ReleaseGPU()
		return nil, nil, fmt.Errorf("failed to load target: %w", err)
	}
	
	// Apply transform if set
	if ds.transform != nil {
		transformedInput, transformedTarget, err := ds.transform(input, target)
		if err != nil {
			input.ReleaseGPU()
			target.ReleaseGPU()
			return nil, nil, fmt.Errorf("transform failed: %w", err)
		}
		
		// Release original tensors if they were replaced
		if transformedInput != input {
			input.ReleaseGPU()
		}
		if transformedTarget != target {
			target.ReleaseGPU()
		}
		
		return transformedInput, transformedTarget, nil
	}
	
	return input, target, nil
}

// loadTensorFromFile loads a tensor from a file (placeholder implementation)
func (ds *FileDataset) loadTensorFromFile(path string, shape []int) (*tensor.Tensor, error) {
	// This is a placeholder implementation
	// In practice, you would implement file loading based on your data format
	// (e.g., binary files, images, etc.)
	
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]float32, size)
	// Load actual data from file here
	
	return tensor.NewTensor(shape, data)
}

// Len returns the size of the dataset
func (ds *FileDataset) Len() int {
	return len(ds.inputPaths)
}

// GetShape returns the shape of input and target tensors
func (ds *FileDataset) GetShape() ([]int, []int) {
	return ds.inputShape, ds.targetShape
}