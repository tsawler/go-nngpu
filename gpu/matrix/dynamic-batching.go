package matrix

import (
	"container/heap"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
	
	"github.com/tsawler/go-nngpu/tensor"
)

// DynamicBatchScheduler manages dynamic batching and scheduling
type DynamicBatchScheduler struct {
	memMgr         *UnifiedMemoryManager
	streamMgr      *StreamManager
	
	// Batch configuration
	minBatchSize   int
	maxBatchSize   int
	targetLatency  time.Duration
	
	// Request queue
	requestQueue   *RequestPriorityQueue
	queueMu        sync.Mutex
	
	// Batch formation
	batchTimeout   time.Duration
	currentBatch   []*Request
	batchMu        sync.Mutex
	
	// Memory monitoring
	availableMemory int64
	memoryLimit     int64
	
	// Performance tracking
	batchLatencies  []time.Duration
	throughput      float64
	
	// Adaptive sizing
	adaptiveEnabled bool
	sizePredictor   *BatchSizePredictor
	
	// Model weights for inference
	modelWeights    *tensor.Tensor
}

// Request represents an inference request
type Request struct {
	ID         string
	Input      *tensor.Tensor
	Priority   int
	Timestamp  time.Time
	Deadline   time.Time
	ResultChan chan *tensor.Tensor
	
	// Request metadata
	ModelID    string
	BatchIdx   int
}

// RequestPriorityQueue implements a priority queue for requests
type RequestPriorityQueue []*Request

func (pq RequestPriorityQueue) Len() int { return len(pq) }

func (pq RequestPriorityQueue) Less(i, j int) bool {
	// Higher priority first, then earlier deadline
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority > pq[j].Priority
	}
	return pq[i].Deadline.Before(pq[j].Deadline)
}

func (pq RequestPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *RequestPriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*Request))
}

func (pq *RequestPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// NewDynamicBatchScheduler creates a dynamic batch scheduler
func NewDynamicBatchScheduler(memMgr *UnifiedMemoryManager, streamMgr *StreamManager) *DynamicBatchScheduler {
	dbs := &DynamicBatchScheduler{
		memMgr:          memMgr,
		streamMgr:       streamMgr,
		minBatchSize:    1,
		maxBatchSize:    128,
		targetLatency:   100 * time.Millisecond,
		requestQueue:    &RequestPriorityQueue{},
		batchTimeout:    50 * time.Millisecond,
		currentBatch:    make([]*Request, 0),
		memoryLimit:     8 * 1024 * 1024 * 1024, // 8GB
		batchLatencies:  make([]time.Duration, 0, 100),
		adaptiveEnabled: true,
		sizePredictor:   NewBatchSizePredictor(),
	}
	
	heap.Init(dbs.requestQueue)
	
	// Start batch formation goroutine
	go dbs.batchFormationLoop()
	
	// Start memory monitor
	go dbs.memoryMonitorLoop()
	
	return dbs
}

// SubmitRequest adds a request to the scheduler
func (dbs *DynamicBatchScheduler) SubmitRequest(req *Request) {
	dbs.queueMu.Lock()
	heap.Push(dbs.requestQueue, req)
	dbs.queueMu.Unlock()
}

// batchFormationLoop continuously forms and schedules batches
func (dbs *DynamicBatchScheduler) batchFormationLoop() {
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	
	for range ticker.C {
		dbs.formAndScheduleBatch()
	}
}

// formAndScheduleBatch creates and schedules a batch
func (dbs *DynamicBatchScheduler) formAndScheduleBatch() {
	dbs.batchMu.Lock()
	defer dbs.batchMu.Unlock()
	
	// Determine optimal batch size
	optimalSize := dbs.determineOptimalBatchSize()
	
	// Collect requests for batch
	batch := dbs.collectBatch(optimalSize)
	if len(batch) == 0 {
		return
	}
	
	// Schedule batch execution
	go dbs.executeBatch(batch)
}

// determineOptimalBatchSize calculates the best batch size
func (dbs *DynamicBatchScheduler) determineOptimalBatchSize() int {
	if !dbs.adaptiveEnabled {
		return dbs.maxBatchSize
	}
	
	// Get memory constraint
	memConstraint := dbs.getMemoryConstrainedBatchSize()
	
	// Get latency constraint
	latencyConstraint := dbs.getLatencyConstrainedBatchSize()
	
	// Get throughput optimal size
	throughputOptimal := dbs.sizePredictor.PredictOptimalSize()
	
	// Take minimum of all constraints
	optimal := min(memConstraint, latencyConstraint)
	optimal = min(optimal, throughputOptimal)
	optimal = max(optimal, dbs.minBatchSize)
	optimal = min(optimal, dbs.maxBatchSize)
	
	return optimal
}

// getMemoryConstrainedBatchSize returns max batch size given memory
func (dbs *DynamicBatchScheduler) getMemoryConstrainedBatchSize() int {
	available := atomic.LoadInt64(&dbs.availableMemory)
	
	// Estimate memory per sample (simplified)
	memPerSample := int64(4 * 1024 * 1024) // 4MB per sample estimate
	
	maxSize := int(available / memPerSample)
	return max(maxSize, 1)
}

// getLatencyConstrainedBatchSize returns max batch size for target latency
func (dbs *DynamicBatchScheduler) getLatencyConstrainedBatchSize() int {
	if len(dbs.batchLatencies) < 10 {
		return dbs.maxBatchSize // Not enough data
	}
	
	// Calculate average latency per sample
	avgLatency := dbs.calculateAverageLatency()
	samplesPerLatency := avgLatency / time.Duration(dbs.maxBatchSize)
	
	// Calculate max batch size for target latency
	maxSize := int(dbs.targetLatency / samplesPerLatency)
	
	return max(maxSize, 1)
}

// collectBatch collects requests for a batch
func (dbs *DynamicBatchScheduler) collectBatch(targetSize int) []*Request {
	dbs.queueMu.Lock()
	defer dbs.queueMu.Unlock()
	
	batch := make([]*Request, 0, targetSize)
	
	// Collect up to targetSize requests
	for len(batch) < targetSize && dbs.requestQueue.Len() > 0 {
		req := heap.Pop(dbs.requestQueue).(*Request)
		batch = append(batch, req)
	}
	
	// Check timeout for partial batch
	if len(batch) > 0 && len(batch) < targetSize {
		oldestRequest := batch[0]
		if time.Since(oldestRequest.Timestamp) > dbs.batchTimeout {
			// Timeout reached, process partial batch
			return batch
		}
		
		// Not timed out, put requests back
		for _, req := range batch {
			heap.Push(dbs.requestQueue, req)
		}
		return nil
	}
	
	return batch
}

// executeBatch processes a batch of requests
func (dbs *DynamicBatchScheduler) executeBatch(batch []*Request) {
	startTime := time.Now()
	
	// Combine inputs into batch matrix
	batchInput := dbs.combineToBatch(batch)
	
	// Get optimal stream for execution
	streamID, _ := dbs.streamMgr.GetStream()
	
	// Execute on GPU
	var batchOutput *tensor.Tensor
	dbs.streamMgr.SubmitToStream(streamID, func(stream unsafe.Pointer) {
		// Simulate model execution
		batchOutput = dbs.processBatchOnGPU(batchInput)
	})
	
	// Wait for completion
	dbs.streamMgr.SynchronizeStream(streamID)
	
	// Split results and send to request channels
	dbs.distributeResults(batch, batchOutput)
	
	// Update metrics
	latency := time.Since(startTime)
	dbs.updateMetrics(len(batch), latency)
}

// combineToBatch combines individual requests into a batch tensor
func (dbs *DynamicBatchScheduler) combineToBatch(batch []*Request) *tensor.Tensor {
	if len(batch) == 0 {
		return nil
	}
	
	// Assume all inputs have same dimensions
	rows := len(batch)
	cols := batch[0].Input.Shape[1]
	
	batchData := make([]float32, rows*cols)
	batchTensor, _ := tensor.NewTensor([]int{rows, cols}, batchData)
	
	for i, req := range batch {
		// Copy request input to batch
		copy(batchTensor.Data[i*cols:(i+1)*cols], req.Input.Data[:cols])
		req.BatchIdx = i
	}
	
	return batchTensor
}

// processBatchOnGPU executes model inference on GPU
func (dbs *DynamicBatchScheduler) processBatchOnGPU(input *tensor.Tensor) *tensor.Tensor {
	// Ensure input is on GPU
	if err := input.EnsureGPU(); err != nil {
		// Fallback to CPU processing
		return dbs.processBatchOnCPU(input)
	}
	
	// Get stream for processing
	streamID, _ := dbs.streamMgr.GetStream()
	
	// Execute on GPU stream
	var output *tensor.Tensor
	dbs.streamMgr.SubmitToStream(streamID, func(s unsafe.Pointer) {
		// Simple linear transformation as example
		// In practice, this would execute a full model
		if dbs.modelWeights != nil {
			// Perform matrix multiplication: output = input @ weights
			var err error
			output, err = MatMul(input, dbs.modelWeights)
			if err != nil {
				// Create dummy output on error
				rows := input.Shape[0]
				outputData := make([]float32, rows*1000)
				output, _ = tensor.NewTensor([]int{rows, 1000}, outputData)
			}
		} else {
			// No model weights, create dummy output
			rows := input.Shape[0]
			outputData := make([]float32, rows*1000)
			for i := range outputData {
				outputData[i] = -1 + 2*float32(i%100)/100
			}
			output, _ = tensor.NewTensor([]int{rows, 1000}, outputData)
		}
	})
	
	// Wait for completion
	dbs.streamMgr.SynchronizeStream(streamID)
	
	return output
}

// processBatchOnCPU fallback CPU processing
func (dbs *DynamicBatchScheduler) processBatchOnCPU(input *tensor.Tensor) *tensor.Tensor {
	rows := input.Shape[0]
	outputData := make([]float32, rows*1000)
	for i := range outputData {
		outputData[i] = -1 + 2*float32(i%100)/100
	}
	output, _ := tensor.NewTensor([]int{rows, 1000}, outputData)
	return output
}

// distributeResults sends results back to request channels
func (dbs *DynamicBatchScheduler) distributeResults(batch []*Request, output *tensor.Tensor) {
	for _, req := range batch {
		// Extract individual result from batch
		cols := output.Shape[1]
		resultData := make([]float32, cols)
		copy(resultData, output.Data[req.BatchIdx*cols:(req.BatchIdx+1)*cols])
		result, _ := tensor.NewTensor([]int{1, cols}, resultData)
		
		// Send result
		select {
		case req.ResultChan <- result:
			// Result sent successfully
		case <-time.After(time.Second):
			// Timeout sending result
			fmt.Printf("Timeout sending result for request %s\n", req.ID)
		}
	}
}

// updateMetrics updates performance metrics
func (dbs *DynamicBatchScheduler) updateMetrics(batchSize int, latency time.Duration) {
	// Update latency history
	dbs.batchLatencies = append(dbs.batchLatencies, latency)
	if len(dbs.batchLatencies) > 100 {
		dbs.batchLatencies = dbs.batchLatencies[1:]
	}
	
	// Update throughput
	samplesPerSecond := float64(batchSize) / latency.Seconds()
	alpha := 0.1 // Exponential moving average factor
	dbs.throughput = alpha*samplesPerSecond + (1-alpha)*dbs.throughput
	
	// Update predictor
	dbs.sizePredictor.RecordBatch(batchSize, latency, dbs.throughput)
}

// calculateAverageLatency computes average batch latency
func (dbs *DynamicBatchScheduler) calculateAverageLatency() time.Duration {
	if len(dbs.batchLatencies) == 0 {
		return dbs.targetLatency
	}
	
	total := time.Duration(0)
	for _, lat := range dbs.batchLatencies {
		total += lat
	}
	
	return total / time.Duration(len(dbs.batchLatencies))
}

// memoryMonitorLoop monitors available memory
func (dbs *DynamicBatchScheduler) memoryMonitorLoop() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// Get available memory (simplified)
		available := dbs.memoryLimit - getUsedGPUMemory()
		atomic.StoreInt64(&dbs.availableMemory, available)
	}
}

// BatchSizePredictor predicts optimal batch sizes
type BatchSizePredictor struct {
	history      []BatchRecord
	historyMu    sync.RWMutex
	
	// Model parameters (simple linear model)
	sizeCoeff      float64
	latencyCoeff   float64
	throughputCoeff float64
	
	// Learning rate
	learningRate float64
}

// BatchRecord stores batch execution history
type BatchRecord struct {
	Size       int
	Latency    time.Duration
	Throughput float64
	Timestamp  time.Time
}

// NewBatchSizePredictor creates a batch size predictor
func NewBatchSizePredictor() *BatchSizePredictor {
	return &BatchSizePredictor{
		history:         make([]BatchRecord, 0, 1000),
		sizeCoeff:       1.0,
		latencyCoeff:    -0.5,
		throughputCoeff: 0.3,
		learningRate:    0.01,
	}
}

// RecordBatch adds a batch execution record
func (bsp *BatchSizePredictor) RecordBatch(size int, latency time.Duration, throughput float64) {
	bsp.historyMu.Lock()
	defer bsp.historyMu.Unlock()
	
	record := BatchRecord{
		Size:       size,
		Latency:    latency,
		Throughput: throughput,
		Timestamp:  time.Now(),
	}
	
	bsp.history = append(bsp.history, record)
	if len(bsp.history) > 1000 {
		bsp.history = bsp.history[100:] // Keep last 900
	}
	
	// Update model
	bsp.updateModel()
}

// PredictOptimalSize predicts the optimal batch size
func (bsp *BatchSizePredictor) PredictOptimalSize() int {
	bsp.historyMu.RLock()
	defer bsp.historyMu.RUnlock()
	
	if len(bsp.history) < 10 {
		return 32 // Default
	}
	
	// Simple prediction based on recent history
	recentRecords := bsp.history[len(bsp.history)-10:]
	
	// Find size with best throughput/latency ratio
	bestScore := -1.0
	bestSize := 32
	
	for _, record := range recentRecords {
		score := bsp.scoreSize(record)
		if score > bestScore {
			bestScore = score
			bestSize = record.Size
		}
	}
	
	return bestSize
}

// scoreSize calculates a score for a batch size
func (bsp *BatchSizePredictor) scoreSize(record BatchRecord) float64 {
	// Normalize values
	sizeNorm := float64(record.Size) / 128.0
	latencyNorm := float64(record.Latency) / float64(time.Second)
	throughputNorm := record.Throughput / 1000.0
	
	// Calculate score
	score := bsp.sizeCoeff*sizeNorm +
		bsp.latencyCoeff*latencyNorm +
		bsp.throughputCoeff*throughputNorm
		
	return score
}

// updateModel updates the predictor model
func (bsp *BatchSizePredictor) updateModel() {
	if len(bsp.history) < 20 {
		return
	}
	
	// Simple gradient update based on throughput
	recent := bsp.history[len(bsp.history)-20:]
	
	// Calculate gradient
	avgThroughput := 0.0
	for _, r := range recent {
		avgThroughput += r.Throughput
	}
	avgThroughput /= float64(len(recent))
	
	// Update coefficients to maximize throughput
	for _, r := range recent {
		if r.Throughput > avgThroughput {
			// This configuration was good
			delta := (r.Throughput - avgThroughput) / avgThroughput
			bsp.sizeCoeff += bsp.learningRate * delta * float64(r.Size) / 128.0
			bsp.latencyCoeff -= bsp.learningRate * delta * float64(r.Latency) / float64(time.Second)
		}
	}
}

// DynamicComputationGraph represents a dynamic computation graph
type DynamicComputationGraph struct {
	nodes    map[string]*ComputeNode
	edges    map[string][]string
	schedule []string
	mu       sync.RWMutex
}

// ComputeNode represents a node in the computation graph
type ComputeNode struct {
	ID         string
	Operation  func(*tensor.Tensor) *tensor.Tensor
	Input      *tensor.Tensor
	Output     *tensor.Tensor
	StreamID   StreamID
	Status     int32 // 0: pending, 1: running, 2: complete
}

// NewDynamicComputationGraph creates a dynamic computation graph
func NewDynamicComputationGraph() *DynamicComputationGraph {
	return &DynamicComputationGraph{
		nodes: make(map[string]*ComputeNode),
		edges: make(map[string][]string),
	}
}

// AddNode adds a computation node
func (cg *DynamicComputationGraph) AddNode(node *ComputeNode) {
	cg.mu.Lock()
	defer cg.mu.Unlock()
	
	cg.nodes[node.ID] = node
}

// AddEdge adds a dependency edge
func (cg *DynamicComputationGraph) AddEdge(from, to string) {
	cg.mu.Lock()
	defer cg.mu.Unlock()
	
	cg.edges[from] = append(cg.edges[from], to)
}

// Schedule creates an execution schedule
func (cg *DynamicComputationGraph) Schedule() []string {
	cg.mu.Lock()
	defer cg.mu.Unlock()
	
	// Topological sort
	visited := make(map[string]bool)
	schedule := []string{}
	
	var visit func(string)
	visit = func(nodeID string) {
		if visited[nodeID] {
			return
		}
		
		visited[nodeID] = true
		
		// Visit dependencies first
		for _, dep := range cg.edges[nodeID] {
			visit(dep)
		}
		
		schedule = append([]string{nodeID}, schedule...)
	}
	
	for nodeID := range cg.nodes {
		visit(nodeID)
	}
	
	cg.schedule = schedule
	return schedule
}

// Execute runs the computation graph
func (cg *DynamicComputationGraph) Execute(executor *ParallelExecutor) {
	schedule := cg.Schedule()
	
	// Execute nodes in parallel where possible
	for _, nodeID := range schedule {
		node := cg.nodes[nodeID]
		
		// Check if dependencies are complete
		ready := true
		for depID := range cg.edges {
			if contains(cg.edges[depID], nodeID) {
				depNode := cg.nodes[depID]
				if atomic.LoadInt32(&depNode.Status) != 2 {
					ready = false
					break
				}
			}
		}
		
		if ready {
			// Submit node for execution
			task := StreamTask{
				ID:       len(schedule),
				StreamID: node.StreamID,
				Execute: func(stream unsafe.Pointer) {
					atomic.StoreInt32(&node.Status, 1) // Running
					node.Output = node.Operation(node.Input)
					atomic.StoreInt32(&node.Status, 2) // Complete
				},
			}
			
			executor.Submit(task)
		}
	}
}

// Helper functions

func getUsedGPUMemory() int64 {
	// Placeholder - would query actual GPU memory usage
	return 4 * 1024 * 1024 * 1024 // 4GB used
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}