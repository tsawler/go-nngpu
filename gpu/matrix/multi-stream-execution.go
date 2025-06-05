package matrix

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation
// #include "../../internal/cgo/metal_bridge.h"
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// StreamManager manages multiple Metal command queues for parallel execution
type StreamManager struct {
	device      unsafe.Pointer
	streams     []unsafe.Pointer // Multiple command queues
	numStreams  int
	currentIdx  int32
	mu          sync.RWMutex
	
	// Stream synchronization
	streamEvents map[int][]unsafe.Pointer
	eventPool    *sync.Pool
	
	// Performance metrics
	streamLoads  []int64
	streamTimes  []int64
}

// StreamID represents a specific command queue
type StreamID int

const (
	DefaultStream StreamID = 0
	ComputeStream StreamID = 1
	TransferStream StreamID = 2
	MaxStreams = 8
)

// NewStreamManager creates a manager for multi-stream execution
func NewStreamManager(device unsafe.Pointer, numStreams int) (*StreamManager, error) {
	if numStreams <= 0 || numStreams > MaxStreams {
		numStreams = 4 // Default to 4 streams
	}
	
	sm := &StreamManager{
		device:       device,
		numStreams:   numStreams,
		streams:      make([]unsafe.Pointer, numStreams),
		streamEvents: make(map[int][]unsafe.Pointer),
		streamLoads:  make([]int64, numStreams),
		streamTimes:  make([]int64, numStreams),
		eventPool: &sync.Pool{
			New: func() interface{} {
				return createMetalEvent(device)
			},
		},
	}
	
	// Create multiple command queues
	for i := 0; i < numStreams; i++ {
		queue := createCommandQueue(device)
		if queue == nil {
			return nil, fmt.Errorf("failed to create command queue %d", i)
		}
		sm.streams[i] = queue
	}
	
	return sm, nil
}

// GetStream returns the least loaded stream for load balancing
func (sm *StreamManager) GetStream() (StreamID, unsafe.Pointer) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	// Find least loaded stream
	minLoad := atomic.LoadInt64(&sm.streamLoads[0])
	minIdx := 0
	
	for i := 1; i < sm.numStreams; i++ {
		load := atomic.LoadInt64(&sm.streamLoads[i])
		if load < minLoad {
			minLoad = load
			minIdx = i
		}
	}
	
	return StreamID(minIdx), sm.streams[minIdx]
}

// GetSpecificStream returns a specific stream by ID
func (sm *StreamManager) GetSpecificStream(id StreamID) unsafe.Pointer {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	if int(id) >= sm.numStreams {
		return sm.streams[0]
	}
	return sm.streams[id]
}

// SubmitToStream submits work to a specific stream
func (sm *StreamManager) SubmitToStream(id StreamID, work func(unsafe.Pointer)) {
	stream := sm.GetSpecificStream(id)
	
	// Increment load counter
	atomic.AddInt64(&sm.streamLoads[int(id)], 1)
	
	// Execute work
	work(stream)
	
	// Decrement load counter
	atomic.AddInt64(&sm.streamLoads[int(id)], -1)
}

// SynchronizeStream waits for a specific stream to complete
func (sm *StreamManager) SynchronizeStream(id StreamID) {
	stream := sm.GetSpecificStream(id)
	synchronizeCommandQueue(stream)
}

// SynchronizeAll waits for all streams to complete
func (sm *StreamManager) SynchronizeAll() {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	for i := 0; i < sm.numStreams; i++ {
		synchronizeCommandQueue(sm.streams[i])
	}
}

// ParallelExecutor manages parallel kernel execution
type ParallelExecutor struct {
	streamMgr    *StreamManager
	taskQueue    chan StreamTask
	workers      int
	wg           sync.WaitGroup
	shutdown     chan struct{}
	
	// Kernel fusion optimizer
	fusionBuffer []StreamTask
	fusionMu     sync.Mutex
}

// StreamTask represents a GPU computation task
type StreamTask struct {
	ID       int
	StreamID StreamID
	Execute  func(unsafe.Pointer)
	Depends  []int // Task dependencies
	Priority int
}

// NewParallelExecutor creates a new parallel execution manager
func NewParallelExecutor(streamMgr *StreamManager) *ParallelExecutor {
	workers := runtime.NumCPU()
	
	pe := &ParallelExecutor{
		streamMgr:    streamMgr,
		taskQueue:    make(chan StreamTask, workers*2),
		workers:      workers,
		shutdown:     make(chan struct{}),
		fusionBuffer: make([]StreamTask, 0, 16),
	}
	
	// Start worker goroutines
	pe.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go pe.worker()
	}
	
	return pe
}

// worker processes tasks from the queue
func (pe *ParallelExecutor) worker() {
	defer pe.wg.Done()
	
	for {
		select {
		case task := <-pe.taskQueue:
			pe.executeTask(task)
		case <-pe.shutdown:
			return
		}
	}
}

// executeTask executes a single task on the appropriate stream
func (pe *ParallelExecutor) executeTask(task StreamTask) {
	// Get stream based on task specification or load balancing
	var stream unsafe.Pointer
	if task.StreamID >= 0 {
		stream = pe.streamMgr.GetSpecificStream(task.StreamID)
	} else {
		_, stream = pe.streamMgr.GetStream()
	}
	
	// Execute the task
	task.Execute(stream)
}

// Submit adds a task to the execution queue
func (pe *ParallelExecutor) Submit(task StreamTask) {
	pe.taskQueue <- task
}

// SubmitBatch submits multiple tasks for potential fusion
func (pe *ParallelExecutor) SubmitBatch(tasks []StreamTask) {
	pe.fusionMu.Lock()
	defer pe.fusionMu.Unlock()
	
	// Try to fuse compatible tasks
	fusedTasks := pe.tryFuseTasks(tasks)
	
	// Submit fused and remaining tasks
	for _, task := range fusedTasks {
		pe.taskQueue <- task
	}
}

// tryFuseTasks attempts to fuse compatible operations
func (pe *ParallelExecutor) tryFuseTasks(tasks []StreamTask) []StreamTask {
	// Simple fusion strategy: combine adjacent element-wise operations
	result := make([]StreamTask, 0, len(tasks))
	
	for i := 0; i < len(tasks); {
		if i+1 < len(tasks) && pe.canFuse(tasks[i], tasks[i+1]) {
			// Create fused task
			fusedTask := pe.createFusedTask(tasks[i], tasks[i+1])
			result = append(result, fusedTask)
			i += 2
		} else {
			result = append(result, tasks[i])
			i++
		}
	}
	
	return result
}

// canFuse checks if two tasks can be fused
func (pe *ParallelExecutor) canFuse(t1, t2 StreamTask) bool {
	// Check if tasks have no dependencies between them
	for _, dep := range t2.Depends {
		if dep == t1.ID {
			return false
		}
	}
	
	// Check if they target the same stream
	return t1.StreamID == t2.StreamID
}

// createFusedTask creates a single task from two compatible tasks
func (pe *ParallelExecutor) createFusedTask(t1, t2 StreamTask) StreamTask {
	return StreamTask{
		ID:       t1.ID, // Use first task's ID
		StreamID: t1.StreamID,
		Execute: func(stream unsafe.Pointer) {
			// Execute both operations in sequence
			t1.Execute(stream)
			t2.Execute(stream)
		},
		Depends:  append(t1.Depends, t2.Depends...),
		Priority: max(t1.Priority, t2.Priority),
	}
}

// Shutdown gracefully shuts down the executor
func (pe *ParallelExecutor) Shutdown() {
	close(pe.shutdown)
	pe.wg.Wait()
	close(pe.taskQueue)
}

// StreamScheduler optimizes task scheduling across streams
type StreamScheduler struct {
	executor    *ParallelExecutor
	streamMgr   *StreamManager
	taskGraph   map[int][]int // Dependency graph
	readyTasks  []StreamTask
	mu          sync.Mutex
}

// NewStreamScheduler creates an optimized task scheduler
func NewStreamScheduler(executor *ParallelExecutor, streamMgr *StreamManager) *StreamScheduler {
	return &StreamScheduler{
		executor:   executor,
		streamMgr:  streamMgr,
		taskGraph:  make(map[int][]int),
		readyTasks: make([]StreamTask, 0),
	}
}

// ScheduleDAG schedules a directed acyclic graph of tasks
func (ss *StreamScheduler) ScheduleDAG(tasks []StreamTask) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	
	// Build dependency graph
	ss.buildDependencyGraph(tasks)
	
	// Find tasks with no dependencies
	for _, task := range tasks {
		if len(task.Depends) == 0 {
			ss.readyTasks = append(ss.readyTasks, task)
		}
	}
	
	// Schedule ready tasks
	ss.scheduleReady()
}

// buildDependencyGraph creates a reverse dependency mapping
func (ss *StreamScheduler) buildDependencyGraph(tasks []StreamTask) {
	for _, task := range tasks {
		for _, dep := range task.Depends {
			ss.taskGraph[dep] = append(ss.taskGraph[dep], task.ID)
		}
	}
}

// scheduleReady schedules all ready tasks
func (ss *StreamScheduler) scheduleReady() {
	// Sort by priority
	sortTasksByPriority(ss.readyTasks)
	
	// Submit tasks to executor
	for _, task := range ss.readyTasks {
		ss.executor.Submit(task)
	}
	
	ss.readyTasks = ss.readyTasks[:0]
}

// Helper functions implemented in Metal bridge

func createCommandQueue(device unsafe.Pointer) unsafe.Pointer {
	return C.createCommandQueue(device)
}

func synchronizeCommandQueue(queue unsafe.Pointer) {
	C.synchronizeCommandQueue(queue)
}

func createMetalEvent(device unsafe.Pointer) unsafe.Pointer {
	return C.createMetalEvent(device)
}

func sortTasksByPriority(tasks []StreamTask) {
	// Simple priority sort
	for i := 0; i < len(tasks)-1; i++ {
		for j := i + 1; j < len(tasks); j++ {
			if tasks[j].Priority > tasks[i].Priority {
				tasks[i], tasks[j] = tasks[j], tasks[i]
			}
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}