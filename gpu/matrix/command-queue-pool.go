package matrix

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"sync"
	"unsafe"
)

// CommandQueuePool manages a pool of reusable Metal command queues
type CommandQueuePool struct {
	device     unsafe.Pointer
	queues     []unsafe.Pointer
	available  chan unsafe.Pointer
	inUse      map[unsafe.Pointer]bool
	mu         sync.RWMutex
	poolSize   int
}

var (
	globalQueuePool *CommandQueuePool
	queuePoolOnce   sync.Once
)

// GetGlobalCommandQueuePool returns the singleton command queue pool
func GetGlobalCommandQueuePool(device unsafe.Pointer) *CommandQueuePool {
	queuePoolOnce.Do(func() {
		globalQueuePool = NewCommandQueuePool(device, 4) // 4 reusable queues
	})
	return globalQueuePool
}

// NewCommandQueuePool creates a new pool of command queues
func NewCommandQueuePool(device unsafe.Pointer, poolSize int) *CommandQueuePool {
	if poolSize <= 0 {
		poolSize = 4
	}
	
	pool := &CommandQueuePool{
		device:    device,
		queues:    make([]unsafe.Pointer, 0, poolSize),
		available: make(chan unsafe.Pointer, poolSize),
		inUse:     make(map[unsafe.Pointer]bool),
		poolSize:  poolSize,
	}
	
	// Pre-create command queues
	for i := 0; i < poolSize; i++ {
		queue := C.createCommandQueue(device)
		if queue != nil {
			pool.queues = append(pool.queues, queue)
			pool.available <- queue
		}
	}
	
	return pool
}

// GetQueue gets a command queue from the pool (blocks if none available)
func (p *CommandQueuePool) GetQueue() unsafe.Pointer {
	queue := <-p.available
	
	p.mu.Lock()
	p.inUse[queue] = true
	p.mu.Unlock()
	
	return queue
}

// ReturnQueue returns a command queue to the pool
func (p *CommandQueuePool) ReturnQueue(queue unsafe.Pointer) {
	p.mu.Lock()
	delete(p.inUse, queue)
	p.mu.Unlock()
	
	select {
	case p.available <- queue:
		// Successfully returned to pool
	default:
		// Pool is full, this shouldn't happen
	}
}

// GetQueueNonBlocking tries to get a queue without blocking
func (p *CommandQueuePool) GetQueueNonBlocking() unsafe.Pointer {
	select {
	case queue := <-p.available:
		p.mu.Lock()
		p.inUse[queue] = true
		p.mu.Unlock()
		return queue
	default:
		// No queue available, create a temporary one
		return C.createCommandQueue(p.device)
	}
}

// Size returns the total number of queues in the pool
func (p *CommandQueuePool) Size() int {
	return len(p.queues)
}

// Available returns the number of available queues
func (p *CommandQueuePool) Available() int {
	return len(p.available)
}

// InUse returns the number of queues currently in use
func (p *CommandQueuePool) InUse() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.inUse)
}