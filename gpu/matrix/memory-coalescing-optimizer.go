package matrix

import (
	"fmt"
	"sync"
	"time"

	"github.com/tsawler/go-nngpu/tensor"
)

// Phase 8C: Memory Coalescing Optimizer
// This component optimizes memory access patterns for better GPU performance

// MemoryCoalescingOptimizer manages memory access pattern optimization
type MemoryCoalescingOptimizer struct {
	accessPatterns    map[string]*AccessPattern
	optimizationCache map[string]*CoalescingStrategy
	statistics        CoalescingStatistics
	mutex             sync.RWMutex
	enabled           bool
}

// AccessPattern describes how memory is accessed during an operation
type AccessPattern struct {
	OperationType    string
	TensorShapes     [][]int
	AccessOrder      AccessOrder
	StridePattern    []int
	AccessFrequency  int64
	LastAccess       time.Time
	IsCoalesced      bool
	BankConflicts    int
}

// AccessOrder defines how tensor elements are accessed
type AccessOrder int

const (
	AccessOrderRowMajor AccessOrder = iota
	AccessOrderColumnMajor
	AccessOrderTiled
	AccessOrderBlocked
	AccessOrderStrided
	AccessOrderRandom
)

// CoalescingStrategy defines how to optimize memory access for coalescing
type CoalescingStrategy struct {
	OperationType      string
	RecommendedLayout  TensorLayout
	RecommendedStride  []int
	TileSize          []int
	BlockSize         []int
	PaddingRequired   []int
	ExpectedSpeedup   float64
	MemoryOverhead    float64
}

// CoalescingStatistics tracks memory coalescing performance
type CoalescingStatistics struct {
	TotalOptimizations    int64
	SuccessfulCoalescing  int64
	BankConflictsReduced  int64
	AverageSpeedup        float64
	TotalMemoryAccesses   int64
	CoalescedAccesses     int64
}

// NewMemoryCoalescingOptimizer creates a new memory coalescing optimizer
func NewMemoryCoalescingOptimizer() *MemoryCoalescingOptimizer {
	return &MemoryCoalescingOptimizer{
		accessPatterns:    make(map[string]*AccessPattern),
		optimizationCache: make(map[string]*CoalescingStrategy),
		enabled:           true,
	}
}

// AnalyzeAccessPattern analyzes the memory access pattern for an operation
func (mco *MemoryCoalescingOptimizer) AnalyzeAccessPattern(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*AccessPattern, error) {
	if !mco.enabled {
		return nil, fmt.Errorf("memory coalescing optimizer is disabled")
	}

	mco.mutex.Lock()
	defer mco.mutex.Unlock()

	// Create pattern key
	key := mco.createPatternKey(operationType, tensors)
	
	// Check if pattern exists
	if pattern, exists := mco.accessPatterns[key]; exists {
		pattern.AccessFrequency++
		pattern.LastAccess = time.Now()
		return pattern, nil
	}

	// Analyze new pattern
	pattern := &AccessPattern{
		OperationType:   operationType,
		TensorShapes:    make([][]int, len(tensors)),
		AccessFrequency: 1,
		LastAccess:      time.Now(),
	}

	for i, t := range tensors {
		pattern.TensorShapes[i] = make([]int, len(t.Shape))
		copy(pattern.TensorShapes[i], t.Shape)
	}

	// Determine access order based on operation type
	pattern.AccessOrder = mco.determineAccessOrder(operationType, tensors)
	pattern.StridePattern = mco.calculateStridePattern(tensors)
	pattern.IsCoalesced = mco.checkCoalescing(pattern)
	pattern.BankConflicts = mco.estimateBankConflicts(pattern)

	mco.accessPatterns[key] = pattern
	return pattern, nil
}

// OptimizeCoalescing optimizes memory access patterns for better coalescing
func (mco *MemoryCoalescingOptimizer) OptimizeCoalescing(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*CoalescingStrategy, error) {
	if !mco.enabled {
		return nil, fmt.Errorf("memory coalescing optimizer is disabled")
	}

	mco.mutex.Lock()
	defer mco.mutex.Unlock()

	key := mco.createPatternKey(operationType, tensors)

	// Check cache first
	if strategy, exists := mco.optimizationCache[key]; exists {
		return strategy, nil
	}

	// Analyze access pattern
	pattern, err := mco.AnalyzeAccessPattern(operationType, tensors, params)
	if err != nil {
		return nil, err
	}

	// Create optimization strategy
	strategy := &CoalescingStrategy{
		OperationType: operationType,
	}

	switch operationType {
	case "matrix_multiply":
		strategy = mco.optimizeMatrixMultiplyCoalescing(tensors, pattern)
	case "convolution":
		strategy = mco.optimizeConvolutionCoalescing(tensors, pattern)
	case "elementwise":
		strategy = mco.optimizeElementwiseCoalescing(tensors, pattern)
	default:
		strategy = mco.optimizeGeneralCoalescing(tensors, pattern)
	}

	// Cache the strategy
	mco.optimizationCache[key] = strategy
	
	// Update statistics
	mco.statistics.TotalOptimizations++
	if strategy.ExpectedSpeedup > 1.1 {
		mco.statistics.SuccessfulCoalescing++
	}

	return strategy, nil
}

// optimizeMatrixMultiplyCoalescing optimizes coalescing for matrix multiplication
func (mco *MemoryCoalescingOptimizer) optimizeMatrixMultiplyCoalescing(
	tensors []*tensor.Tensor,
	pattern *AccessPattern,
) *CoalescingStrategy {
	strategy := &CoalescingStrategy{
		OperationType:     "matrix_multiply",
		RecommendedLayout: LayoutTiled,
		ExpectedSpeedup:   1.2,
		MemoryOverhead:    5.0,
	}

	if len(tensors) >= 2 {
		// For matrix multiplication, use tiled access pattern
		rows, cols := tensors[0].Shape[0], tensors[1].Shape[1]
		
		// Calculate optimal tile size based on shared memory constraints
		maxSharedMem := 48 * 1024 // 48KB typical shared memory
		floatSize := 4           // 4 bytes per float32
		
		// Try different tile sizes
		for tileSize := 32; tileSize >= 8; tileSize /= 2 {
			memRequired := 2 * tileSize * tileSize * floatSize
			if memRequired <= maxSharedMem {
				strategy.TileSize = []int{tileSize, tileSize}
				strategy.BlockSize = []int{tileSize, tileSize}
				break
			}
		}

		// Estimate speedup based on cache efficiency
		if strategy.TileSize != nil && len(strategy.TileSize) >= 2 {
			tileSize := strategy.TileSize[0]
			cacheEfficiency := float64(tileSize*tileSize) / float64(rows*cols)
			strategy.ExpectedSpeedup = 1.0 + cacheEfficiency*0.5
		}
	}

	return strategy
}

// optimizeConvolutionCoalescing optimizes coalescing for convolution
func (mco *MemoryCoalescingOptimizer) optimizeConvolutionCoalescing(
	tensors []*tensor.Tensor,
	pattern *AccessPattern,
) *CoalescingStrategy {
	strategy := &CoalescingStrategy{
		OperationType:     "convolution",
		RecommendedLayout: LayoutNHWC,
		ExpectedSpeedup:   1.3,
		MemoryOverhead:    10.0,
	}

	if len(tensors) >= 1 && len(tensors[0].Shape) == 4 {
		// For convolution, optimize for spatial locality
		_, height, width, channels := tensors[0].Shape[0], tensors[0].Shape[1], tensors[0].Shape[2], tensors[0].Shape[3]
		
		// Use small tiles for convolution to maximize data reuse
		tileH := min(height, 16)
		tileW := min(width, 16)
		
		strategy.TileSize = []int{tileH, tileW}
		strategy.BlockSize = []int{1, tileH, tileW, min(channels, 16)}
		
		// Minimal padding for alignment
		if channels%4 != 0 {
			strategy.PaddingRequired = []int{0, 0, 0, 4 - (channels % 4)}
		}
		
		// Calculate expected speedup based on spatial locality
		spatialReuse := float64(tileH * tileW) / 256.0 // Normalize to typical cache line
		strategy.ExpectedSpeedup = 1.0 + spatialReuse*0.4
	}

	return strategy
}

// optimizeElementwiseCoalescing optimizes coalescing for elementwise operations
func (mco *MemoryCoalescingOptimizer) optimizeElementwiseCoalescing(
	tensors []*tensor.Tensor,
	pattern *AccessPattern,
) *CoalescingStrategy {
	strategy := &CoalescingStrategy{
		OperationType:     "elementwise",
		RecommendedLayout: LayoutRowMajor,
		ExpectedSpeedup:   1.1,
		MemoryOverhead:    2.0,
	}

	if len(tensors) >= 1 {
		// For elementwise operations, ensure contiguous access
		totalElements := 1
		for _, dim := range tensors[0].Shape {
			totalElements *= dim
		}
		
		// Use vector-friendly block sizes
		blockSize := min(totalElements, 256) // Process in blocks of 256 elements
		strategy.BlockSize = []int{blockSize}
		
		// Calculate speedup based on memory bandwidth utilization
		if totalElements >= 1024 {
			strategy.ExpectedSpeedup = 1.15 // Good vectorization potential
		}
	}

	return strategy
}

// optimizeGeneralCoalescing provides general coalescing optimization
func (mco *MemoryCoalescingOptimizer) optimizeGeneralCoalescing(
	tensors []*tensor.Tensor,
	pattern *AccessPattern,
) *CoalescingStrategy {
	return &CoalescingStrategy{
		OperationType:     "general",
		RecommendedLayout: LayoutRowMajor,
		ExpectedSpeedup:   1.05,
		MemoryOverhead:    1.0,
		BlockSize:         []int{32}, // Default block size
	}
}

// Helper functions

func (mco *MemoryCoalescingOptimizer) createPatternKey(operationType string, tensors []*tensor.Tensor) string {
	key := operationType
	for _, t := range tensors {
		for _, dim := range t.Shape {
			key += fmt.Sprintf("_%d", dim)
		}
	}
	return key
}

func (mco *MemoryCoalescingOptimizer) determineAccessOrder(operationType string, tensors []*tensor.Tensor) AccessOrder {
	switch operationType {
	case "matrix_multiply":
		return AccessOrderTiled
	case "convolution":
		return AccessOrderBlocked
	case "elementwise":
		return AccessOrderRowMajor
	default:
		return AccessOrderRowMajor
	}
}

func (mco *MemoryCoalescingOptimizer) calculateStridePattern(tensors []*tensor.Tensor) []int {
	if len(tensors) == 0 || len(tensors[0].Shape) == 0 {
		return []int{1}
	}
	
	shape := tensors[0].Shape
	stride := make([]int, len(shape))
	stride[len(stride)-1] = 1
	
	for i := len(stride) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * shape[i+1]
	}
	
	return stride
}

func (mco *MemoryCoalescingOptimizer) checkCoalescing(pattern *AccessPattern) bool {
	// Simple heuristic: row-major access with unit stride is coalesced
	return pattern.AccessOrder == AccessOrderRowMajor || pattern.AccessOrder == AccessOrderTiled
}

func (mco *MemoryCoalescingOptimizer) estimateBankConflicts(pattern *AccessPattern) int {
	// Simple estimation based on access pattern
	switch pattern.AccessOrder {
	case AccessOrderRowMajor, AccessOrderTiled:
		return 0 // Good coalescing, minimal conflicts
	case AccessOrderColumnMajor:
		return 2 // Some conflicts
	case AccessOrderStrided:
		return 4 // More conflicts
	default:
		return 8 // Random access, many conflicts
	}
}

// GetStatistics returns current coalescing statistics
func (mco *MemoryCoalescingOptimizer) GetStatistics() CoalescingStatistics {
	mco.mutex.RLock()
	defer mco.mutex.RUnlock()
	return mco.statistics
}

// ClearCache clears the optimization cache
func (mco *MemoryCoalescingOptimizer) ClearCache() {
	mco.mutex.Lock()
	defer mco.mutex.Unlock()
	
	mco.accessPatterns = make(map[string]*AccessPattern)
	mco.optimizationCache = make(map[string]*CoalescingStrategy)
}

// Enable enables or disables the memory coalescing optimizer
func (mco *MemoryCoalescingOptimizer) Enable(enable bool) {
	mco.mutex.Lock()
	defer mco.mutex.Unlock()
	mco.enabled = enable
}

// IsEnabled returns whether the optimizer is enabled
func (mco *MemoryCoalescingOptimizer) IsEnabled() bool {
	mco.mutex.RLock()
	defer mco.mutex.RUnlock()
	return mco.enabled
}

// Utility function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Global memory coalescing optimizer instance
var globalMemoryOptimizer *MemoryCoalescingOptimizer
var memoryOptimizerOnce sync.Once

// GetGlobalMemoryOptimizer returns the global memory coalescing optimizer
func GetGlobalMemoryOptimizer() *MemoryCoalescingOptimizer {
	memoryOptimizerOnce.Do(func() {
		globalMemoryOptimizer = NewMemoryCoalescingOptimizer()
	})
	return globalMemoryOptimizer
}

// OptimizeMemoryCoalescing optimizes memory coalescing for an operation using the global optimizer
func OptimizeMemoryCoalescing(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*CoalescingStrategy, error) {
	optimizer := GetGlobalMemoryOptimizer()
	return optimizer.OptimizeCoalescing(operationType, tensors, params)
}

// AnalyzeMemoryAccessPattern analyzes memory access patterns using the global optimizer
func AnalyzeMemoryAccessPattern(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*AccessPattern, error) {
	optimizer := GetGlobalMemoryOptimizer()
	return optimizer.AnalyzeAccessPattern(operationType, tensors, params)
}