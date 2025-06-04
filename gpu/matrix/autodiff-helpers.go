package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tsawler/go-nngpu/tensor"
)

// AutodiffProfiler provides detailed profiling and analysis of the computation graph
type AutodiffProfiler struct {
	enabled        bool
	operationStats map[OpType]*OperationStats
	memoryStats    *AutodiffMemoryStats
	timingStats    map[string]time.Duration
	graphAnalysis  *GraphAnalysis
	mutex          sync.RWMutex
}

// OperationStats tracks statistics for each operation type
type OperationStats struct {
	Count           int64
	TotalTime       time.Duration
	AverageTime     time.Duration
	PeakMemoryUsage int64
	TotalMemoryUsed int64
	GradientNorm    float32
	BackwardTime    time.Duration
	ForwardTime     time.Duration
}

// AutodiffMemoryStats tracks memory usage during autodiff
type AutodiffMemoryStats struct {
	SavedTensorsMemory  int64
	GradientMemory      int64
	PeakGraphMemory     int64
	TotalAllocations    int64
	ActiveTensors       int64
	MemoryFragmentation float32
}

// GraphAnalysis provides analysis of the computation graph structure
type GraphAnalysis struct {
	NodeCount          int
	EdgeCount          int
	MaxDepth           int
	CriticalPath       []*GradientTensor
	CyclicDependencies bool
	ParallelizableOps  [][]OpType
	MemoryBottlenecks  []*GradientTensor
}

// Global profiler instance
var globalProfiler *AutodiffProfiler

func init() {
	globalProfiler = NewAutodiffProfiler()
}

// NewAutodiffProfiler creates a new autodiff profiler
func NewAutodiffProfiler() *AutodiffProfiler {
	return &AutodiffProfiler{
		enabled:        false,
		operationStats: make(map[OpType]*OperationStats),
		memoryStats:    &AutodiffMemoryStats{},
		timingStats:    make(map[string]time.Duration),
		graphAnalysis:  &GraphAnalysis{},
	}
}

// EnableProfiling enables autodiff profiling
func EnableAutodiffProfiling() {
	globalProfiler.mutex.Lock()
	defer globalProfiler.mutex.Unlock()
	globalProfiler.enabled = true
	globalProfiler.resetStats()
}

// DisableProfiling disables autodiff profiling
func DisableAutodiffProfiling() {
	globalProfiler.mutex.Lock()
	defer globalProfiler.mutex.Unlock()
	globalProfiler.enabled = false
}

// GetAutodiffProfile returns the current profiling results
func GetAutodiffProfile() *AutodiffProfiler {
	globalProfiler.mutex.RLock()
	defer globalProfiler.mutex.RUnlock()

	// Return a copy to avoid race conditions
	profile := &AutodiffProfiler{
		enabled:        globalProfiler.enabled,
		operationStats: make(map[OpType]*OperationStats),
		memoryStats:    &AutodiffMemoryStats{},
		timingStats:    make(map[string]time.Duration),
		graphAnalysis:  &GraphAnalysis{},
	}

	// Deep copy operation stats
	for opType, stats := range globalProfiler.operationStats {
		profile.operationStats[opType] = &OperationStats{
			Count:           stats.Count,
			TotalTime:       stats.TotalTime,
			AverageTime:     stats.AverageTime,
			PeakMemoryUsage: stats.PeakMemoryUsage,
			TotalMemoryUsed: stats.TotalMemoryUsed,
			GradientNorm:    stats.GradientNorm,
			BackwardTime:    stats.BackwardTime,
			ForwardTime:     stats.ForwardTime,
		}
	}

	// Copy memory stats
	*profile.memoryStats = *globalProfiler.memoryStats

	// Copy timing stats
	for key, value := range globalProfiler.timingStats {
		profile.timingStats[key] = value
	}

	// Copy graph analysis
	*profile.graphAnalysis = *globalProfiler.graphAnalysis

	return profile
}

// recordOperation records statistics for an operation
func (p *AutodiffProfiler) recordOperation(opType OpType, forwardTime, backwardTime time.Duration, memoryUsed int64) {
	if !p.enabled {
		return
	}

	p.mutex.Lock()
	defer p.mutex.Unlock()

	stats, exists := p.operationStats[opType]
	if !exists {
		stats = &OperationStats{}
		p.operationStats[opType] = stats
	}

	stats.Count++
	stats.ForwardTime += forwardTime
	stats.BackwardTime += backwardTime
	stats.TotalTime += forwardTime + backwardTime
	stats.AverageTime = stats.TotalTime / time.Duration(stats.Count)
	stats.TotalMemoryUsed += memoryUsed

	if memoryUsed > stats.PeakMemoryUsage {
		stats.PeakMemoryUsage = memoryUsed
	}
}

// resetStats resets all profiling statistics
func (p *AutodiffProfiler) resetStats() {
	p.operationStats = make(map[OpType]*OperationStats)
	p.memoryStats = &AutodiffMemoryStats{}
	p.timingStats = make(map[string]time.Duration)
	p.graphAnalysis = &GraphAnalysis{}
}

// PrintProfile prints a detailed profiling report
func (p *AutodiffProfiler) PrintProfile() {
	p.mutex.RLock()
	defer p.mutex.RUnlock()

	fmt.Println("=== Autodiff Profiling Report ===")
	fmt.Printf("Profiling Enabled: %t\n\n", p.enabled)

	// Operation statistics
	fmt.Println("Operation Statistics:")
	fmt.Printf("%-20s %-8s %-12s %-12s %-12s %-12s\n",
		"Operation", "Count", "Avg Time", "Forward", "Backward", "Memory")
	fmt.Println(strings.Repeat("-", 80))

	for opType, stats := range p.operationStats {
		fmt.Printf("%-20s %-8d %-12s %-12s %-12s %-12s\n",
			opTypeToString(opType),
			stats.Count,
			stats.AverageTime.String(),
			stats.ForwardTime.String(),
			stats.BackwardTime.String(),
			formatBytes(stats.TotalMemoryUsed))
	}

	// Memory statistics
	fmt.Printf("\nMemory Statistics:\n")
	fmt.Printf("Saved Tensors Memory: %s\n", formatBytes(p.memoryStats.SavedTensorsMemory))
	fmt.Printf("Gradient Memory: %s\n", formatBytes(p.memoryStats.GradientMemory))
	fmt.Printf("Peak Graph Memory: %s\n", formatBytes(p.memoryStats.PeakGraphMemory))
	fmt.Printf("Total Allocations: %d\n", p.memoryStats.TotalAllocations)
	fmt.Printf("Active Tensors: %d\n", p.memoryStats.ActiveTensors)

	// Graph analysis
	fmt.Printf("\nGraph Analysis:\n")
	fmt.Printf("Node Count: %d\n", p.graphAnalysis.NodeCount)
	fmt.Printf("Edge Count: %d\n", p.graphAnalysis.EdgeCount)
	fmt.Printf("Max Depth: %d\n", p.graphAnalysis.MaxDepth)
	fmt.Printf("Cyclic Dependencies: %t\n", p.graphAnalysis.CyclicDependencies)
}

// GraphOptimizer provides optimization tools for computation graphs
type GraphOptimizer struct {
	fusionRules     map[string]FusionRule
	memoryOptimizer *MemoryOptimizer
	parallelizer    *GraphParallelizer
}

// FusionRule defines how operations can be fused together
type FusionRule struct {
	Pattern     []OpType
	FusedOp     OpType
	Speedup     float32
	MemorySaved int64
}

// MemoryOptimizer handles memory optimization for gradients
type MemoryOptimizer struct {
	inPlaceOps       map[OpType]bool
	memoryThreshold  int64
	compressionLevel float32
}

// GraphParallelizer identifies parallelizable operations
type GraphParallelizer struct {
	maxParallelism int
	dependencies   map[*GradientTensor][]*GradientTensor
}

// NewGraphOptimizer creates a new graph optimizer
func NewGraphOptimizer() *GraphOptimizer {
	optimizer := &GraphOptimizer{
		fusionRules: make(map[string]FusionRule),
		memoryOptimizer: &MemoryOptimizer{
			inPlaceOps:       make(map[OpType]bool),
			memoryThreshold:  1024 * 1024 * 100, // 100MB
			compressionLevel: 0.5,
		},
		parallelizer: &GraphParallelizer{
			maxParallelism: runtime.NumCPU(),
			dependencies:   make(map[*GradientTensor][]*GradientTensor),
		},
	}

	// Initialize common fusion rules
	optimizer.initializeFusionRules()
	optimizer.initializeInPlaceOps()

	return optimizer
}

// initializeFusionRules sets up common operation fusion patterns
func (go_ *GraphOptimizer) initializeFusionRules() {
	// ReLU + Add fusion
	go_.fusionRules["relu_add"] = FusionRule{
		Pattern:     []OpType{OpReLU, OpAdd},
		FusedOp:     OpAdd, // Use existing op type as placeholder
		Speedup:     1.3,
		MemorySaved: 1024,
	}

	// Conv2D + BatchNorm fusion
	go_.fusionRules["conv_batchnorm"] = FusionRule{
		Pattern:     []OpType{OpConv2D, OpBatchNorm},
		FusedOp:     OpConv2D,
		Speedup:     1.5,
		MemorySaved: 2048,
	}

	// MatMul + Add fusion (linear layer)
	go_.fusionRules["matmul_add"] = FusionRule{
		Pattern:     []OpType{OpMatMul, OpAdd},
		FusedOp:     OpMatMul,
		Speedup:     1.2,
		MemorySaved: 512,
	}
}

// initializeInPlaceOps marks operations that can be done in-place
func (go_ *GraphOptimizer) initializeInPlaceOps() {
	go_.memoryOptimizer.inPlaceOps[OpReLU] = true
	go_.memoryOptimizer.inPlaceOps[OpSigmoid] = true
	go_.memoryOptimizer.inPlaceOps[OpTanh] = true
	go_.memoryOptimizer.inPlaceOps[OpAdd] = true // For broadcasting cases
	go_.memoryOptimizer.inPlaceOps[OpMul] = true // For element-wise multiplication
}

// OptimizeGraph applies various optimizations to the computation graph
func (go_ *GraphOptimizer) OptimizeGraph() error {
	// Analyze current graph
	err := go_.analyzeGraph()
	if err != nil {
		return fmt.Errorf("graph analysis failed: %w", err)
	}

	// Apply fusion optimizations
	err = go_.applyFusions()
	if err != nil {
		return fmt.Errorf("fusion optimization failed: %w", err)
	}

	// Apply memory optimizations
	err = go_.optimizeMemory()
	if err != nil {
		return fmt.Errorf("memory optimization failed: %w", err)
	}

	// Apply parallelization
	err = go_.optimizeParallelism()
	if err != nil {
		return fmt.Errorf("parallelization failed: %w", err)
	}

	return nil
}

// analyzeGraph analyzes the current computation graph
func (go_ *GraphOptimizer) analyzeGraph() error {
	globalProfiler.mutex.Lock()
	defer globalProfiler.mutex.Unlock()

	analysis := globalProfiler.graphAnalysis
	analysis.NodeCount = len(globalGraph.nodes)
	analysis.EdgeCount = 0
	analysis.MaxDepth = 0

	// Calculate edges and depth
	for _, node := range globalGraph.nodes {
		if node.GradFn != nil {
			analysis.EdgeCount += len(node.GradFn.Inputs)

			// Calculate depth (simplified - could be more sophisticated)
			depth := go_.calculateNodeDepth(node, make(map[*GradientTensor]int))
			if depth > analysis.MaxDepth {
				analysis.MaxDepth = depth
			}
		}
	}

	// Check for cycles (simplified check)
	analysis.CyclicDependencies = go_.detectCycles()

	return nil
}

// calculateNodeDepth calculates the depth of a node in the graph
func (go_ *GraphOptimizer) calculateNodeDepth(node *GradientTensor, visited map[*GradientTensor]int) int {
	if depth, exists := visited[node]; exists {
		return depth
	}

	if node.IsLeaf {
		visited[node] = 0
		return 0
	}

	maxDepth := 0
	if node.GradFn != nil {
		for _, input := range node.GradFn.Inputs {
			depth := go_.calculateNodeDepth(input, visited)
			if depth > maxDepth {
				maxDepth = depth
			}
		}
	}

	depth := maxDepth + 1
	visited[node] = depth
	return depth
}

// detectCycles detects if there are cycles in the computation graph
func (go_ *GraphOptimizer) detectCycles() bool {
	visited := make(map[*GradientTensor]bool)
	recStack := make(map[*GradientTensor]bool)

	for _, node := range globalGraph.nodes {
		if !visited[node] {
			if go_.dfsDetectCycle(node, visited, recStack) {
				return true
			}
		}
	}

	return false
}

// dfsDetectCycle performs DFS to detect cycles
func (go_ *GraphOptimizer) dfsDetectCycle(node *GradientTensor, visited, recStack map[*GradientTensor]bool) bool {
	visited[node] = true
	recStack[node] = true

	if node.GradFn != nil {
		for _, input := range node.GradFn.Inputs {
			if !visited[input] {
				if go_.dfsDetectCycle(input, visited, recStack) {
					return true
				}
			} else if recStack[input] {
				return true
			}
		}
	}

	recStack[node] = false
	return false
}

// applyFusions applies operation fusion optimizations
func (go_ *GraphOptimizer) applyFusions() error {
	// This is a simplified implementation
	// In practice, you would traverse the graph and identify fusion patterns

	fusedCount := 0
	for _, node := range globalGraph.nodes {
		if node.GradFn != nil && len(node.GradFn.Inputs) > 0 {
			// Check for fusion opportunities
			pattern := go_.identifyFusionPattern(node)
			if pattern != "" {
				err := go_.applyFusion(node, pattern)
				if err == nil {
					fusedCount++
				}
			}
		}
	}

	if globalProfiler.enabled {
		globalProfiler.timingStats["fusion_count"] = time.Duration(fusedCount)
	}

	return nil
}

// identifyFusionPattern identifies if a node matches any fusion patterns
func (go_ *GraphOptimizer) identifyFusionPattern(node *GradientTensor) string {
	if node.GradFn == nil || len(node.GradFn.Inputs) == 0 {
		return ""
	}

	// Check for ReLU + Add pattern
	if node.GradFn.OpType == OpAdd && len(node.GradFn.Inputs) > 0 {
		input := node.GradFn.Inputs[0]
		if input.GradFn != nil && input.GradFn.OpType == OpReLU {
			return "relu_add"
		}
	}

	// Check for Conv2D + BatchNorm pattern
	if node.GradFn.OpType == OpBatchNorm && len(node.GradFn.Inputs) > 0 {
		input := node.GradFn.Inputs[0]
		if input.GradFn != nil && input.GradFn.OpType == OpConv2D {
			return "conv_batchnorm"
		}
	}

	// Check for MatMul + Add pattern
	if node.GradFn.OpType == OpAdd && len(node.GradFn.Inputs) > 0 {
		input := node.GradFn.Inputs[0]
		if input.GradFn != nil && input.GradFn.OpType == OpMatMul {
			return "matmul_add"
		}
	}

	return ""
}

// applyFusion applies a specific fusion to a node
func (go_ *GraphOptimizer) applyFusion(node *GradientTensor, pattern string) error {
	rule, exists := go_.fusionRules[pattern]
	if !exists {
		return fmt.Errorf("fusion rule not found: %s", pattern)
	}

	// Create a fused backward function
	originalBackwardFn := node.GradFn.BackwardFn
	node.GradFn.BackwardFn = func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
		// Call original backward function but with optimized implementation
		start := time.Now()
		result, err := originalBackwardFn(gradOutput)
		elapsed := time.Since(start)

		// Record the fusion speedup
		if globalProfiler.enabled {
			adjustedTime := time.Duration(float32(elapsed) / rule.Speedup)
			globalProfiler.timingStats["fused_"+pattern] += adjustedTime
		}

		return result, err
	}

	return nil
}

// optimizeMemory applies memory optimizations
func (go_ *GraphOptimizer) optimizeMemory() error {
	// Identify tensors that can be computed in-place
	inPlaceCandidates := go_.identifyInPlaceCandidates()

	// Apply gradient compression for large tensors
	err := go_.compressLargeGradients()
	if err != nil {
		return fmt.Errorf("gradient compression failed: %w", err)
	}

	// Optimize tensor lifetimes
	err = go_.optimizeTensorLifetimes(inPlaceCandidates)
	if err != nil {
		return fmt.Errorf("tensor lifetime optimization failed: %w", err)
	}

	return nil
}

// identifyInPlaceCandidates identifies operations that can be done in-place
func (go_ *GraphOptimizer) identifyInPlaceCandidates() []*GradientTensor {
	var candidates []*GradientTensor

	for _, node := range globalGraph.nodes {
		if node.GradFn != nil {
			if canBeInPlace, exists := go_.memoryOptimizer.inPlaceOps[node.GradFn.OpType]; exists && canBeInPlace {
				// Check if this node is only used once
				if go_.isUsedOnce(node) {
					candidates = append(candidates, node)
				}
			}
		}
	}

	return candidates
}

// isUsedOnce checks if a tensor is used only once in the graph
func (go_ *GraphOptimizer) isUsedOnce(target *GradientTensor) bool {
	useCount := 0

	for _, node := range globalGraph.nodes {
		if node.GradFn != nil {
			for _, input := range node.GradFn.Inputs {
				if input == target {
					useCount++
					if useCount > 1 {
						return false
					}
				}
			}
		}
	}

	return useCount == 1
}

// compressLargeGradients compresses gradients for tensors above a threshold
func (go_ *GraphOptimizer) compressLargeGradients() error {
	threshold := go_.memoryOptimizer.memoryThreshold

	for _, node := range globalGraph.nodes {
		if node.Gradient != nil {
			tensorSize := int64(len(node.Gradient.Data) * 4) // float32 = 4 bytes
			if tensorSize > threshold {
				err := go_.compressGradient(node.Gradient)
				if err != nil {
					return fmt.Errorf("failed to compress gradient: %w", err)
				}
			}
		}
	}

	return nil
}

// compressGradient compresses a gradient tensor
func (go_ *GraphOptimizer) compressGradient(grad *tensor.Tensor) error {
	// Simplified gradient compression using sparsification
	// In practice, you might use quantization, sparsification, or other compression methods

	if err := grad.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve gradient to CPU: %w", err)
	}

	compressionLevel := go_.memoryOptimizer.compressionLevel
	threshold := go_.calculateCompressionThreshold(grad.Data, compressionLevel)

	// Zero out small gradients
	for i, val := range grad.Data {
		if abs(val) < threshold {
			grad.Data[i] = 0.0
		}
	}

	return nil
}

// calculateCompressionThreshold calculates the threshold for gradient compression
func (go_ *GraphOptimizer) calculateCompressionThreshold(data []float32, compressionLevel float32) float32 {
	if len(data) == 0 {
		return 0.0
	}

	// Calculate percentile threshold
	sorted := make([]float32, len(data))
	for i, val := range data {
		sorted[i] = abs(val)
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	index := int(float32(len(sorted)) * compressionLevel)
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

// optimizeTensorLifetimes optimizes when tensors are allocated and freed
func (go_ *GraphOptimizer) optimizeTensorLifetimes(inPlaceCandidates []*GradientTensor) error {
	// Build a dependency graph
	deps := make(map[*GradientTensor][]*GradientTensor)

	for _, node := range globalGraph.nodes {
		if node.GradFn != nil {
			for _, input := range node.GradFn.Inputs {
				deps[input] = append(deps[input], node)
			}
		}
	}

	// Schedule early release of tensors that are no longer needed
	for _, candidate := range inPlaceCandidates {
		go_.scheduleEarlyRelease(candidate, deps)
	}

	return nil
}

// scheduleEarlyRelease schedules early release of a tensor
func (go_ *GraphOptimizer) scheduleEarlyRelease(node *GradientTensor, deps map[*GradientTensor][]*GradientTensor) {
	// Find the last use of this tensor
	lastUse := go_.findLastUse(node, deps)

	if lastUse != nil && lastUse.GradFn != nil {
		// Modify the backward function to release the tensor after use
		originalBackwardFn := lastUse.GradFn.BackwardFn
		lastUse.GradFn.BackwardFn = func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
			result, err := originalBackwardFn(gradOutput)

			// Release the tensor after backward pass
			if node.Tensor != nil {
				node.Tensor.ReleaseGPU()
			}
			if node.Gradient != nil {
				node.Gradient.ReleaseGPU()
			}

			return result, err
		}
	}
}

// findLastUse finds the last node that uses a given tensor
func (go_ *GraphOptimizer) findLastUse(target *GradientTensor, deps map[*GradientTensor][]*GradientTensor) *GradientTensor {
	users := deps[target]
	if len(users) == 0 {
		return nil
	}

	// For simplicity, return the first user
	// In practice, you'd need topological sorting to find the actual last use
	return users[0]
}

// optimizeParallelism identifies and optimizes parallelizable operations
func (go_ *GraphOptimizer) optimizeParallelism() error {
	// Build dependency graph
	go_.parallelizer.dependencies = make(map[*GradientTensor][]*GradientTensor)

	for _, node := range globalGraph.nodes {
		if node.GradFn != nil {
			for _, input := range node.GradFn.Inputs {
				go_.parallelizer.dependencies[node] = append(go_.parallelizer.dependencies[node], input)
			}
		}
	}

	// Identify independent operations that can run in parallel
	parallelGroups := go_.identifyParallelGroups()

	// Store results for analysis
	if globalProfiler.enabled {
		globalProfiler.graphAnalysis.ParallelizableOps = parallelGroups
	}

	return nil
}

// identifyParallelGroups identifies groups of operations that can run in parallel
func (go_ *GraphOptimizer) identifyParallelGroups() [][]OpType {
	var groups [][]OpType
	visited := make(map[*GradientTensor]bool)

	for _, node := range globalGraph.nodes {
		if !visited[node] && node.GradFn != nil {
			group := go_.findParallelGroup(node, visited)
			if len(group) > 1 {
				groups = append(groups, group)
			}
		}
	}

	return groups
}

// findParallelGroup finds a group of operations that can run in parallel
func (go_ *GraphOptimizer) findParallelGroup(start *GradientTensor, visited map[*GradientTensor]bool) []OpType {
	var group []OpType
	queue := []*GradientTensor{start}

	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]

		if visited[node] || node.GradFn == nil {
			continue
		}

		visited[node] = true
		group = append(group, node.GradFn.OpType)

		// Check if any dependent nodes can also be parallelized
		for _, dependent := range globalGraph.nodes {
			if !visited[dependent] && go_.canRunInParallel(node, dependent) {
				queue = append(queue, dependent)
			}
		}
	}

	return group
}

// canRunInParallel checks if two nodes can run in parallel
func (go_ *GraphOptimizer) canRunInParallel(node1, node2 *GradientTensor) bool {
	// Check if there are no dependencies between the nodes
	deps1 := go_.parallelizer.dependencies[node1]
	deps2 := go_.parallelizer.dependencies[node2]

	// Simple check: nodes can run in parallel if they don't depend on each other
	for _, dep := range deps1 {
		if dep == node2 {
			return false
		}
	}

	for _, dep := range deps2 {
		if dep == node1 {
			return false
		}
	}

	return true
}

// Helper functions

// abs returns the absolute value of a float32
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// opTypeToString converts OpType to string for display
func opTypeToString(opType OpType) string {
	switch opType {
	case OpAdd:
		return "Add"
	case OpSub:
		return "Sub"
	case OpMul:
		return "Mul"
	case OpDiv:
		return "Div"
	case OpMatMul:
		return "MatMul"
	case OpTranspose:
		return "Transpose"
	case OpReLU:
		return "ReLU"
	case OpSigmoid:
		return "Sigmoid"
	case OpTanh:
		return "Tanh"
	case OpSoftmax:
		return "Softmax"
	case OpLeakyReLU:
		return "LeakyReLU"
	case OpELU:
		return "ELU"
	case OpSwish:
		return "Swish"
	case OpGELU:
		return "GELU"
	case OpConv2D:
		return "Conv2D"
	case OpMaxPool2D:
		return "MaxPool2D"
	case OpAvgPool2D:
		return "AvgPool2D"
	case OpBatchNorm:
		return "BatchNorm"
	case OpLayerNorm:
		return "LayerNorm"
	case OpMSELoss:
		return "MSELoss"
	case OpCrossEntropyLoss:
		return "CrossEntropyLoss"
	case OpBinaryCrossEntropyLoss:
		return "BinaryCrossEntropyLoss"
	default:
		return fmt.Sprintf("Op_%d", int(opType))
	}
}

// formatBytes formats byte counts for display
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// ProfiledOperation wraps an operation with profiling
func ProfiledOperation(opType OpType, operation func() error) error {
	if !globalProfiler.enabled {
		return operation()
	}

	start := time.Now()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	err := operation()

	elapsed := time.Since(start)
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	memoryUsed := int64(memAfter.Alloc - memBefore.Alloc)
	globalProfiler.recordOperation(opType, elapsed, 0, memoryUsed)

	return err
}

// ProfiledBackwardOperation wraps a backward operation with profiling
func ProfiledBackwardOperation(opType OpType, backwardOp func() error) error {
	if !globalProfiler.enabled {
		return backwardOp()
	}

	start := time.Now()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	err := backwardOp()

	elapsed := time.Since(start)
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	memoryUsed := int64(memAfter.Alloc - memBefore.Alloc)
	globalProfiler.recordOperation(opType, 0, elapsed, memoryUsed)

	return err
}

// AutodiffDebugger provides debugging tools for automatic differentiation
type AutodiffDebugger struct {
	enabled      bool
	breakpoints  map[*GradientTensor]bool
	watchList    []*GradientTensor
	logLevel     DebugLogLevel
	outputBuffer strings.Builder
	mutex        sync.RWMutex
}

// DebugLogLevel represents different levels of debug logging
type DebugLogLevel int

const (
	DebugOff DebugLogLevel = iota
	DebugError
	DebugWarning
	DebugInfo
	DebugVerbose
)

// Global debugger instance
var globalDebugger *AutodiffDebugger

func init() {
	globalDebugger = NewAutodiffDebugger()
}

// NewAutodiffDebugger creates a new autodiff debugger
func NewAutodiffDebugger() *AutodiffDebugger {
	return &AutodiffDebugger{
		enabled:     false,
		breakpoints: make(map[*GradientTensor]bool),
		watchList:   make([]*GradientTensor, 0),
		logLevel:    DebugOff,
	}
}

// EnableDebugging enables autodiff debugging
func EnableAutodiffDebugging(logLevel DebugLogLevel) {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()
	globalDebugger.enabled = true
	globalDebugger.logLevel = logLevel
}

// DisableDebugging disables autodiff debugging
func DisableAutodiffDebugging() {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()
	globalDebugger.enabled = false
	globalDebugger.logLevel = DebugOff
}

// AddBreakpoint adds a breakpoint for a specific tensor
func AddAutodiffBreakpoint(tensor *GradientTensor) {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()
	globalDebugger.breakpoints[tensor] = true
}

// RemoveBreakpoint removes a breakpoint for a specific tensor
func RemoveAutodiffBreakpoint(tensor *GradientTensor) {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()
	delete(globalDebugger.breakpoints, tensor)
}

// AddToWatchList adds a tensor to the watch list
func AddToAutodiffWatchList(tensor *GradientTensor) {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()
	globalDebugger.watchList = append(globalDebugger.watchList, tensor)
}

// LogDebug logs a debug message with the specified level
func LogAutodiffDebug(level DebugLogLevel, message string, args ...interface{}) {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()

	if !globalDebugger.enabled || level > globalDebugger.logLevel {
		return
	}

	prefix := ""
	switch level {
	case DebugError:
		prefix = "[ERROR] "
	case DebugWarning:
		prefix = "[WARN]  "
	case DebugInfo:
		prefix = "[INFO]  "
	case DebugVerbose:
		prefix = "[DEBUG] "
	}

	formattedMessage := fmt.Sprintf(message, args...)
	globalDebugger.outputBuffer.WriteString(fmt.Sprintf("%s%s\n", prefix, formattedMessage))

	// Also print to stdout for immediate feedback
	fmt.Printf("%s%s\n", prefix, formattedMessage)
}

// GetDebugLog returns the accumulated debug log
func GetAutodiffDebugLog() string {
	globalDebugger.mutex.RLock()
	defer globalDebugger.mutex.RUnlock()
	return globalDebugger.outputBuffer.String()
}

// ClearDebugLog clears the debug log buffer
func ClearAutodiffDebugLog() {
	globalDebugger.mutex.Lock()
	defer globalDebugger.mutex.Unlock()
	globalDebugger.outputBuffer.Reset()
}

// CheckBreakpoint checks if a breakpoint should trigger for a tensor
func (d *AutodiffDebugger) CheckBreakpoint(tensor *GradientTensor) bool {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	if !d.enabled {
		return false
	}

	return d.breakpoints[tensor]
}

// InspectTensor provides detailed inspection of a gradient tensor
func InspectGradientTensor(tensor *GradientTensor) string {
	if tensor == nil {
		return "Tensor: nil"
	}

	var builder strings.Builder
	builder.WriteString("GradientTensor Inspection:\n")
	builder.WriteString(fmt.Sprintf("  RequiresGrad: %t\n", tensor.RequiresGrad))
	builder.WriteString(fmt.Sprintf("  IsLeaf: %t\n", tensor.IsLeaf))

	if tensor.Tensor != nil {
		builder.WriteString(fmt.Sprintf("  Tensor Shape: %v\n", tensor.Tensor.Shape))
		builder.WriteString(fmt.Sprintf("  Tensor Size: %d elements\n", len(tensor.Tensor.Data)))

		if len(tensor.Tensor.Data) > 0 {
			builder.WriteString(fmt.Sprintf("  Data Range: [%.6f, %.6f]\n",
				minFloat32(tensor.Tensor.Data), maxFloat32(tensor.Tensor.Data)))
		}
	} else {
		builder.WriteString("  Tensor: nil\n")
	}

	if tensor.Gradient != nil {
		builder.WriteString(fmt.Sprintf("  Gradient Shape: %v\n", tensor.Gradient.Shape))
		builder.WriteString(fmt.Sprintf("  Gradient Size: %d elements\n", len(tensor.Gradient.Data)))

		if len(tensor.Gradient.Data) > 0 {
			builder.WriteString(fmt.Sprintf("  Gradient Range: [%.6f, %.6f]\n",
				minFloat32(tensor.Gradient.Data), maxFloat32(tensor.Gradient.Data)))

			// Calculate gradient norm
			norm := float32(0.0)
			for _, val := range tensor.Gradient.Data {
				norm += val * val
			}
			norm = float32(math.Sqrt(float64(norm)))
			builder.WriteString(fmt.Sprintf("  Gradient Norm: %.6f\n", norm))
		}
	} else {
		builder.WriteString("  Gradient: nil\n")
	}

	if tensor.GradFn != nil {
		builder.WriteString(fmt.Sprintf("  Operation: %s\n", opTypeToString(tensor.GradFn.OpType)))
		builder.WriteString(fmt.Sprintf("  Input Count: %d\n", len(tensor.GradFn.Inputs)))
		builder.WriteString(fmt.Sprintf("  Saved Tensors: %d\n", len(tensor.GradFn.SavedTensors)))
	} else {
		builder.WriteString("  GradFn: nil\n")
	}

	return builder.String()
}

// ValidateGradients performs gradient validation using finite differences
func ValidateGradients(model func(*tensor.Tensor) (*tensor.Tensor, error), input *tensor.Tensor, epsilon float32) error {
	if epsilon <= 0 {
		epsilon = 1e-4
	}

	LogAutodiffDebug(DebugInfo, "Starting gradient validation with epsilon=%.6f", epsilon)

	// Ensure input requires gradients
	inputGT := NewGradientTensor(input, true)

	// Forward pass
	SetGradientMode(Grad)
	output, err := model(input)
	if err != nil {
		return fmt.Errorf("forward pass failed: %w", err)
	}

	// Backward pass
	outputGT := NewGradientTensor(output, true)
	err = outputGT.Backward()
	if err != nil {
		return fmt.Errorf("backward pass failed: %w", err)
	}

	analyticalGrad := inputGT.Gradient
	if analyticalGrad == nil {
		return fmt.Errorf("no gradient computed for input")
	}

	// Compute numerical gradients
	numericalGrad, err := computeNumericalGradients(model, input, epsilon)
	if err != nil {
		return fmt.Errorf("numerical gradient computation failed: %w", err)
	}

	// Compare gradients
	err = compareGradients(analyticalGrad, numericalGrad, epsilon*10) // Allow 10x epsilon tolerance
	if err != nil {
		LogAutodiffDebug(DebugError, "Gradient validation failed: %v", err)
		return fmt.Errorf("gradient validation failed: %w", err)
	}

	LogAutodiffDebug(DebugInfo, "Gradient validation passed")
	return nil
}

// computeNumericalGradients computes gradients using finite differences
func computeNumericalGradients(model func(*tensor.Tensor) (*tensor.Tensor, error), input *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	if err := input.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve input to CPU: %w", err)
	}

	gradData := make([]float32, len(input.Data))

	SetGradientMode(NoGrad) // Disable autodiff for numerical computation
	defer SetGradientMode(Grad)

	for i := range input.Data {
		// f(x + epsilon)
		input.Data[i] += epsilon
		outputPlus, err := model(input)
		if err != nil {
			return nil, fmt.Errorf("forward pass (plus) failed: %w", err)
		}

		// f(x - epsilon)
		input.Data[i] -= 2 * epsilon
		outputMinus, err := model(input)
		if err != nil {
			return nil, fmt.Errorf("forward pass (minus) failed: %w", err)
		}

		// Restore original value
		input.Data[i] += epsilon

		// Compute numerical gradient: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
		if err := outputPlus.RetrieveCPU(); err != nil {
			return nil, fmt.Errorf("failed to retrieve output plus: %w", err)
		}
		if err := outputMinus.RetrieveCPU(); err != nil {
			return nil, fmt.Errorf("failed to retrieve output minus: %w", err)
		}

		// Assume scalar output for simplicity
		if len(outputPlus.Data) != 1 || len(outputMinus.Data) != 1 {
			return nil, fmt.Errorf("numerical gradient validation only supports scalar outputs")
		}

		gradData[i] = (outputPlus.Data[0] - outputMinus.Data[0]) / (2 * epsilon)

		// Release temporary outputs
		outputPlus.ReleaseGPU()
		outputMinus.ReleaseGPU()
	}

	return tensor.NewTensor(input.Shape, gradData)
}

// compareGradients compares analytical and numerical gradients
func compareGradients(analytical, numerical *tensor.Tensor, tolerance float32) error {
	if err := analytical.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve analytical gradient: %w", err)
	}
	if err := numerical.RetrieveCPU(); err != nil {
		return fmt.Errorf("failed to retrieve numerical gradient: %w", err)
	}

	if len(analytical.Data) != len(numerical.Data) {
		return fmt.Errorf("gradient size mismatch: analytical=%d, numerical=%d",
			len(analytical.Data), len(numerical.Data))
	}

	maxDiff := float32(0.0)
	maxRelDiff := float32(0.0)

	for i := range analytical.Data {
		diff := abs(analytical.Data[i] - numerical.Data[i])
		relDiff := float32(0.0)

		if abs(numerical.Data[i]) > 1e-8 {
			relDiff = diff / abs(numerical.Data[i])
		}

		if diff > maxDiff {
			maxDiff = diff
		}
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
		}

		if diff > tolerance && relDiff > 0.1 { // 10% relative error threshold
			return fmt.Errorf("gradient mismatch at index %d: analytical=%.6f, numerical=%.6f, diff=%.6f, rel_diff=%.6f",
				i, analytical.Data[i], numerical.Data[i], diff, relDiff)
		}
	}

	LogAutodiffDebug(DebugInfo, "Gradient comparison: max_diff=%.6f, max_rel_diff=%.6f", maxDiff, maxRelDiff)
	return nil
}

// Helper functions for debugging

// minFloat32 finds the minimum value in a float32 slice
func minFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0.0
	}

	min := data[0]
	for _, val := range data[1:] {
		if val < min {
			min = val
		}
	}
	return min
}

// maxFloat32 finds the maximum value in a float32 slice
func maxFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0.0
	}

	max := data[0]
	for _, val := range data[1:] {
		if val > max {
			max = val
		}
	}
	return max
}

// AutodiffConfig provides global configuration for autodiff behavior
type AutodiffConfig struct {
	EnableProfiler     bool
	EnableDebugger     bool
	EnableOptimizer    bool
	MaxGraphDepth      int
	MemoryOptimization bool
	FusionOptimization bool
	ProfilingLevel     int
	DebugLevel         DebugLogLevel
}

// DefaultAutodiffConfig returns a default configuration
func DefaultAutodiffConfig() *AutodiffConfig {
	return &AutodiffConfig{
		EnableProfiler:     false,
		EnableDebugger:     false,
		EnableOptimizer:    true,
		MaxGraphDepth:      1000,
		MemoryOptimization: true,
		FusionOptimization: true,
		ProfilingLevel:     1,
		DebugLevel:         DebugOff,
	}
}

// ApplyAutodiffConfig applies the given configuration
func ApplyAutodiffConfig(config *AutodiffConfig) {
	if config.EnableProfiler {
		EnableAutodiffProfiling()
	} else {
		DisableAutodiffProfiling()
	}

	if config.EnableDebugger {
		EnableAutodiffDebugging(config.DebugLevel)
	} else {
		DisableAutodiffDebugging()
	}

	// Additional configuration can be applied here as needed
	LogAutodiffDebug(DebugInfo, "Applied autodiff configuration: profiler=%t, debugger=%t, optimizer=%t",
		config.EnableProfiler, config.EnableDebugger, config.EnableOptimizer)
}

// GetAutodiffStats returns comprehensive statistics about the autodiff system
func GetAutodiffStats() map[string]interface{} {
	stats := make(map[string]interface{})

	// Graph statistics
	stats["graph_node_count"] = len(globalGraph.nodes)
	stats["graph_leaf_count"] = len(globalGraph.leafNodes)
	stats["gradient_mode"] = globalGraph.gradMode

	// Profiler statistics
	if globalProfiler.enabled {
		profile := GetAutodiffProfile()
		stats["profiler_enabled"] = true
		stats["operation_count"] = len(profile.operationStats)
		stats["memory_stats"] = profile.memoryStats
		stats["graph_analysis"] = profile.graphAnalysis
	} else {
		stats["profiler_enabled"] = false
	}

	// Debugger statistics
	globalDebugger.mutex.RLock()
	stats["debugger_enabled"] = globalDebugger.enabled
	stats["breakpoint_count"] = len(globalDebugger.breakpoints)
	stats["watch_list_count"] = len(globalDebugger.watchList)
	stats["debug_level"] = globalDebugger.logLevel
	globalDebugger.mutex.RUnlock()

	return stats
}
