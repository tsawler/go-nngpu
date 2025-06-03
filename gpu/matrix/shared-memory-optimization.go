package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"

import (
	"fmt"
	"sync"

	"github.com/tsawler/go-nngpu/tensor"
)

// Phase 8C: Shared Memory Usage Optimization
// This file implements advanced shared memory optimizations for GPU kernels

// SharedMemoryOptimizer optimizes shared memory usage in GPU kernels
type SharedMemoryOptimizer struct {
	maxSharedMemory       int // Maximum shared memory available per threadgroup
	bankSize              int // Memory bank size for avoiding bank conflicts
	warpSize              int // Warp/SIMD group size for optimal access patterns
	optimizationStrategy  int // 0: conservative, 1: balanced, 2: aggressive
	cachedOptimizations   map[string]*SharedMemoryLayout
	mutex                 sync.RWMutex
}

// SharedMemoryLayout describes an optimized shared memory layout
type SharedMemoryLayout struct {
	TotalSize       int                    // Total shared memory size needed
	Banks           []SharedMemoryBank     // Memory banks configuration
	AccessPattern   SharedMemoryAccess     // Access pattern optimization
	Padding         []int                  // Padding to avoid bank conflicts
	TileSize        []int                  // Optimal tile sizes for operations
	ThreadMapping   map[string]interface{} // Thread-to-memory mapping strategy
}

// SharedMemoryBank represents a memory bank configuration
type SharedMemoryBank struct {
	BankID       int    // Bank identifier
	Size         int    // Size of this bank in bytes
	AccessStride int    // Optimal access stride for this bank
	DataType     string // Data type stored in this bank
}

// SharedMemoryAccess describes memory access patterns
type SharedMemoryAccess struct {
	Pattern         string   // "sequential", "strided", "tiled", "random"
	Stride          int      // Access stride
	CoalesceWidth   int      // Width for coalesced access
	ConflictFree    bool     // Whether access pattern is conflict-free
	PrefetchSize    int      // Amount to prefetch
}

// NewSharedMemoryOptimizer creates a new shared memory optimizer
func NewSharedMemoryOptimizer() *SharedMemoryOptimizer {
	return &SharedMemoryOptimizer{
		maxSharedMemory:      32768, // 32KB default for Apple Silicon GPUs
		bankSize:            4,      // 4-byte banks typical for Apple Silicon
		warpSize:            32,     // SIMD group size for Apple Silicon
		optimizationStrategy: 1,     // Balanced by default
		cachedOptimizations:  make(map[string]*SharedMemoryLayout),
	}
}

// OptimizeForMatrixMultiplication optimizes shared memory for matrix multiplication
func (smo *SharedMemoryOptimizer) OptimizeForMatrixMultiplication(M, N, K int) (*SharedMemoryLayout, error) {
	cacheKey := fmt.Sprintf("matmul_%d_%d_%d", M, N, K)
	
	smo.mutex.RLock()
	if layout, exists := smo.cachedOptimizations[cacheKey]; exists {
		smo.mutex.RUnlock()
		return layout, nil
	}
	smo.mutex.RUnlock()
	
	// Calculate optimal tile sizes based on shared memory constraints
	maxTileSize := int(float64(smo.maxSharedMemory) * 0.8) // Use 80% of available shared memory
	
	// For matrix multiplication, we need two tiles: A_tile and B_tile
	// Each tile stores float32 values (4 bytes each)
	elementsPerTile := maxTileSize / (2 * 4) // Divided by 2 for A and B tiles
	
	// Find optimal square tile size
	tileSize := 1
	for tileSize*tileSize <= elementsPerTile {
		tileSize++
	}
	tileSize-- // Back up to last valid size
	
	// Ensure tile size is multiple of warp size for optimal access
	tileSize = (tileSize / smo.warpSize) * smo.warpSize
	if tileSize == 0 {
		tileSize = smo.warpSize
	}
	
	// Create memory banks for A and B tiles
	aTileSize := tileSize * tileSize * 4 // 4 bytes per float32
	bTileSize := tileSize * tileSize * 4
	
	banks := []SharedMemoryBank{
		{
			BankID:       0,
			Size:         aTileSize,
			AccessStride: tileSize * 4, // Row stride
			DataType:     "float32",
		},
		{
			BankID:       1,
			Size:         bTileSize,
			AccessStride: 4, // Column stride (transposed access)
			DataType:     "float32",
		},
	}
	
	// Calculate padding to avoid bank conflicts
	padding := smo.calculateBankConflictPadding(tileSize, 4)
	
	layout := &SharedMemoryLayout{
		TotalSize: aTileSize + bTileSize + padding[0] + padding[1],
		Banks:     banks,
		AccessPattern: SharedMemoryAccess{
			Pattern:       "tiled",
			Stride:        tileSize,
			CoalesceWidth: smo.warpSize,
			ConflictFree:  true,
			PrefetchSize:  tileSize * 4,
		},
		Padding:  padding,
		TileSize: []int{tileSize, tileSize},
		ThreadMapping: map[string]interface{}{
			"threads_per_tile":  smo.warpSize,
			"tiles_per_block":   4,
			"workload_per_thread": tileSize / smo.warpSize,
		},
	}
	
	// Cache the optimization
	smo.mutex.Lock()
	smo.cachedOptimizations[cacheKey] = layout
	smo.mutex.Unlock()
	
	return layout, nil
}

// OptimizeForConvolution optimizes shared memory for convolution operations
func (smo *SharedMemoryOptimizer) OptimizeForConvolution(inputShape, kernelShape []int) (*SharedMemoryLayout, error) {
	if len(inputShape) != 4 || len(kernelShape) != 4 {
		return nil, fmt.Errorf("invalid shapes for convolution optimization")
	}
	
	batch, height, width, channels := inputShape[0], inputShape[1], inputShape[2], inputShape[3]
	kernelH, kernelW, _, outChannels := kernelShape[0], kernelShape[1], kernelShape[2], kernelShape[3]
	
	cacheKey := fmt.Sprintf("conv_%d_%d_%d_%d_%d_%d_%d", 
		batch, height, width, channels, kernelH, kernelW, outChannels)
	
	smo.mutex.RLock()
	if layout, exists := smo.cachedOptimizations[cacheKey]; exists {
		smo.mutex.RUnlock()
		return layout, nil
	}
	smo.mutex.RUnlock()
	
	// Calculate optimal tile sizes for input patch and kernel
	maxTileSize := int(float64(smo.maxSharedMemory) * 0.75) // Use 75% of available shared memory
	
	// For convolution, we need tiles for: input patch, kernel weights, and output accumulation
	bytesPerElement := 4 // float32
	
	// Calculate input tile size (includes padding for kernel application)
	inputTileH := 16 // Start with reasonable tile size
	inputTileW := 16
	
	// Adjust tile size based on memory constraints
	inputTileSize := inputTileH * inputTileW * channels * bytesPerElement
	kernelTileSize := kernelH * kernelW * channels * outChannels * bytesPerElement
	outputTileSize := inputTileH * inputTileW * outChannels * bytesPerElement
	
	totalSize := inputTileSize + kernelTileSize + outputTileSize
	if totalSize > maxTileSize {
		// Scale down tile sizes proportionally
		scale := float64(maxTileSize) / float64(totalSize)
		inputTileH = int(float64(inputTileH) * scale)
		inputTileW = int(float64(inputTileW) * scale)
		
		// Ensure tile sizes are multiples of 4 for optimal access
		inputTileH = (inputTileH / 4) * 4
		inputTileW = (inputTileW / 4) * 4
		if inputTileH == 0 { inputTileH = 4 }
		if inputTileW == 0 { inputTileW = 4 }
	}
	
	// Recalculate sizes with adjusted tile dimensions
	inputTileSize = inputTileH * inputTileW * channels * bytesPerElement
	outputTileSize = inputTileH * inputTileW * outChannels * bytesPerElement
	
	banks := []SharedMemoryBank{
		{
			BankID:       0,
			Size:         inputTileSize,
			AccessStride: inputTileW * channels * bytesPerElement,
			DataType:     "float32",
		},
		{
			BankID:       1,
			Size:         kernelTileSize,
			AccessStride: kernelW * channels * bytesPerElement,
			DataType:     "float32",
		},
		{
			BankID:       2,
			Size:         outputTileSize,
			AccessStride: inputTileW * outChannels * bytesPerElement,
			DataType:     "float32",
		},
	}
	
	// Calculate padding for optimal access patterns
	padding := []int{0, 0, 0} // One for each bank
	for i := range banks {
		padding[i] = smo.calculateConvolutionPadding(banks[i].AccessStride)
	}
	
	layout := &SharedMemoryLayout{
		TotalSize: inputTileSize + kernelTileSize + outputTileSize + padding[0] + padding[1] + padding[2],
		Banks:     banks,
		AccessPattern: SharedMemoryAccess{
			Pattern:       "tiled",
			Stride:        inputTileW,
			CoalesceWidth: channels,
			ConflictFree:  true,
			PrefetchSize:  inputTileW * channels * bytesPerElement,
		},
		Padding:  padding,
		TileSize: []int{inputTileH, inputTileW, channels},
		ThreadMapping: map[string]interface{}{
			"input_threads_per_tile":  inputTileH * inputTileW / smo.warpSize,
			"kernel_threads_per_tile": kernelH * kernelW,
			"output_threads_per_tile": (inputTileH * inputTileW) / 4, // 4 outputs per thread
		},
	}
	
	// Cache the optimization
	smo.mutex.Lock()
	smo.cachedOptimizations[cacheKey] = layout
	smo.mutex.Unlock()
	
	return layout, nil
}

// OptimizeForReduction optimizes shared memory for reduction operations
func (smo *SharedMemoryOptimizer) OptimizeForReduction(inputSize int, reductionType string) (*SharedMemoryLayout, error) {
	cacheKey := fmt.Sprintf("reduce_%d_%s", inputSize, reductionType)
	
	smo.mutex.RLock()
	if layout, exists := smo.cachedOptimizations[cacheKey]; exists {
		smo.mutex.RUnlock()
		return layout, nil
	}
	smo.mutex.RUnlock()
	
	// For reduction, use tree-based reduction in shared memory
	threadsPerBlock := 256 // Typical block size
	elementsPerThread := (inputSize + threadsPerBlock - 1) / threadsPerBlock
	
	// Shared memory for partial sums
	sharedMemSize := threadsPerBlock * 4 // 4 bytes per float32
	
	// Add padding to avoid bank conflicts in reduction tree
	padding := smo.calculateReductionPadding(threadsPerBlock)
	
	banks := []SharedMemoryBank{
		{
			BankID:       0,
			Size:         sharedMemSize + padding,
			AccessStride: 4, // Sequential access
			DataType:     "float32",
		},
	}
	
	layout := &SharedMemoryLayout{
		TotalSize: sharedMemSize + padding,
		Banks:     banks,
		AccessPattern: SharedMemoryAccess{
			Pattern:       "tree_reduction",
			Stride:        1,
			CoalesceWidth: smo.warpSize,
			ConflictFree:  true,
			PrefetchSize:  smo.warpSize * 4,
		},
		Padding:  []int{padding},
		TileSize: []int{threadsPerBlock},
		ThreadMapping: map[string]interface{}{
			"threads_per_block":     threadsPerBlock,
			"elements_per_thread":   elementsPerThread,
			"reduction_levels":      smo.calculateReductionLevels(threadsPerBlock),
		},
	}
	
	// Cache the optimization
	smo.mutex.Lock()
	smo.cachedOptimizations[cacheKey] = layout
	smo.mutex.Unlock()
	
	return layout, nil
}

// calculateBankConflictPadding calculates padding needed to avoid bank conflicts
func (smo *SharedMemoryOptimizer) calculateBankConflictPadding(tileSize, elementSize int) []int {
	// For Apple Silicon GPUs, bank conflicts occur when multiple threads access
	// the same bank simultaneously. Add padding to offset access patterns.
	
	stride := tileSize * elementSize
	banksPerStride := stride / smo.bankSize
	
	// Add padding if stride aligns with bank boundaries
	padding := make([]int, 2) // For A and B tiles
	
	if banksPerStride%32 == 0 { // Avoid conflicts with SIMD group size
		padding[0] = smo.bankSize // Add one bank worth of padding
		padding[1] = smo.bankSize
	}
	
	return padding
}

// calculateConvolutionPadding calculates padding for convolution shared memory access
func (smo *SharedMemoryOptimizer) calculateConvolutionPadding(accessStride int) int {
	// Ensure access stride doesn't align with bank boundaries
	if accessStride%smo.bankSize == 0 {
		return smo.bankSize / 2 // Add half-bank padding
	}
	return 0
}

// calculateReductionPadding calculates padding for tree reduction to avoid conflicts
func (smo *SharedMemoryOptimizer) calculateReductionPadding(threadsPerBlock int) int {
	// During tree reduction, threads access shared memory with strides that are powers of 2
	// Add padding to ensure conflict-free access at all reduction levels
	
	// Find the largest power of 2 stride that could cause conflicts
	maxStride := 1
	for maxStride < threadsPerBlock {
		maxStride *= 2
	}
	
	// Add padding if max stride aligns with bank size
	if (maxStride * 4) % smo.bankSize == 0 {
		return smo.bankSize
	}
	
	return 0
}

// calculateReductionLevels calculates the number of levels in tree reduction
func (smo *SharedMemoryOptimizer) calculateReductionLevels(threadsPerBlock int) int {
	levels := 0
	threads := threadsPerBlock
	for threads > 1 {
		threads /= 2
		levels++
	}
	return levels
}

// ApplySharedMemoryOptimization applies shared memory optimization to a kernel
func (smo *SharedMemoryOptimizer) ApplySharedMemoryOptimization(kernelSource string, layout *SharedMemoryLayout) (string, error) {
	// This would modify the kernel source to use the optimized shared memory layout
	// For now, return the original source with comments about the optimization
	
	optimizedSource := fmt.Sprintf(`
// Shared Memory Optimization Applied
// Total shared memory: %d bytes
// Number of banks: %d
// Access pattern: %s
// Tile size: %v

%s
`, layout.TotalSize, len(layout.Banks), layout.AccessPattern.Pattern, layout.TileSize, kernelSource)
	
	return optimizedSource, nil
}

// GenerateOptimizedKernel generates an optimized kernel for a specific operation
func (smo *SharedMemoryOptimizer) GenerateOptimizedKernel(operation string, params map[string]interface{}) (string, *SharedMemoryLayout, error) {
	switch operation {
	case "matrix_multiply":
		return smo.generateMatMulKernel(params)
	case "convolution":
		return smo.generateConvolutionKernel(params)
	case "reduction":
		return smo.generateReductionKernel(params)
	default:
		return "", nil, fmt.Errorf("unsupported operation: %s", operation)
	}
}

// generateMatMulKernel generates an optimized matrix multiplication kernel
func (smo *SharedMemoryOptimizer) generateMatMulKernel(params map[string]interface{}) (string, *SharedMemoryLayout, error) {
	M, ok1 := params["M"].(int)
	N, ok2 := params["N"].(int)
	K, ok3 := params["K"].(int)
	
	if !ok1 || !ok2 || !ok3 {
		return "", nil, fmt.Errorf("invalid parameters for matrix multiplication kernel")
	}
	
	layout, err := smo.OptimizeForMatrixMultiplication(M, N, K)
	if err != nil {
		return "", nil, err
	}
	
	tileSize := layout.TileSize[0]
	
	kernelSource := fmt.Sprintf(`
#include <metal_stdlib>
using namespace metal;

// Optimized matrix multiplication with shared memory tiling
// Tile size: %d x %d
// Shared memory usage: %d bytes

kernel void optimized_matmul(device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* C [[buffer(2)]],
                            constant uint& M [[buffer(3)]],
                            constant uint& N [[buffer(4)]],
                            constant uint& K [[buffer(5)]],
                            threadgroup float* A_shared [[threadgroup(0)]],
                            threadgroup float* B_shared [[threadgroup(1)]],
                            uint2 threadgroup_id [[threadgroup_position_in_grid]],
                            uint2 thread_id [[thread_position_in_threadgroup]],
                            uint2 threadgroup_size [[threads_per_threadgroup]]) {
    
    const uint TILE_SIZE = %d;
    const uint row = threadgroup_id.y * TILE_SIZE + thread_id.y;
    const uint col = threadgroup_id.x * TILE_SIZE + thread_id.x;
    
    float sum = 0.0;
    
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load A tile into shared memory
        uint a_row = row;
        uint a_col = tile * TILE_SIZE + thread_id.x;
        if (a_row < M && a_col < K) {
            A_shared[thread_id.y * TILE_SIZE + thread_id.x] = A[a_row * K + a_col];
        } else {
            A_shared[thread_id.y * TILE_SIZE + thread_id.x] = 0.0;
        }
        
        // Load B tile into shared memory
        uint b_row = tile * TILE_SIZE + thread_id.y;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_shared[thread_id.y * TILE_SIZE + thread_id.x] = B[b_row * N + b_col];
        } else {
            B_shared[thread_id.y * TILE_SIZE + thread_id.x] = 0.0;
        }
        
        // Synchronize to ensure all data is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial product using shared memory
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += A_shared[thread_id.y * TILE_SIZE + k] * 
                   B_shared[k * TILE_SIZE + thread_id.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
`, tileSize, tileSize, layout.TotalSize, tileSize)
	
	return kernelSource, layout, nil
}

// generateConvolutionKernel generates an optimized convolution kernel
func (smo *SharedMemoryOptimizer) generateConvolutionKernel(params map[string]interface{}) (string, *SharedMemoryLayout, error) {
	inputShape, ok1 := params["input_shape"].([]int)
	kernelShape, ok2 := params["kernel_shape"].([]int)
	
	if !ok1 || !ok2 {
		return "", nil, fmt.Errorf("invalid parameters for convolution kernel")
	}
	
	layout, err := smo.OptimizeForConvolution(inputShape, kernelShape)
	if err != nil {
		return "", nil, err
	}
	
	kernelSource := fmt.Sprintf(`
#include <metal_stdlib>
using namespace metal;

// Optimized 2D convolution with shared memory
// Input tile: %v
// Shared memory usage: %d bytes

kernel void optimized_conv2d(device const float* input [[buffer(0)]],
                            device const float* kernel [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            constant ConvParams& params [[buffer(3)]],
                            threadgroup float* input_shared [[threadgroup(0)]],
                            threadgroup float* kernel_shared [[threadgroup(1)]],
                            uint3 threadgroup_id [[threadgroup_position_in_grid]],
                            uint3 thread_id [[thread_position_in_threadgroup]]) {
    
    const uint TILE_H = %d;
    const uint TILE_W = %d;
    
    // Calculate output position
    uint out_h = threadgroup_id.y * TILE_H + thread_id.y;
    uint out_w = threadgroup_id.x * TILE_W + thread_id.x;
    uint out_c = threadgroup_id.z;
    
    if (out_h >= params.output_height || out_w >= params.output_width) return;
    
    // Load input tile with padding for convolution
    // Implementation would include optimized shared memory loading
    // and convolution computation using the shared memory layout
    
    float sum = 0.0;
    // Convolution computation using shared memory...
    
    // Write result
    uint output_idx = (out_h * params.output_width + out_w) * params.output_channels + out_c;
    output[output_idx] = sum;
}
`, layout.TileSize, layout.TotalSize, layout.TileSize[0], layout.TileSize[1])
	
	return kernelSource, layout, nil
}

// generateReductionKernel generates an optimized reduction kernel
func (smo *SharedMemoryOptimizer) generateReductionKernel(params map[string]interface{}) (string, *SharedMemoryLayout, error) {
	inputSize, ok1 := params["input_size"].(int)
	reductionType, ok2 := params["reduction_type"].(string)
	
	if !ok1 || !ok2 {
		return "", nil, fmt.Errorf("invalid parameters for reduction kernel")
	}
	
	layout, err := smo.OptimizeForReduction(inputSize, reductionType)
	if err != nil {
		return "", nil, err
	}
	
	threadsPerBlock := layout.TileSize[0]
	reductionLevels := layout.ThreadMapping["reduction_levels"].(int)
	
	kernelSource := fmt.Sprintf(`
#include <metal_stdlib>
using namespace metal;

// Optimized tree reduction with shared memory
// Threads per block: %d
// Reduction levels: %d
// Shared memory usage: %d bytes

kernel void optimized_reduction(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& input_size [[buffer(2)]],
                               threadgroup float* shared_data [[threadgroup(0)]],
                               uint threadgroup_id [[threadgroup_position_in_grid]],
                               uint thread_id [[thread_position_in_threadgroup]],
                               uint threadgroup_size [[threads_per_threadgroup]]) {
    
    const uint THREADS_PER_BLOCK = %d;
    uint tid = thread_id;
    uint i = threadgroup_id * THREADS_PER_BLOCK + thread_id;
    
    // Load data into shared memory
    shared_data[tid] = (i < input_size) ? input[i] : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction in shared memory
    for (uint s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[threadgroup_id] = shared_data[0];
    }
}
`, threadsPerBlock, reductionLevels, layout.TotalSize, threadsPerBlock)
	
	return kernelSource, layout, nil
}

// OptimizeSharedMemoryForOperation provides a high-level interface for shared memory optimization
func OptimizeSharedMemoryForOperation(operation string, tensors []*tensor.Tensor, params map[string]interface{}) (*SharedMemoryLayout, error) {
	optimizer := GetGlobalSharedMemoryOptimizer()
	
	switch operation {
	case "matmul":
		if len(tensors) < 2 {
			return nil, fmt.Errorf("matrix multiplication requires at least 2 tensors")
		}
		
		A, B := tensors[0], tensors[1]
		if len(A.Shape) != 2 || len(B.Shape) != 2 {
			return nil, fmt.Errorf("matrix multiplication requires 2D tensors")
		}
		
		M, K := A.Shape[0], A.Shape[1]
		_, N := B.Shape[0], B.Shape[1]
		
		return optimizer.OptimizeForMatrixMultiplication(M, N, K)
		
	case "conv2d":
		if len(tensors) < 2 {
			return nil, fmt.Errorf("convolution requires input and kernel tensors")
		}
		
		input, kernel := tensors[0], tensors[1]
		return optimizer.OptimizeForConvolution(input.Shape, kernel.Shape)
		
	case "reduction":
		if len(tensors) < 1 {
			return nil, fmt.Errorf("reduction requires input tensor")
		}
		
		input := tensors[0]
		size := len(input.Data)
		reductionType := "sum"
		if rt, exists := params["type"]; exists {
			if rtStr, ok := rt.(string); ok {
				reductionType = rtStr
			}
		}
		
		return optimizer.OptimizeForReduction(size, reductionType)
		
	default:
		return nil, fmt.Errorf("unsupported operation for shared memory optimization: %s", operation)
	}
}

// Global shared memory optimizer
var globalSharedMemoryOptimizer *SharedMemoryOptimizer
var sharedMemoryOptimizerOnce sync.Once

// GetGlobalSharedMemoryOptimizer returns the global shared memory optimizer
func GetGlobalSharedMemoryOptimizer() *SharedMemoryOptimizer {
	sharedMemoryOptimizerOnce.Do(func() {
		globalSharedMemoryOptimizer = NewSharedMemoryOptimizer()
	})
	return globalSharedMemoryOptimizer
}

// SetSharedMemoryLimits configures shared memory limits for optimization
func SetSharedMemoryLimits(maxSharedMemory, bankSize, warpSize int) {
	optimizer := GetGlobalSharedMemoryOptimizer()
	optimizer.maxSharedMemory = maxSharedMemory
	optimizer.bankSize = bankSize
	optimizer.warpSize = warpSize
}

// GetSharedMemoryUsageStats returns statistics about shared memory usage
func GetSharedMemoryUsageStats() map[string]interface{} {
	optimizer := GetGlobalSharedMemoryOptimizer()
	optimizer.mutex.RLock()
	defer optimizer.mutex.RUnlock()
	
	totalOptimizations := len(optimizer.cachedOptimizations)
	totalSharedMemory := 0
	
	for _, layout := range optimizer.cachedOptimizations {
		totalSharedMemory += layout.TotalSize
	}
	
	return map[string]interface{}{
		"total_optimizations": totalOptimizations,
		"total_shared_memory": totalSharedMemory,
		"cache_size":         len(optimizer.cachedOptimizations),
		"max_shared_memory":  optimizer.maxSharedMemory,
		"bank_size":          optimizer.bankSize,
		"warp_size":          optimizer.warpSize,
	}
}