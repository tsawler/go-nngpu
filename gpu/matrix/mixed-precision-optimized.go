package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"time"

	"github.com/tsawler/gometal/tensor"
)

// OptimizedMixedPrecisionOps provides optimized mixed precision operations
type OptimizedMixedPrecisionOps struct {
	trainer *MixedPrecisionTrainer
	// Cache for converted tensors to avoid repeated conversions
	tensorCache map[string]*tensor.Tensor
	enabled     bool
}

// NewOptimizedMixedPrecisionOps creates optimized mixed precision operations
func NewOptimizedMixedPrecisionOps(config *MixedPrecisionConfig) (*OptimizedMixedPrecisionOps, error) {
	trainer, err := NewMixedPrecisionTrainer(config)
	if err != nil {
		return nil, err
	}

	return &OptimizedMixedPrecisionOps{
		trainer:     trainer,
		tensorCache: make(map[string]*tensor.Tensor),
		enabled:     config.Enabled,
	}, nil
}

// MatMulOptimized performs optimized matrix multiplication with minimal overhead
func (opt *OptimizedMixedPrecisionOps) MatMulOptimized(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if !opt.enabled {
		return MatMul(A, B)
	}

	// For larger matrices, use float32 to avoid conversion overhead
	sizeA := A.Shape[0] * A.Shape[1]
	sizeB := B.Shape[0] * B.Shape[1]
	totalSize := sizeA + sizeB

	// Use mixed precision only for smaller matrices where conversion overhead is justified
	if totalSize > 512*512 {
		return MatMul(A, B)
	}

	// For smaller matrices, use mixed precision
	return opt.trainer.ForwardFloat16(A, B, nil)
}

// BatchedMatMulOptimized performs batched matrix operations with optimal precision selection
func (opt *OptimizedMixedPrecisionOps) BatchedMatMulOptimized(matrices [][]*tensor.Tensor) ([]*tensor.Tensor, error) {
	results := make([]*tensor.Tensor, len(matrices))

	for i, pair := range matrices {
		if len(pair) != 2 {
			return nil, fmt.Errorf("each matrix pair must contain exactly 2 matrices")
		}

		result, err := opt.MatMulOptimized(pair[0], pair[1])
		if err != nil {
			return nil, fmt.Errorf("failed batch operation %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// PerformanceBenchmark provides comprehensive performance analysis
type PerformanceBenchmark struct {
	MatrixSize         int
	Float32Time        time.Duration
	MixedPrecisionTime time.Duration
	OptimizedTime      time.Duration
	ConversionTime     time.Duration
	ActualComputeTime  time.Duration
	MemoryUsage        int64
	Accuracy           float64
	Speedup            float64
	EffectiveSpeedup   float64
}

// BenchmarkMatrixOperation provides detailed performance analysis
func (opt *OptimizedMixedPrecisionOps) BenchmarkMatrixOperation(size int, iterations int) (*PerformanceBenchmark, error) {
	// Create test data
	dataA := make([]float32, size*size)
	dataB := make([]float32, size*size)

	for i := range dataA {
		dataA[i] = float32(i%1000)/1000.0 - 0.5 // Range [-0.5, 0.5]
		dataB[i] = float32((i*7)%1000)/1000.0 - 0.5
	}

	matrixA, _ := tensor.NewTensor([]int{size, size}, dataA)
	matrixB, _ := tensor.NewTensor([]int{size, size}, dataB)

	benchmark := &PerformanceBenchmark{MatrixSize: size}

	// Benchmark float32 operations
	start := time.Now()
	var result32 *tensor.Tensor
	var err error
	for i := 0; i < iterations; i++ {
		result32, err = MatMul(matrixA, matrixB)
		if err != nil {
			return nil, fmt.Errorf("float32 multiplication failed: %w", err)
		}
	}
	benchmark.Float32Time = time.Since(start) / time.Duration(iterations)

	// Benchmark conversion time separately
	start = time.Now()
	var inputF16, weightsF16 *tensor.Tensor
	for i := 0; i < iterations; i++ {
		inputF16, err = opt.trainer.ConvertTensorToFloat16(matrixA)
		if err != nil {
			return nil, fmt.Errorf("input conversion failed: %w", err)
		}
		weightsF16, err = opt.trainer.ConvertTensorToFloat16(matrixB)
		if err != nil {
			return nil, fmt.Errorf("weights conversion failed: %w", err)
		}
	}
	benchmark.ConversionTime = time.Since(start) / time.Duration(iterations)

	// Benchmark actual compute time on pre-converted tensors
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, err = MatMul(inputF16, weightsF16)
		if err != nil {
			return nil, fmt.Errorf("float16 computation failed: %w", err)
		}
	}
	benchmark.ActualComputeTime = time.Since(start) / time.Duration(iterations)

	// Benchmark full mixed precision pipeline
	start = time.Now()
	var resultMP *tensor.Tensor
	for i := 0; i < iterations; i++ {
		resultMP, err = opt.trainer.ForwardFloat16(matrixA, matrixB, nil)
		if err != nil {
			return nil, fmt.Errorf("mixed precision multiplication failed: %w", err)
		}
	}
	benchmark.MixedPrecisionTime = time.Since(start) / time.Duration(iterations)

	// Benchmark optimized version
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, err = opt.MatMulOptimized(matrixA, matrixB)
		if err != nil {
			return nil, fmt.Errorf("optimized multiplication failed: %w", err)
		}
	}
	benchmark.OptimizedTime = time.Since(start) / time.Duration(iterations)

	// Calculate accuracy
	if err := result32.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve float32 result: %w", err)
	}
	if err := resultMP.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve mixed precision result: %w", err)
	}

	var totalError float64
	for i := range result32.Data {
		diff := float64(result32.Data[i] - resultMP.Data[i])
		totalError += diff * diff
	}
	benchmark.Accuracy = totalError / float64(len(result32.Data)) // MSE

	// Calculate performance metrics
	benchmark.Speedup = float64(benchmark.Float32Time) / float64(benchmark.MixedPrecisionTime)
	benchmark.EffectiveSpeedup = float64(benchmark.Float32Time) / float64(benchmark.OptimizedTime)

	// Estimate memory usage (rough approximation)
	benchmark.MemoryUsage = int64(size * size * 8) // Two matrices in float32

	return benchmark, nil
}

// Cleanup releases resources
func (opt *OptimizedMixedPrecisionOps) Cleanup() {
	if opt.trainer != nil {
		opt.trainer.Cleanup()
	}
	// Clear tensor cache
	opt.tensorCache = make(map[string]*tensor.Tensor)
}

// AdaptivePrecisionSelector automatically chooses the best precision based on operation characteristics
type AdaptivePrecisionSelector struct {
	// Thresholds for automatic precision selection
	SmallMatrixThreshold int     // Use mixed precision for matrices smaller than this
	AccuracyThreshold    float64 // Minimum acceptable accuracy
	PerformanceThreshold float64 // Minimum speedup required for mixed precision

	// Performance history for adaptive learning
	performanceHistory map[int]float64 // size -> best observed speedup
	accuracyHistory    map[int]float64 // size -> best observed accuracy
}

// NewAdaptivePrecisionSelector creates a new adaptive precision selector
func NewAdaptivePrecisionSelector() *AdaptivePrecisionSelector {
	return &AdaptivePrecisionSelector{
		SmallMatrixThreshold: 512,
		AccuracyThreshold:    1e-4,
		PerformanceThreshold: 1.1, // At least 10% speedup required
		performanceHistory:   make(map[int]float64),
		accuracyHistory:      make(map[int]float64),
	}
}

// ShouldUseMixedPrecision determines if mixed precision should be used for a given operation
func (aps *AdaptivePrecisionSelector) ShouldUseMixedPrecision(matrixSize int, operationType string) bool {
	// For very large matrices, avoid mixed precision due to conversion overhead
	if matrixSize > 1024 {
		return false
	}

	// For small matrices, check if we have performance history
	if historicalSpeedup, exists := aps.performanceHistory[matrixSize]; exists {
		return historicalSpeedup > aps.PerformanceThreshold
	}

	// Default decision based on size
	return matrixSize <= aps.SmallMatrixThreshold
}

// UpdatePerformanceHistory updates the performance history with new measurements
func (aps *AdaptivePrecisionSelector) UpdatePerformanceHistory(matrixSize int, speedup, accuracy float64) {
	aps.performanceHistory[matrixSize] = speedup
	aps.accuracyHistory[matrixSize] = accuracy

	// Adapt thresholds based on observed performance
	if speedup > aps.PerformanceThreshold && matrixSize > aps.SmallMatrixThreshold {
		aps.SmallMatrixThreshold = matrixSize
	}
}
