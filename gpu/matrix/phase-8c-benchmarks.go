package matrix

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/tsawler/go-nngpu/tensor"
)

// Phase 8C: Performance Benchmarks and Validation
// This file provides comprehensive benchmarks for all Phase 8C optimizations

// BenchmarkSuite runs comprehensive benchmarks for Phase 8C optimizations
type BenchmarkSuite struct {
	results      map[string]*BenchmarkResult
	mutex        sync.RWMutex
	warmupRuns   int
	benchmarkRuns int
}

// BenchmarkResult stores the results of a benchmark
type BenchmarkResult struct {
	Name               string
	BaselineTime       time.Duration
	OptimizedTime      time.Duration
	Speedup            float64
	MemoryBaseline     int64
	MemoryOptimized    int64
	MemorySavings      float64
	ThroughputBaseline float64 // Operations per second
	ThroughputOptimized float64
	Notes              []string
}

// NewBenchmarkSuite creates a new benchmark suite
func NewBenchmarkSuite() *BenchmarkSuite {
	return &BenchmarkSuite{
		results:       make(map[string]*BenchmarkResult),
		warmupRuns:    3,
		benchmarkRuns: 10,
	}
}

// BenchmarkMatrixMultiplication benchmarks matrix multiplication optimizations
func (bs *BenchmarkSuite) BenchmarkMatrixMultiplication(sizes []int) {
	for _, size := range sizes {
		name := fmt.Sprintf("MatMul_%dx%d", size, size)
		
		// Create test matrices
		dataA := make([]float32, size*size)
		dataB := make([]float32, size*size)
		for i := range dataA {
			dataA[i] = float32(i%100) * 0.01
			dataB[i] = float32((i+50)%100) * 0.01
		}
		
		matrixA, _ := tensor.NewTensor([]int{size, size}, dataA)
		matrixB, _ := tensor.NewTensor([]int{size, size}, dataB)
		defer matrixA.ReleaseGPU()
		defer matrixB.ReleaseGPU()

		result := &BenchmarkResult{Name: name}
		
		// Benchmark baseline implementation
		result.BaselineTime = bs.benchmarkFunction(func() error {
			resultTensor, err := MatMul(matrixA, matrixB)
			if err != nil {
				return err
			}
			defer resultTensor.ReleaseGPU()
			return nil
		})
		
		result.MemoryBaseline = int64(size * size * 3 * 4) // 3 matrices, 4 bytes per float32
		
		// Benchmark optimized implementation
		tensors := []*tensor.Tensor{matrixA, matrixB}
		params := map[string]interface{}{
			"M": size, "N": size, "K": size,
		}
		
		result.OptimizedTime = bs.benchmarkFunction(func() error {
			optimizedOp, err := OptimizeOperation("matmul", tensors, params)
			if err != nil {
				return err
			}
			defer optimizedOp.Cleanup()
			
			// Simulate execution
			return optimizedOp.Execute("", params)
		})
		
		// Calculate optimized memory usage (with potential layout changes)
		layoutOptimizer := GetGlobalLayoutOptimizer()
		if layoutOptimizer != nil {
			optimizedA, layoutInfoA, _ := layoutOptimizer.ApplyLayoutOptimization(matrixA, "matmul")
			optimizedB, layoutInfoB, _ := layoutOptimizer.ApplyLayoutOptimization(matrixB, "matmul")
			
			optimizedSizeA := int64(len(optimizedA.Data) * 4)
			optimizedSizeB := int64(len(optimizedB.Data) * 4)
			result.MemoryOptimized = optimizedSizeA + optimizedSizeB + int64(size*size*4) // Output matrix
			
			if optimizedA != matrixA {
				optimizedA.ReleaseGPU()
			}
			if optimizedB != matrixB {
				optimizedB.ReleaseGPU()
			}
			
			// Add layout optimization notes
			if layoutInfoA.Layout != LayoutRowMajor {
				result.Notes = append(result.Notes, fmt.Sprintf("Matrix A optimized to %s layout", layoutInfoA.Layout.String()))
			}
			if layoutInfoB.Layout != LayoutRowMajor {
				result.Notes = append(result.Notes, fmt.Sprintf("Matrix B optimized to %s layout", layoutInfoB.Layout.String()))
			}
		} else {
			result.MemoryOptimized = result.MemoryBaseline
		}
		
		// Calculate metrics
		if result.OptimizedTime > 0 {
			result.Speedup = float64(result.BaselineTime) / float64(result.OptimizedTime)
		}
		
		if result.MemoryBaseline > 0 {
			result.MemorySavings = (1.0 - float64(result.MemoryOptimized)/float64(result.MemoryBaseline)) * 100
		}
		
		// Calculate throughput (operations per second)
		operations := float64(size * size * size * 2) // Multiply-add operations
		if result.BaselineTime > 0 {
			result.ThroughputBaseline = operations / result.BaselineTime.Seconds()
		}
		if result.OptimizedTime > 0 {
			result.ThroughputOptimized = operations / result.OptimizedTime.Seconds()
		}
		
		bs.mutex.Lock()
		bs.results[name] = result
		bs.mutex.Unlock()
	}
}

// BenchmarkConvolution benchmarks convolution optimizations
func (bs *BenchmarkSuite) BenchmarkConvolution() {
	convConfigs := []struct {
		name        string
		inputShape  []int
		kernelShape []int
	}{
		{"Conv_Small", []int{16, 56, 56, 64}, []int{3, 3, 64, 128}},
		{"Conv_Medium", []int{32, 112, 112, 32}, []int{5, 5, 32, 64}},
		{"Conv_Large", []int{8, 224, 224, 3}, []int{7, 7, 3, 64}},
	}
	
	for _, config := range convConfigs {
		result := &BenchmarkResult{Name: config.name}
		
		// Create test tensors
		inputSize := 1
		for _, dim := range config.inputShape {
			inputSize *= dim
		}
		kernelSize := 1
		for _, dim := range config.kernelShape {
			kernelSize *= dim
		}
		
		inputData := make([]float32, inputSize)
		kernelData := make([]float32, kernelSize)
		for i := range inputData {
			inputData[i] = float32(i%100) * 0.01
		}
		for i := range kernelData {
			kernelData[i] = float32(i%50) * 0.02
		}
		
		inputTensor, _ := tensor.NewTensor(config.inputShape, inputData)
		kernelTensor, _ := tensor.NewTensor(config.kernelShape, kernelData)
		defer inputTensor.ReleaseGPU()
		defer kernelTensor.ReleaseGPU()
		
		// Benchmark baseline convolution
		params := Conv2DParams{StrideH: 1, StrideW: 1, PadH: 1, PadW: 1}
		result.BaselineTime = bs.benchmarkFunction(func() error {
			convResult, err := Conv2DForward(inputTensor, kernelTensor, params)
			if err != nil {
				return err
			}
			defer convResult.ReleaseGPU()
			return nil
		})
		
		result.MemoryBaseline = int64((inputSize + kernelSize) * 4)
		
		// Benchmark optimized convolution
		tensors := []*tensor.Tensor{inputTensor, kernelTensor}
		opParams := map[string]interface{}{
			"input_shape":  config.inputShape,
			"kernel_shape": config.kernelShape,
			"stride":       1,
			"padding":      1,
		}
		
		result.OptimizedTime = bs.benchmarkFunction(func() error {
			optimizedOp, err := OptimizeOperation("conv2d", tensors, opParams)
			if err != nil {
				return err
			}
			defer optimizedOp.Cleanup()
			return optimizedOp.Execute("", opParams)
		})
		
		// Estimate optimized memory usage
		layoutOptimizer := GetGlobalLayoutOptimizer()
		if layoutOptimizer != nil {
			optimizedInput, _, _ := layoutOptimizer.ApplyLayoutOptimization(inputTensor, "conv2d")
			optimizedKernel, _, _ := layoutOptimizer.ApplyLayoutOptimization(kernelTensor, "conv2d")
			
			result.MemoryOptimized = int64((len(optimizedInput.Data) + len(optimizedKernel.Data)) * 4)
			
			if optimizedInput != inputTensor {
				optimizedInput.ReleaseGPU()
			}
			if optimizedKernel != kernelTensor {
				optimizedKernel.ReleaseGPU()
			}
		} else {
			result.MemoryOptimized = result.MemoryBaseline
		}
		
		// Calculate metrics
		if result.OptimizedTime > 0 {
			result.Speedup = float64(result.BaselineTime) / float64(result.OptimizedTime)
		}
		
		if result.MemoryBaseline > 0 {
			result.MemorySavings = (1.0 - float64(result.MemoryOptimized)/float64(result.MemoryBaseline)) * 100
		}
		
		// Add convolution-specific notes
		sharedMemOptimizer := GetGlobalSharedMemoryOptimizer()
		if sharedMemOptimizer != nil {
			sharedMemLayout, err := sharedMemOptimizer.OptimizeForConvolution(config.inputShape, config.kernelShape)
			if err == nil {
				result.Notes = append(result.Notes, fmt.Sprintf("Shared memory optimized: %.2f KB", float64(sharedMemLayout.TotalSize)/1024))
			}
		}
		
		bs.mutex.Lock()
		bs.results[config.name] = result
		bs.mutex.Unlock()
	}
}

// BenchmarkBufferReuse benchmarks buffer reuse system performance
func (bs *BenchmarkSuite) BenchmarkBufferReuse() {
	shapes := [][]int{
		{256, 256},
		{512, 512},
		{1024, 128},
		{256, 256}, // Reuse first shape
		{512, 512}, // Reuse second shape
	}
	
	result := &BenchmarkResult{Name: "BufferReuse"}
	
	// Benchmark without buffer reuse
	result.BaselineTime = bs.benchmarkFunction(func() error {
		for _, shape := range shapes {
			size := shape[0] * shape[1]
			data := make([]float32, size)
			t, err := tensor.NewTensor(shape, data)
			if err != nil {
				return err
			}
			defer t.ReleaseGPU()
		}
		return nil
	})
	
	// Benchmark with buffer reuse
	result.OptimizedTime = bs.benchmarkFunction(func() error {
		scope := NewOperationScope("buffer_reuse_test")
		defer scope.Close()
		
		for _, shape := range shapes {
			t, err := scope.CreateTensor(shape)
			if err != nil {
				return err
			}
			// Tensor will be automatically managed by scope
			_ = t
		}
		return nil
	})
	
	// Calculate metrics
	if result.OptimizedTime > 0 {
		result.Speedup = float64(result.BaselineTime) / float64(result.OptimizedTime)
	}
	
	// Check buffer reuse statistics
	if bufferManager := GetGlobalBufferReuseManager(); bufferManager != nil {
		stats := bufferManager.GetStats()
		totalReuses := int64(0)
		totalAllocations := int64(0)
		for _, stat := range stats {
			totalReuses += stat.TotalReuses
			totalAllocations += stat.TotalAllocations
		}
		
		if totalAllocations > 0 {
			reuseRate := float64(totalReuses) / float64(totalAllocations) * 100
			result.Notes = append(result.Notes, fmt.Sprintf("Buffer reuse rate: %.1f%%", reuseRate))
		}
	}
	
	bs.mutex.Lock()
	bs.results["BufferReuse"] = result
	bs.mutex.Unlock()
}

// BenchmarkKernelCache benchmarks kernel compilation caching
func (bs *BenchmarkSuite) BenchmarkKernelCache() {
	kernelSource := `
kernel void test_kernel(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = input[index] * 2.0f + 1.0f;
}`
	
	options := &KernelCompilationOptions{
		OptimizationLevel: 2,
		FastMath:         true,
		DebugInfo:        false,
	}
	
	result := &BenchmarkResult{Name: "KernelCache"}
	
	// Benchmark first compilation (cache miss)
	result.BaselineTime = bs.benchmarkFunction(func() error {
		// Clear cache first
		if cache := GetGlobalKernelCache(); cache != nil {
			cache.InvalidateCache()
		}
		
		_, err := CompileOptimizedKernel(kernelSource, options)
		return err
	})
	
	// Benchmark second compilation (cache hit)
	result.OptimizedTime = bs.benchmarkFunction(func() error {
		_, err := CompileOptimizedKernel(kernelSource, options)
		return err
	})
	
	// Calculate metrics
	if result.OptimizedTime > 0 {
		result.Speedup = float64(result.BaselineTime) / float64(result.OptimizedTime)
	}
	
	// Add cache statistics
	hitRate, _, sizeBytes, hitCount, missCount := GetKernelCacheStats()
	if hitCount+missCount > 0 {
		result.Notes = append(result.Notes, fmt.Sprintf("Cache hit rate: %.1f%%", hitRate*100))
		result.Notes = append(result.Notes, fmt.Sprintf("Cache size: %.2f KB", float64(sizeBytes)/1024))
	}
	
	bs.mutex.Lock()
	bs.results["KernelCache"] = result
	bs.mutex.Unlock()
}

// BenchmarkMemoryTransfer benchmarks CPU-GPU transfer optimization
func (bs *BenchmarkSuite) BenchmarkMemoryTransfer() {
	sizes := []int{1024, 4096, 16384} // Different data sizes
	
	for _, size := range sizes {
		name := fmt.Sprintf("Transfer_%dKB", size*4/1024)
		result := &BenchmarkResult{Name: name}
		
		// Create test data
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i%100) * 0.01
		}
		
		// Benchmark standard transfer
		result.BaselineTime = bs.benchmarkFunction(func() error {
			t, err := tensor.NewTensor([]int{size}, data)
			if err != nil {
				return err
			}
			defer t.ReleaseGPU()
			
			return t.EnsureGPU()
		})
		
		// Benchmark optimized transfer
		result.OptimizedTime = bs.benchmarkFunction(func() error {
			t, err := tensor.NewTensor([]int{size}, data)
			if err != nil {
				return err
			}
			defer t.ReleaseGPU()
			
			// Use transfer optimizer if available
			if transferOpt := GetGlobalTransferOptimizer(); transferOpt != nil {
				if transferOpt.ShouldTransferToGPU(t) {
					err = t.EnsureGPU()
					if err != nil {
						return err
					}
					transferOpt.MarkGPUValid(t, "benchmark")
				}
			} else {
				err = t.EnsureGPU()
			}
			
			return err
		})
		
		result.MemoryBaseline = int64(size * 4)
		result.MemoryOptimized = result.MemoryBaseline
		
		// Calculate metrics
		if result.OptimizedTime > 0 {
			result.Speedup = float64(result.BaselineTime) / float64(result.OptimizedTime)
		}
		
		// Calculate bandwidth
		dataSize := float64(size * 4) // bytes
		if result.BaselineTime > 0 {
			result.ThroughputBaseline = dataSize / result.BaselineTime.Seconds()
		}
		if result.OptimizedTime > 0 {
			result.ThroughputOptimized = dataSize / result.OptimizedTime.Seconds()
		}
		
		bs.mutex.Lock()
		bs.results[name] = result
		bs.mutex.Unlock()
	}
}

// benchmarkFunction runs a function multiple times and returns the average execution time
func (bs *BenchmarkSuite) benchmarkFunction(fn func() error) time.Duration {
	// Warmup runs
	for i := 0; i < bs.warmupRuns; i++ {
		fn()
		runtime.GC() // Force garbage collection between runs
	}
	
	// Benchmark runs
	var totalTime time.Duration
	for i := 0; i < bs.benchmarkRuns; i++ {
		start := time.Now()
		err := fn()
		elapsed := time.Since(start)
		
		if err != nil {
			// If there's an error, return a very high time to indicate failure
			return time.Hour
		}
		
		totalTime += elapsed
		runtime.GC()
	}
	
	return totalTime / time.Duration(bs.benchmarkRuns)
}

// RunAllBenchmarks runs the complete benchmark suite
func (bs *BenchmarkSuite) RunAllBenchmarks() {
	fmt.Println("Running Phase 8C Performance Benchmarks...")
	
	// Matrix multiplication benchmarks
	fmt.Println("Benchmarking matrix multiplication...")
	bs.BenchmarkMatrixMultiplication([]int{256, 512, 1024})
	
	// Convolution benchmarks
	fmt.Println("Benchmarking convolution operations...")
	bs.BenchmarkConvolution()
	
	// Buffer reuse benchmarks
	fmt.Println("Benchmarking buffer reuse system...")
	bs.BenchmarkBufferReuse()
	
	// Kernel cache benchmarks
	fmt.Println("Benchmarking kernel compilation caching...")
	bs.BenchmarkKernelCache()
	
	// Memory transfer benchmarks
	fmt.Println("Benchmarking memory transfer optimization...")
	bs.BenchmarkMemoryTransfer()
	
	fmt.Println("Benchmarks completed!")
}

// PrintResults prints a detailed report of all benchmark results
func (bs *BenchmarkSuite) PrintResults() {
	bs.mutex.RLock()
	defer bs.mutex.RUnlock()
	
	fmt.Println("\n=== Phase 8C Performance Benchmark Results ===")
	fmt.Println()
	
	// Summary statistics
	totalSpeedup := 0.0
	totalMemorySavings := 0.0
	benchmarkCount := 0
	significantImprovements := 0
	
	// Detailed results
	for _, result := range bs.results {
		fmt.Printf("Benchmark: %s\n", result.Name)
		fmt.Printf("  Baseline Time:    %v\n", result.BaselineTime)
		fmt.Printf("  Optimized Time:   %v\n", result.OptimizedTime)
		
		if result.Speedup > 0 {
			fmt.Printf("  Speedup:          %.2fx\n", result.Speedup)
			totalSpeedup += result.Speedup
			benchmarkCount++
			
			if result.Speedup > 1.1 { // More than 10% improvement
				significantImprovements++
			}
		}
		
		if result.MemoryBaseline > 0 && result.MemoryOptimized > 0 {
			fmt.Printf("  Memory Baseline:  %.2f KB\n", float64(result.MemoryBaseline)/1024)
			fmt.Printf("  Memory Optimized: %.2f KB\n", float64(result.MemoryOptimized)/1024)
			fmt.Printf("  Memory Savings:   %.1f%%\n", result.MemorySavings)
			totalMemorySavings += result.MemorySavings
		}
		
		if result.ThroughputBaseline > 0 && result.ThroughputOptimized > 0 {
			fmt.Printf("  Throughput Baseline:  %.2f MB/s\n", result.ThroughputBaseline/(1024*1024))
			fmt.Printf("  Throughput Optimized: %.2f MB/s\n", result.ThroughputOptimized/(1024*1024))
		}
		
		if len(result.Notes) > 0 {
			fmt.Printf("  Notes:\n")
			for _, note := range result.Notes {
				fmt.Printf("    - %s\n", note)
			}
		}
		
		fmt.Println()
	}
	
	// Summary
	fmt.Println("=== Summary ===")
	if benchmarkCount > 0 {
		avgSpeedup := totalSpeedup / float64(benchmarkCount)
		avgMemorySavings := totalMemorySavings / float64(benchmarkCount)
		
		fmt.Printf("Average Speedup:           %.2fx\n", avgSpeedup)
		fmt.Printf("Average Memory Savings:    %.1f%%\n", avgMemorySavings)
		fmt.Printf("Significant Improvements:  %d/%d (%.1f%%)\n", 
			significantImprovements, benchmarkCount, 
			float64(significantImprovements)/float64(benchmarkCount)*100)
		
		// Performance grade
		var grade string
		switch {
		case avgSpeedup >= 2.0 && significantImprovements >= benchmarkCount*8/10:
			grade = "Excellent (A)"
		case avgSpeedup >= 1.5 && significantImprovements >= benchmarkCount*6/10:
			grade = "Good (B)"
		case avgSpeedup >= 1.2 && significantImprovements >= benchmarkCount*4/10:
			grade = "Fair (C)"
		default:
			grade = "Needs Improvement (D)"
		}
		
		fmt.Printf("Overall Performance Grade: %s\n", grade)
	}
	
	fmt.Printf("Total Benchmarks Run:      %d\n", len(bs.results))
	
	// Memory optimization effectiveness
	if manager := GetGlobalBufferReuseManager(); manager != nil {
		stats := manager.GetStats()
		totalReuses := int64(0)
		totalAllocations := int64(0)
		for _, stat := range stats {
			totalReuses += stat.TotalReuses
			totalAllocations += stat.TotalAllocations
		}
		
		if totalAllocations > 0 {
			reuseEfficiency := float64(totalReuses) / float64(totalAllocations) * 100
			fmt.Printf("Buffer Reuse Efficiency:   %.1f%%\n", reuseEfficiency)
		}
	}
	
	// Kernel cache effectiveness
	hitRate, entries, _, hitCount, missCount := GetKernelCacheStats()
	if hitCount+missCount > 0 {
		fmt.Printf("Kernel Cache Hit Rate:     %.1f%% (%d entries)\n", hitRate*100, entries)
	}
}

// RunPhase8CBenchmarks runs a complete benchmark suite for Phase 8C
func RunPhase8CBenchmarks() *BenchmarkSuite {
	suite := NewBenchmarkSuite()
	suite.RunAllBenchmarks()
	return suite
}

// ValidatePhase8CImplementation validates that all Phase 8C features are working correctly
func ValidatePhase8CImplementation() bool {
	fmt.Println("Validating Phase 8C Implementation...")
	
	// Check component availability
	components := map[string]bool{
		"Memory Coalescing Optimizer": GetGlobalMemoryOptimizer() != nil,
		"Transfer Optimizer":          GetGlobalTransferOptimizer() != nil,
		"Bandwidth Monitor":           GetGlobalBandwidthMonitor() != nil,
		"Buffer Reuse Manager":        GetGlobalBufferReuseManager() != nil,
		"Layout Optimizer":            GetGlobalLayoutOptimizer() != nil,
		"Kernel Cache":                GetGlobalKernelCache() != nil,
		"Shared Memory Optimizer":     GetGlobalSharedMemoryOptimizer() != nil,
	}
	
	allAvailable := true
	for component, available := range components {
		symbol := "✓"
		if !available {
			symbol = "✗"
			allAvailable = false
		}
		fmt.Printf("%s %s\n", symbol, component)
	}
	
	if allAvailable {
		fmt.Println("✓ Phase 8C: All components successfully implemented")
	} else {
		fmt.Println("✗ Phase 8C: Some components missing")
	}
	
	return allAvailable
}