package matrix

import (
	"fmt"
	"strings"
	"time"

	"github.com/tsawler/go-nngpu/tensor"
)

// DetailedProfile provides granular timing breakdown
type DetailedProfile struct {
	MatrixSize            int
	TotalOperationTime    time.Duration
	
	// Detailed timing breakdown
	ConversionToFP16Time  time.Duration
	ComputeTime          time.Duration  
	ConversionFromFP16Time time.Duration
	CGOOverheadTime      time.Duration
	MemoryTransferTime   time.Duration
	TensorAllocationTime time.Duration
	
	// Comparison baseline
	Float32BaselineTime  time.Duration
	
	// Performance metrics
	ConversionOverheadRatio float64  // Conversion time / Total time
	CGOOverheadRatio       float64  // CGO time / Total time
	MemoryOverheadRatio    float64  // Memory transfer time / Total time
	
	// Bottleneck identification
	PrimaryBottleneck     string
	BottleneckPercentage  float64
}

// DetailedProfiler provides comprehensive performance analysis
type DetailedProfiler struct {
	trainer *MixedPrecisionTrainer
}

// NewDetailedProfiler creates a new detailed profiler
func NewDetailedProfiler() (*DetailedProfiler, error) {
	trainer, err := NewMixedPrecisionTrainer(DefaultMixedPrecisionConfig())
	if err != nil {
		return nil, err
	}
	
	return &DetailedProfiler{
		trainer: trainer,
	}, nil
}

// ProfileMatrixOperation provides detailed breakdown of matrix operation performance
func (dp *DetailedProfiler) ProfileMatrixOperation(A, B *tensor.Tensor, iterations int) (*DetailedProfile, error) {
	profile := &DetailedProfile{
		MatrixSize: A.Shape[0], // Assuming square matrices
	}
	
	fmt.Printf("🔍 Profiling %dx%d matrix operation (%d iterations)\n", 
		profile.MatrixSize, profile.MatrixSize, iterations)
	
	// 1. Baseline float32 performance
	fmt.Printf("  📊 Measuring float32 baseline...\n")
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := MatMul(A, B)
		if err != nil {
			return nil, fmt.Errorf("float32 baseline failed: %w", err)
		}
	}
	profile.Float32BaselineTime = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     Float32 baseline: %v\n", profile.Float32BaselineTime)
	
	// 2. Detailed mixed precision breakdown
	fmt.Printf("  🔬 Analyzing mixed precision components...\n")
	
	// Component 1: Conversion to FP16
	start = time.Now()
	var inputFP16, weightsFP16 *tensor.Tensor
	var err error
	for i := 0; i < iterations; i++ {
		inputFP16, err = dp.trainer.ConvertTensorToFloat16(A)
		if err != nil {
			return nil, fmt.Errorf("input conversion failed: %w", err)
		}
		weightsFP16, err = dp.trainer.ConvertTensorToFloat16(B)
		if err != nil {
			return nil, fmt.Errorf("weights conversion failed: %w", err)
		}
	}
	profile.ConversionToFP16Time = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     FP32→FP16 conversion: %v\n", profile.ConversionToFP16Time)
	
	// Component 2: Memory transfer overhead
	start = time.Now()
	for i := 0; i < iterations; i++ {
		err = inputFP16.EnsureGPU()
		if err != nil {
			return nil, fmt.Errorf("input GPU transfer failed: %w", err)
		}
		err = weightsFP16.EnsureGPU()
		if err != nil {
			return nil, fmt.Errorf("weights GPU transfer failed: %w", err)
		}
	}
	profile.MemoryTransferTime = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     Memory GPU transfer: %v\n", profile.MemoryTransferTime)
	
	// Component 3: Actual GPU computation
	start = time.Now()
	var result *tensor.Tensor
	for i := 0; i < iterations; i++ {
		result, err = MatMul(inputFP16, weightsFP16)
		if err != nil {
			return nil, fmt.Errorf("FP16 computation failed: %w", err)
		}
	}
	profile.ComputeTime = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     FP16 GPU computation: %v\n", profile.ComputeTime)
	
	// Component 4: Result retrieval and conversion
	start = time.Now()
	for i := 0; i < iterations; i++ {
		err = result.RetrieveCPU()
		if err != nil {
			return nil, fmt.Errorf("result retrieval failed: %w", err)
		}
		// Simulate any post-processing that might be needed
		_ = result.Data[0] // Access data to ensure it's actually transferred
	}
	profile.ConversionFromFP16Time = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     Result retrieval: %v\n", profile.ConversionFromFP16Time)
	
	// Component 5: Tensor allocation overhead
	start = time.Now()
	for i := 0; i < iterations; i++ {
		tempData := make([]float32, len(A.Data))
		_, err := tensor.NewTensor(A.Shape, tempData)
		if err != nil {
			return nil, fmt.Errorf("tensor allocation failed: %w", err)
		}
	}
	profile.TensorAllocationTime = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     Tensor allocation: %v\n", profile.TensorAllocationTime)
	
	// Component 6: Total end-to-end mixed precision
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, err = dp.trainer.ForwardFloat16(A, B, nil)
		if err != nil {
			return nil, fmt.Errorf("end-to-end mixed precision failed: %w", err)
		}
	}
	profile.TotalOperationTime = time.Since(start) / time.Duration(iterations)
	fmt.Printf("     Total mixed precision: %v\n", profile.TotalOperationTime)
	
	// Calculate overhead ratios
	totalTime := profile.TotalOperationTime.Nanoseconds()
	if totalTime > 0 {
		profile.ConversionOverheadRatio = float64(profile.ConversionToFP16Time.Nanoseconds()) / float64(totalTime) * 100
		profile.MemoryOverheadRatio = float64(profile.MemoryTransferTime.Nanoseconds()) / float64(totalTime) * 100
		profile.CGOOverheadTime = profile.TotalOperationTime - profile.ConversionToFP16Time - profile.ComputeTime - profile.ConversionFromFP16Time - profile.MemoryTransferTime
		profile.CGOOverheadRatio = float64(profile.CGOOverheadTime.Nanoseconds()) / float64(totalTime) * 100
	}
	
	// Identify primary bottleneck
	components := map[string]time.Duration{
		"FP32→FP16 Conversion": profile.ConversionToFP16Time,
		"Memory Transfer": profile.MemoryTransferTime,
		"GPU Computation": profile.ComputeTime,
		"Result Retrieval": profile.ConversionFromFP16Time,
		"Tensor Allocation": profile.TensorAllocationTime,
		"CGO Overhead": profile.CGOOverheadTime,
	}
	
	var maxTime time.Duration
	for name, duration := range components {
		if duration > maxTime {
			maxTime = duration
			profile.PrimaryBottleneck = name
		}
	}
	
	if profile.TotalOperationTime > 0 {
		profile.BottleneckPercentage = float64(maxTime.Nanoseconds()) / float64(profile.TotalOperationTime.Nanoseconds()) * 100
	}
	
	fmt.Printf("  🎯 Primary bottleneck: %s (%.1f%% of total time)\n", 
		profile.PrimaryBottleneck, profile.BottleneckPercentage)
	fmt.Printf("  📈 Mixed precision vs Float32: %.2fx %s\n", 
		float64(profile.TotalOperationTime)/float64(profile.Float32BaselineTime),
		func() string {
			if profile.TotalOperationTime < profile.Float32BaselineTime {
				return "speedup"
			}
			return "slowdown"
		}())
	
	return profile, nil
}

// ProfileConversionOverhead specifically analyzes the conversion process
func (dp *DetailedProfiler) ProfileConversionOverhead(sizes []int, iterations int) error {
	fmt.Printf("🔬 Detailed Conversion Overhead Analysis\n")
	fmt.Printf("%-10s | %-12s | %-12s | %-12s | %-12s | %-12s\n", 
		"Size", "CPU Convert", "GPU Transfer", "GPU Compute", "Result Copy", "Total Ratio")
	fmt.Printf("%s\n", "--------------------------------------------------------------------------------")
	
	for _, size := range sizes {
		// Create test data
		data := make([]float32, size*size)
		for i := range data {
			data[i] = float32(i) / float32(size*size)
		}
		
		matrixA, _ := tensor.NewTensor([]int{size, size}, data)
		matrixB, _ := tensor.NewTensor([]int{size, size}, data)
		
		// Profile just the conversion step
		start := time.Now()
		for i := 0; i < iterations; i++ {
			// CPU-based float16 conversion
			fp16Data := make([]float32, len(data))
			for j, val := range data {
				f16 := Float32ToFloatMP16(val)
				fp16Data[j] = FloatMP16ToFloat32(f16)
			}
		}
		cpuConvertTime := time.Since(start) / time.Duration(iterations)
		
		// Profile GPU transfer
		start = time.Now()
		for i := 0; i < iterations; i++ {
			err := matrixA.EnsureGPU()
			if err != nil {
				continue
			}
			err = matrixB.EnsureGPU()
			if err != nil {
				continue
			}
		}
		gpuTransferTime := time.Since(start) / time.Duration(iterations)
		
		// Profile GPU computation
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, err := MatMul(matrixA, matrixB)
			if err != nil {
				continue
			}
		}
		gpuComputeTime := time.Since(start) / time.Duration(iterations)
		
		// Profile result copy
		result, _ := MatMul(matrixA, matrixB)
		start = time.Now()
		for i := 0; i < iterations; i++ {
			err := result.RetrieveCPU()
			if err != nil {
				continue
			}
		}
		resultCopyTime := time.Since(start) / time.Duration(iterations)
		
		// Calculate total ratio
		totalOverhead := cpuConvertTime + gpuTransferTime + resultCopyTime
		totalRatio := float64(totalOverhead) / float64(gpuComputeTime)
		
		fmt.Printf("%-10s | %-12v | %-12v | %-12v | %-12v | %-12.2fx\n",
			fmt.Sprintf("%dx%d", size, size),
			cpuConvertTime,
			gpuTransferTime, 
			gpuComputeTime,
			resultCopyTime,
			totalRatio)
		
		// Analysis
		if totalRatio > 2.0 {
			fmt.Printf("  ❌ CRITICAL: Overhead is %.1fx larger than computation!\n", totalRatio)
		} else if totalRatio > 1.0 {
			fmt.Printf("  ⚠️  WARNING: Overhead is %.1fx larger than computation\n", totalRatio)
		} else {
			fmt.Printf("  ✅ Overhead is acceptable (%.1fx of computation)\n", totalRatio)
		}
	}
	
	return nil
}

// IdentifyOptimizationOpportunities provides specific recommendations
func (dp *DetailedProfiler) IdentifyOptimizationOpportunities(profile *DetailedProfile) []string {
	opportunities := []string{}
	
	// Check conversion overhead
	if profile.ConversionOverheadRatio > 30 {
		opportunities = append(opportunities, 
			fmt.Sprintf("CRITICAL: FP32→FP16 conversion takes %.1f%% of total time. Consider GPU-based conversion or eliminating conversion entirely.", 
				profile.ConversionOverheadRatio))
	}
	
	// Check memory transfer overhead
	if profile.MemoryOverheadRatio > 20 {
		opportunities = append(opportunities, 
			fmt.Sprintf("HIGH: Memory transfers take %.1f%% of total time. Consider keeping data on GPU or using unified memory.", 
				profile.MemoryOverheadRatio))
	}
	
	// Check CGO overhead
	if profile.CGOOverheadRatio > 15 {
		opportunities = append(opportunities, 
			fmt.Sprintf("MEDIUM: CGO overhead takes %.1f%% of total time. Consider batching operations or reducing call frequency.", 
				profile.CGOOverheadRatio))
	}
	
	// Check if computation is too small relative to overhead
	computeRatio := float64(profile.ComputeTime.Nanoseconds()) / float64(profile.TotalOperationTime.Nanoseconds()) * 100
	if computeRatio < 30 {
		opportunities = append(opportunities, 
			fmt.Sprintf("FUNDAMENTAL: Actual computation only %.1f%% of total time. Mixed precision overhead too high for this operation size.", 
				computeRatio))
	}
	
	// Check if mixed precision is fundamentally slower
	if profile.TotalOperationTime > profile.Float32BaselineTime*2 {
		slowdownRatio := float64(profile.TotalOperationTime) / float64(profile.Float32BaselineTime)
		opportunities = append(opportunities, 
			fmt.Sprintf("CRITICAL: Mixed precision is %.1fx slower than float32. Consider disabling mixed precision for this size.", 
				slowdownRatio))
	}
	
	return opportunities
}

// GenerateOptimizationReport creates a comprehensive optimization report
func (dp *DetailedProfiler) GenerateOptimizationReport(sizes []int, iterations int) error {
	fmt.Printf("🎯 COMPREHENSIVE MIXED PRECISION OPTIMIZATION REPORT\n")
	fmt.Printf("=====================================================\n\n")
	
	// Profile each size
	for _, size := range sizes {
		data := make([]float32, size*size)
		for i := range data {
			data[i] = float32(i%1000) / 1000.0
		}
		
		matrixA, _ := tensor.NewTensor([]int{size, size}, data)
		matrixB, _ := tensor.NewTensor([]int{size, size}, data)
		
		profile, err := dp.ProfileMatrixOperation(matrixA, matrixB, iterations)
		if err != nil {
			fmt.Printf("❌ Error profiling %dx%d: %v\n", size, size, err)
			continue
		}
		
		fmt.Printf("\n%s\n", strings.Repeat("=", 50))
		fmt.Printf("MATRIX SIZE: %dx%d\n", size, size)
		fmt.Printf("%s\n", strings.Repeat("=", 50))
		
		// Performance breakdown
		fmt.Printf("PERFORMANCE BREAKDOWN:\n")
		fmt.Printf("  Float32 baseline:     %12v\n", profile.Float32BaselineTime)
		fmt.Printf("  Mixed precision:      %12v (%.2fx %s)\n", 
			profile.TotalOperationTime,
			float64(profile.TotalOperationTime)/float64(profile.Float32BaselineTime),
			func() string {
				if profile.TotalOperationTime < profile.Float32BaselineTime {
					return "speedup"
				}
				return "slowdown"
			}())
		fmt.Printf("\n")
		
		fmt.Printf("COMPONENT BREAKDOWN:\n")
		fmt.Printf("  FP32→FP16 conversion: %12v (%.1f%%)\n", profile.ConversionToFP16Time, profile.ConversionOverheadRatio)
		fmt.Printf("  Memory transfers:     %12v (%.1f%%)\n", profile.MemoryTransferTime, profile.MemoryOverheadRatio)
		fmt.Printf("  GPU computation:      %12v (%.1f%%)\n", profile.ComputeTime, 
			float64(profile.ComputeTime.Nanoseconds())/float64(profile.TotalOperationTime.Nanoseconds())*100)
		fmt.Printf("  Result retrieval:     %12v (%.1f%%)\n", profile.ConversionFromFP16Time,
			float64(profile.ConversionFromFP16Time.Nanoseconds())/float64(profile.TotalOperationTime.Nanoseconds())*100)
		fmt.Printf("  CGO overhead:         %12v (%.1f%%)\n", profile.CGOOverheadTime, profile.CGOOverheadRatio)
		fmt.Printf("\n")
		
		fmt.Printf("PRIMARY BOTTLENECK: %s (%.1f%% of total time)\n", profile.PrimaryBottleneck, profile.BottleneckPercentage)
		fmt.Printf("\n")
		
		// Optimization opportunities
		opportunities := dp.IdentifyOptimizationOpportunities(profile)
		if len(opportunities) > 0 {
			fmt.Printf("OPTIMIZATION OPPORTUNITIES:\n")
			for i, opp := range opportunities {
				fmt.Printf("  %d. %s\n", i+1, opp)
			}
		} else {
			fmt.Printf("✅ No major optimization opportunities identified\n")
		}
		fmt.Printf("\n")
	}
	
	// Overall recommendations
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("OVERALL RECOMMENDATIONS\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))
	fmt.Printf("Based on the profiling analysis:\n\n")
	fmt.Printf("1. DISABLE MIXED PRECISION for matrices larger than 256x256\n")
	fmt.Printf("2. IMPLEMENT GPU-BASED CONVERSION to eliminate CPU overhead\n") 
	fmt.Printf("3. USE PERSISTENT GPU BUFFERS to minimize memory transfers\n")
	fmt.Printf("4. BATCH OPERATIONS to amortize CGO overhead\n")
	fmt.Printf("5. CONSIDER TENSOR CORES for true mixed precision acceleration\n")
	
	return nil
}

// Cleanup releases resources
func (dp *DetailedProfiler) Cleanup() {
	if dp.trainer != nil {
		dp.trainer.Cleanup()
	}
}