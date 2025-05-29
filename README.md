# go-nngpu

Very beginnings. Below app works, but compile code in module first: `go build ./...`

```go
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/tsawler/go-nngpu/gpu/matrix"
	"github.com/tsawler/go-nngpu/tensor"
)

func main() {
	// Example Matrix A (2x3)
	// A = [[1, 2, 3],
	//      [4, 5, 6]]
	aData := []float32{1, 2, 3, 4, 5, 6}
	A, err := tensor.NewTensor([]int{2, 3}, aData)
	if err != nil {
		log.Fatalf("Failed to create tensor A: %v", err)
	}
	defer A.ReleaseGPU() // Ensure GPU memory is released when A goes out of scope

	// Example Matrix B (3x2)
	// B = [[7, 8],
	//      [9, 10],
	//      [11, 12]]
	bData := []float32{7, 8, 9, 10, 11, 12}
	B, err := tensor.NewTensor([]int{3, 2}, bData)
	if err != nil {
		log.Fatalf("Failed to create tensor B: %v", err)
	}
	defer B.ReleaseGPU() // Ensure GPU memory is released

	fmt.Println("=== Small Matrix Test (2x3 * 3x2) ===")

	// --- CPU Matrix Multiplication ---
	fmt.Println("\nPerforming CPU matrix multiplication...")
	start := time.Now()
	cpuResult := matMulCPU(A.Data, A.Shape, B.Data, B.Shape)
	cpuDuration := time.Since(start)
	fmt.Printf("CPU MatMul took %s\n", cpuDuration)

	fmt.Printf("Result C (CPU):\n")
	printMatrix(cpuResult, A.Shape[0], B.Shape[1])

	// --- GPU Matrix Multiplication ---
	fmt.Println("\nPerforming GPU matrix multiplication...")
	start = time.Now()
	C, err := matrix.MatMul(A, B)
	if err != nil {
		log.Fatalf("GPU MatMul failed: %v", err)
	}
	gpuDuration := time.Since(start)
	fmt.Printf("GPU MatMul took %s\n", gpuDuration)

	if err := C.RetrieveCPU(); err != nil {
		log.Fatalf("Failed to retrieve result from GPU: %v", err)
	}
	defer C.ReleaseGPU() // Ensure GPU memory is released for C

	fmt.Printf("Result C (GPU):\n")
	printMatrix(C.Data, C.Shape[0], C.Shape[1])

	// Expected Result (C = A * B)
	// C = [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
	//      [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
	// C = [[7 + 18 + 33, 8 + 20 + 36],
	//      [28 + 45 + 66, 32 + 50 + 72]]
	// C = [[58, 64],
	//      [139, 154]]
	expected := []float32{58, 64, 139, 154}
	fmt.Println("\nExpected Result:")
	printMatrix(expected, C.Shape[0], C.Shape[1])

	// Verification - Compare CPU and GPU results
	fmt.Println("\n=== Verification ===")
	
	// Verify CPU result
	cpuCorrect := true
	for i := range cpuResult {
		if cpuResult[i] != expected[i] {
			fmt.Printf("CPU verification failed at index %d: Got %f, Expected %f\n", i, cpuResult[i], expected[i])
			cpuCorrect = false
		}
	}
	if cpuCorrect {
		fmt.Println("✓ CPU result matches expected values")
	}

	// Verify GPU result
	gpuCorrect := true
	for i := range C.Data {
		if C.Data[i] != expected[i] {
			fmt.Printf("GPU verification failed at index %d: Got %f, Expected %f\n", i, C.Data[i], expected[i])
			gpuCorrect = false
		}
	}
	if gpuCorrect {
		fmt.Println("✓ GPU result matches expected values")
	}

	// Compare CPU vs GPU results
	resultsMatch := compareResults(cpuResult, C.Data, 1e-5)
	if resultsMatch {
		fmt.Println("✓ CPU and GPU results match")
	} else {
		fmt.Println("✗ CPU and GPU results differ!")
	}

	fmt.Printf("\nPerformance comparison (small matrix): GPU was %.2fx the speed of CPU\n", 
		float64(cpuDuration)/float64(gpuDuration))

	// --- Large Matrix Test ---
	fmt.Println("\n=== Large Matrix Test ===")
	rowsA := 1024
	colsA := 512
	rowsB := 512
	colsB := 1024

	rand.Seed(time.Now().UnixNano())
	largeAData := make([]float32, rowsA*colsA)
	largeBData := make([]float32, rowsB*colsB)
	for i := range largeAData {
		largeAData[i] = rand.Float32() * 10
	}
	for i := range largeBData {
		largeBData[i] = rand.Float32() * 10
	}

	largeA, _ := tensor.NewTensor([]int{rowsA, colsA}, largeAData)
	largeB, _ := tensor.NewTensor([]int{rowsB, colsB}, largeBData)
	defer largeA.ReleaseGPU()
	defer largeB.ReleaseGPU()

	fmt.Printf("Multiplying %dx%d by %dx%d matrices...\n", rowsA, colsA, rowsB, colsB)

	// CPU computation for large matrix
	fmt.Println("Performing large CPU matrix multiplication...")
	start = time.Now()
	largeCPUResult := matMulCPU(largeA.Data, largeA.Shape, largeB.Data, largeB.Shape)
	largeCPUDuration := time.Since(start)
	fmt.Printf("Large CPU MatMul took %s\n", largeCPUDuration)

	// GPU computation for large matrix
	fmt.Println("Performing large GPU matrix multiplication...")
	start = time.Now()
	largeC, err := matrix.MatMul(largeA, largeB)
	if err != nil {
		log.Fatalf("Large GPU MatMul failed: %v", err)
	}
	largeGPUDuration := time.Since(start)
	fmt.Printf("Large GPU MatMul took %s\n", largeGPUDuration)

	// Retrieve GPU result for comparison
	if err := largeC.RetrieveCPU(); err != nil {
		log.Fatalf("Failed to retrieve large result from GPU: %v", err)
	}
	defer largeC.ReleaseGPU()

	// Compare large matrix results
	largeResultsMatch := compareResults(largeCPUResult, largeC.Data, 1e-3) // Slightly larger tolerance for large matrices
	if largeResultsMatch {
		fmt.Println("✓ Large matrix CPU and GPU results match")
	} else {
		fmt.Println("✗ Large matrix CPU and GPU results differ!")
		// Print some sample differences for debugging
		printSampleDifferences(largeCPUResult, largeC.Data, 10)
	}

	speedup := float64(largeCPUDuration) / float64(largeGPUDuration)
	fmt.Printf("\nPerformance comparison (large matrix): GPU was %.2fx the speed of CPU\n", speedup)
	
	if speedup > 1.0 {
		fmt.Printf("GPU acceleration achieved! Time saved: %s\n", largeCPUDuration-largeGPUDuration)
	} else {
		fmt.Println("Note: For this size matrix, GPU overhead might outweigh benefits. Try larger matrices for better GPU performance.")
	}
}

// matMulCPU performs naive CPU matrix multiplication
func matMulCPU(aData []float32, aShape []int, bData []float32, bShape []int) []float32 {
	rowsA, colsA := aShape[0], aShape[1]
	rowsB, colsB := bShape[0], bShape[1]
	
	if colsA != rowsB {
		panic(fmt.Sprintf("incompatible matrix dimensions: %dx%d * %dx%d", rowsA, colsA, rowsB, colsB))
	}
	
	result := make([]float32, rowsA*colsB)
	
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := float32(0)
			for k := 0; k < colsA; k++ {
				sum += aData[i*colsA+k] * bData[k*colsB+j]
			}
			result[i*colsB+j] = sum
		}
	}
	
	return result
}

// compareResults compares two float32 slices with a given tolerance
func compareResults(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > float64(tolerance) {
			return false
		}
	}
	return true
}

// printSampleDifferences prints the first n differences between two result arrays
func printSampleDifferences(a, b []float32, n int) {
	fmt.Printf("Sample differences (first %d):\n", n)
	count := 0
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-3 && count < n {
			fmt.Printf("  Index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n", i, a[i], b[i], a[i]-b[i])
			count++
		}
	}
}

func printMatrix(data []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		fmt.Print("[")
		for j := 0; j < cols; j++ {
			fmt.Printf("%.2f", data[i*cols+j])
			if j < cols-1 {
				fmt.Print(", ")
			}
		}
		fmt.Println("]")
	}
}
```

Output should be something like...

```bash
% go run .
=== Small Matrix Test (2x3 * 3x2) ===

Performing CPU matrix multiplication...
CPU MatMul took 19.292µs
Result C (CPU):
[58.00, 64.00]
[139.00, 154.00]

Performing GPU matrix multiplication...
GPU MatMul took 71.746833ms
Result C (GPU):
[58.00, 64.00]
[139.00, 154.00]

Expected Result:
[58.00, 64.00]
[139.00, 154.00]

=== Verification ===
✓ CPU result matches expected values
✓ GPU result matches expected values
✓ CPU and GPU results match

Performance comparison (small matrix): GPU was 0.00x the speed of CPU

=== Large Matrix Test ===
Multiplying 1024x512 by 512x1024 matrices...
Performing large CPU matrix multiplication...
Large CPU MatMul took 429.120916ms
Performing large GPU matrix multiplication...
Large GPU MatMul took 2.931833ms
✓ Large matrix CPU and GPU results match

Performance comparison (large matrix): GPU was 146.37x the speed of CPU
GPU acceleration achieved! Time saved: 426.189083ms
```