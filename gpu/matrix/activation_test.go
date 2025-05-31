package matrix

import (
	"math"
	"testing"

	"github.com/tsawler/go-nngpu/tensor"
)

// Helper function to check if two float32 slices are approximately equal
func approxEqual(a, b []float32, tolerance float32) bool {
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

// Helper function to create a test tensor
func createTestTensor(shape []int, data []float32) *tensor.Tensor {
	t, err := tensor.NewTensor(shape, data)
	if err != nil {
		panic(err)
	}
	return t
}

func TestReLUForward(t *testing.T) {
	input := createTestTensor([]int{2, 3}, []float32{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0})
	expected := []float32{0.0, 0.0, 0.0, 1.0, 2.0, 3.0}

	result, err := ActivationForward(input, ReLU, 0.0)
	if err != nil {
		t.Fatalf("ReLU forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	if !approxEqual(result.Data, expected, 1e-6) {
		t.Errorf("ReLU forward result mismatch. Expected %v, got %v", expected, result.Data)
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestReLUBackward(t *testing.T) {
	gradOutput := createTestTensor([]int{2, 3}, []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0})
	activationOutput := createTestTensor([]int{2, 3}, []float32{0.0, 0.0, 0.0, 1.0, 2.0, 3.0})
	expected := []float32{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}

	result, err := ActivationBackward(gradOutput, activationOutput, ReLU, 0.0)
	if err != nil {
		t.Fatalf("ReLU backward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	if !approxEqual(result.Data, expected, 1e-6) {
		t.Errorf("ReLU backward result mismatch. Expected %v, got %v", expected, result.Data)
	}

	result.ReleaseGPU()
	gradOutput.ReleaseGPU()
	activationOutput.ReleaseGPU()
}

func TestSigmoidForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{-2.0, -1.0, 0.0, 1.0})
	
	result, err := ActivationForward(input, Sigmoid, 0.0)
	if err != nil {
		t.Fatalf("Sigmoid forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that sigmoid values are between 0 and 1
	for i, val := range result.Data {
		if val < 0.0 || val > 1.0 {
			t.Errorf("Sigmoid value at index %d is out of range [0,1]: %f", i, val)
		}
	}

	// Check specific values
	if math.Abs(float64(result.Data[2])-0.5) > 1e-6 {
		t.Errorf("Sigmoid(0) should be 0.5, got %f", result.Data[2])
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestTanhForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{-2.0, -1.0, 0.0, 1.0})
	
	result, err := ActivationForward(input, Tanh, 0.0)
	if err != nil {
		t.Fatalf("Tanh forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that tanh values are between -1 and 1
	for i, val := range result.Data {
		if val < -1.0 || val > 1.0 {
			t.Errorf("Tanh value at index %d is out of range [-1,1]: %f", i, val)
		}
	}

	// Check specific values
	if math.Abs(float64(result.Data[2])) > 1e-6 {
		t.Errorf("Tanh(0) should be 0, got %f", result.Data[2])
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestSoftmaxForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{1.0, 2.0, 3.0, 4.0})
	
	result, err := ActivationForward(input, Softmax, 0.0)
	if err != nil {
		t.Fatalf("Softmax forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that softmax values sum to 1
	sum := float32(0.0)
	for _, val := range result.Data {
		sum += val
	}
	
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("Softmax values should sum to 1, got %f", sum)
	}

	// Check that all values are positive
	for i, val := range result.Data {
		if val <= 0.0 {
			t.Errorf("Softmax value at index %d should be positive: %f", i, val)
		}
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestSoftmax2D(t *testing.T) {
	// Test batch processing
	input := createTestTensor([]int{2, 3}, []float32{
		1.0, 2.0, 3.0, // First batch
		4.0, 5.0, 6.0, // Second batch
	})
	
	result, err := ActivationForward(input, Softmax, 0.0)
	if err != nil {
		t.Fatalf("Softmax 2D forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that each row sums to 1
	rows := input.Shape[0]
	cols := input.Shape[1]
	
	for row := 0; row < rows; row++ {
		sum := float32(0.0)
		for col := 0; col < cols; col++ {
			sum += result.Data[row*cols+col]
		}
		
		if math.Abs(float64(sum)-1.0) > 1e-6 {
			t.Errorf("Softmax row %d should sum to 1, got %f", row, sum)
		}
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestLeakyReLUForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{-2.0, -1.0, 0.0, 1.0})
	alpha := float32(0.1)
	
	result, err := ActivationForward(input, LeakyReLU, alpha)
	if err != nil {
		t.Fatalf("Leaky ReLU forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	expected := []float32{-0.2, -0.1, 0.0, 1.0}
	if !approxEqual(result.Data, expected, 1e-6) {
		t.Errorf("Leaky ReLU result mismatch. Expected %v, got %v", expected, result.Data)
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestELUForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{-2.0, -1.0, 0.0, 1.0})
	alpha := float32(1.0)
	
	result, err := ActivationForward(input, ELU, alpha)
	if err != nil {
		t.Fatalf("ELU forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that positive values are unchanged
	if math.Abs(float64(result.Data[3])-1.0) > 1e-6 {
		t.Errorf("ELU should preserve positive values, expected 1.0, got %f", result.Data[3])
	}

	// Check that negative values are transformed correctly
	if result.Data[0] >= 0.0 {
		t.Errorf("ELU should transform negative values to negative values")
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestSwishForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{-2.0, -1.0, 0.0, 1.0})
	
	result, err := ActivationForward(input, Swish, 0.0)
	if err != nil {
		t.Fatalf("Swish forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that swish(0) = 0
	if math.Abs(float64(result.Data[2])) > 1e-6 {
		t.Errorf("Swish(0) should be 0, got %f", result.Data[2])
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestGELUForward(t *testing.T) {
	input := createTestTensor([]int{1, 4}, []float32{-2.0, -1.0, 0.0, 1.0})
	
	result, err := ActivationForward(input, GELU, 0.0)
	if err != nil {
		t.Fatalf("GELU forward failed: %v", err)
	}

	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	// Check that GELU(0) = 0
	if math.Abs(float64(result.Data[2])) > 1e-6 {
		t.Errorf("GELU(0) should be 0, got %f", result.Data[2])
	}

	result.ReleaseGPU()
	input.ReleaseGPU()
}

func TestBatchActivationForward(t *testing.T) {
	input1 := createTestTensor([]int{2, 2}, []float32{-1.0, 0.0, 1.0, 2.0})
	input2 := createTestTensor([]int{2, 2}, []float32{-2.0, -1.0, 0.0, 1.0})
	inputs := []*tensor.Tensor{input1, input2}
	
	results, err := BatchActivationForward(inputs, ReLU, 0.0)
	if err != nil {
		t.Fatalf("Batch activation forward failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Clean up
	for _, result := range results {
		result.ReleaseGPU()
	}
	input1.ReleaseGPU()
	input2.ReleaseGPU()
}

func TestBatchActivationBackward(t *testing.T) {
	gradOutput1 := createTestTensor([]int{2, 2}, []float32{1.0, 1.0, 1.0, 1.0})
	gradOutput2 := createTestTensor([]int{2, 2}, []float32{1.0, 1.0, 1.0, 1.0})
	activationOutput1 := createTestTensor([]int{2, 2}, []float32{0.0, 0.0, 1.0, 2.0})
	activationOutput2 := createTestTensor([]int{2, 2}, []float32{0.0, 0.0, 0.0, 1.0})
	
	gradOutputs := []*tensor.Tensor{gradOutput1, gradOutput2}
	activationOutputs := []*tensor.Tensor{activationOutput1, activationOutput2}
	
	results, err := BatchActivationBackward(gradOutputs, activationOutputs, ReLU, 0.0)
	if err != nil {
		t.Fatalf("Batch activation backward failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Clean up
	for _, result := range results {
		result.ReleaseGPU()
	}
	gradOutput1.ReleaseGPU()
	gradOutput2.ReleaseGPU()
	activationOutput1.ReleaseGPU()
	activationOutput2.ReleaseGPU()
}

// Benchmark tests
func BenchmarkReLUForward(b *testing.B) {
	input := createTestTensor([]int{1000, 1000}, make([]float32, 1000000))
	for i := range input.Data {
		input.Data[i] = float32(i%100 - 50) // Mix of positive and negative values
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := ActivationForward(input, ReLU, 0.0)
		if err != nil {
			b.Fatalf("ReLU forward failed: %v", err)
		}
		result.ReleaseGPU()
	}

	input.ReleaseGPU()
}

func BenchmarkSigmoidForward(b *testing.B) {
	input := createTestTensor([]int{1000, 1000}, make([]float32, 1000000))
	for i := range input.Data {
		input.Data[i] = float32(i%100 - 50) // Mix of positive and negative values
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := ActivationForward(input, Sigmoid, 0.0)
		if err != nil {
			b.Fatalf("Sigmoid forward failed: %v", err)
		}
		result.ReleaseGPU()
	}

	input.ReleaseGPU()
}

func BenchmarkSoftmaxForward(b *testing.B) {
	input := createTestTensor([]int{100, 1000}, make([]float32, 100000))
	for i := range input.Data {
		input.Data[i] = float32(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := ActivationForward(input, Softmax, 0.0)
		if err != nil {
			b.Fatalf("Softmax forward failed: %v", err)
		}
		result.ReleaseGPU()
	}

	input.ReleaseGPU()
}