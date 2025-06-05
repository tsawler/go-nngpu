package matrix

import (
	"math"
	"testing"

	"github.com/tsawler/gometal/tensor"
	"gonum.org/v1/gonum/mat"
)

// Helper function to create test data
func createTestBatchData(batchSize, features int) *tensor.Tensor {
	data := make([]float32, batchSize*features)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < features; j++ {
			// Create some variation in the data
			data[i*features+j] = float32(i*features+j) * 0.1
		}
	}

	tensor, err := tensor.NewTensor([]int{batchSize, features}, data)
	if err != nil {
		panic(err)
	}
	return tensor
}

// Helper function to create test parameters
func createTestParameters(features int) (*tensor.Tensor, *tensor.Tensor) {
	// Gamma (scale) - initialize to 1
	gammaData := make([]float32, features)
	for i := 0; i < features; i++ {
		gammaData[i] = 1.0
	}
	gamma, _ := tensor.NewTensor([]int{features}, gammaData)

	// Beta (shift) - initialize to 0
	betaData := make([]float32, features)
	beta, _ := tensor.NewTensor([]int{features}, betaData)

	return gamma, beta
}

func TestBatchMean(t *testing.T) {
	batchSize, features := 4, 3
	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	mean, err := BatchMean(input)
	if err != nil {
		t.Fatalf("BatchMean failed: %v", err)
	}
	defer mean.ReleaseGPU()

	if err := mean.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve mean data: %v", err)
	}

	// Check shape
	if len(mean.Shape) != 1 || mean.Shape[0] != features {
		t.Errorf("Expected mean shape [%d], got %v", features, mean.Shape)
	}

	// Verify mean computation manually for first feature
	if err := input.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve input data: %v", err)
	}

	expectedMean := float32(0)
	for i := 0; i < batchSize; i++ {
		expectedMean += input.Data[i*features+0] // First feature
	}
	expectedMean /= float32(batchSize)

	if math.Abs(float64(mean.Data[0]-expectedMean)) > 1e-6 {
		t.Errorf("Expected mean[0] = %f, got %f", expectedMean, mean.Data[0])
	}

	t.Logf("BatchMean test passed. Mean values: %v", mean.Data)
}

func TestBatchVariance(t *testing.T) {
	batchSize, features := 4, 3
	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	mean, err := BatchMean(input)
	if err != nil {
		t.Fatalf("BatchMean failed: %v", err)
	}
	defer mean.ReleaseGPU()

	variance, err := BatchVariance(input, mean)
	if err != nil {
		t.Fatalf("BatchVariance failed: %v", err)
	}
	defer variance.ReleaseGPU()

	if err := variance.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve variance data: %v", err)
	}

	// Check shape
	if len(variance.Shape) != 1 || variance.Shape[0] != features {
		t.Errorf("Expected variance shape [%d], got %v", features, variance.Shape)
	}

	// All variances should be positive
	for i, v := range variance.Data {
		if v < 0 {
			t.Errorf("Variance[%d] should be non-negative, got %f", i, v)
		}
	}

	t.Logf("BatchVariance test passed. Variance values: %v", variance.Data)
}

func TestBatchNormForward(t *testing.T) {
	batchSize, features := 4, 3
	epsilon := float32(1e-5)

	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	// Compute statistics
	mean, err := BatchMean(input)
	if err != nil {
		t.Fatalf("BatchMean failed: %v", err)
	}
	defer mean.ReleaseGPU()

	variance, err := BatchVariance(input, mean)
	if err != nil {
		t.Fatalf("BatchVariance failed: %v", err)
	}
	defer variance.ReleaseGPU()

	// Create parameters
	gamma, beta := createTestParameters(features)
	defer gamma.ReleaseGPU()
	defer beta.ReleaseGPU()

	// Apply batch normalization
	output, err := BatchNormForward(input, mean, variance, gamma, beta, epsilon)
	if err != nil {
		t.Fatalf("BatchNormForward failed: %v", err)
	}
	defer output.ReleaseGPU()

	if err := output.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve output data: %v", err)
	}

	// Check shape
	if len(output.Shape) != 2 || output.Shape[0] != batchSize || output.Shape[1] != features {
		t.Errorf("Expected output shape [%d, %d], got %v", batchSize, features, output.Shape)
	}

	// Verify normalization: compute mean and variance of output for each feature
	if err := mean.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve mean data: %v", err)
	}
	if err := variance.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve variance data: %v", err)
	}

	for f := 0; f < features; f++ {
		// Compute output mean for this feature
		outputMean := float32(0)
		for b := 0; b < batchSize; b++ {
			outputMean += output.Data[b*features+f]
		}
		outputMean /= float32(batchSize)

		// Should be close to 0 (since beta = 0)
		if math.Abs(float64(outputMean)) > 1e-5 {
			t.Errorf("Feature %d output mean should be ~0, got %f", f, outputMean)
		}
	}

	t.Logf("BatchNormForward test passed")
}

func TestBatchNormBackward(t *testing.T) {
	batchSize, features := 4, 3
	epsilon := float32(1e-5)

	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	// Forward pass
	mean, _ := BatchMean(input)
	defer mean.ReleaseGPU()

	variance, _ := BatchVariance(input, mean)
	defer variance.ReleaseGPU()

	gamma, beta := createTestParameters(features)
	defer gamma.ReleaseGPU()
	defer beta.ReleaseGPU()

	output, _ := BatchNormForward(input, mean, variance, gamma, beta, epsilon)
	defer output.ReleaseGPU()

	// Create gradient output (simulate loss gradient)
	gradOutputData := make([]float32, batchSize*features)
	for i := range gradOutputData {
		gradOutputData[i] = 1.0 // Uniform gradient
	}
	gradOutput, _ := tensor.NewTensor([]int{batchSize, features}, gradOutputData)
	defer gradOutput.ReleaseGPU()

	// Backward pass
	gradients, err := BatchNormBackward(gradOutput, input, mean, variance, gamma, epsilon)
	if err != nil {
		t.Fatalf("BatchNormBackward failed: %v", err)
	}
	defer gradients.ReleaseGPU()

	// Check gradient shapes
	if len(gradients.GradInput.Shape) != 2 || gradients.GradInput.Shape[0] != batchSize || gradients.GradInput.Shape[1] != features {
		t.Errorf("Expected grad input shape [%d, %d], got %v", batchSize, features, gradients.GradInput.Shape)
	}

	if len(gradients.GradGamma.Shape) != 1 || gradients.GradGamma.Shape[0] != features {
		t.Errorf("Expected grad gamma shape [%d], got %v", features, gradients.GradGamma.Shape)
	}

	if len(gradients.GradBeta.Shape) != 1 || gradients.GradBeta.Shape[0] != features {
		t.Errorf("Expected grad beta shape [%d], got %v", features, gradients.GradBeta.Shape)
	}

	t.Logf("BatchNormBackward test passed")
}

func TestLayerNormForward(t *testing.T) {
	batchSize, features := 4, 3
	epsilon := float32(1e-5)

	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	gamma, beta := createTestParameters(features)
	defer gamma.ReleaseGPU()
	defer beta.ReleaseGPU()

	result, err := LayerNormForward(input, gamma, beta, epsilon)
	if err != nil {
		t.Fatalf("LayerNormForward failed: %v", err)
	}
	defer result.ReleaseGPU()

	// Check output shape
	if len(result.Output.Shape) != 2 || result.Output.Shape[0] != batchSize || result.Output.Shape[1] != features {
		t.Errorf("Expected output shape [%d, %d], got %v", batchSize, features, result.Output.Shape)
	}

	// Check mean and variance shapes (per-sample)
	if len(result.Mean.Shape) != 1 || result.Mean.Shape[0] != batchSize {
		t.Errorf("Expected mean shape [%d], got %v", batchSize, result.Mean.Shape)
	}

	if len(result.Variance.Shape) != 1 || result.Variance.Shape[0] != batchSize {
		t.Errorf("Expected variance shape [%d], got %v", batchSize, result.Variance.Shape)
	}

	t.Logf("LayerNormForward test passed")
}

func TestBatchNormTraining(t *testing.T) {
	batchSize, features := 4, 3
	epsilon := float32(1e-5)

	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	gamma, beta := createTestParameters(features)
	defer gamma.ReleaseGPU()
	defer beta.ReleaseGPU()

	result, err := BatchNormTraining(input, gamma, beta, epsilon)
	if err != nil {
		t.Fatalf("BatchNormTraining failed: %v", err)
	}
	defer result.ReleaseGPU()

	// Should return output, mean, and variance
	if result.Output == nil || result.Mean == nil || result.Variance == nil {
		t.Fatalf("BatchNormTraining should return output, mean, and variance")
	}

	t.Logf("BatchNormTraining test passed")
}

func TestUpdateRunningStats(t *testing.T) {
	features := 3
	momentum := float32(0.1)

	// Create initial running statistics
	runningMeanData := make([]float32, features)
	runningVarData := make([]float32, features)
	for i := 0; i < features; i++ {
		runningVarData[i] = 1.0 // Initialize to 1
	}

	runningMean, _ := tensor.NewTensor([]int{features}, runningMeanData)
	defer runningMean.ReleaseGPU()

	runningVar, _ := tensor.NewTensor([]int{features}, runningVarData)
	defer runningVar.ReleaseGPU()

	// Create batch statistics
	batchMeanData := []float32{1.0, 2.0, 3.0}
	batchVarData := []float32{0.5, 1.5, 2.5}

	batchMean, _ := tensor.NewTensor([]int{features}, batchMeanData)
	defer batchMean.ReleaseGPU()

	batchVar, _ := tensor.NewTensor([]int{features}, batchVarData)
	defer batchVar.ReleaseGPU()

	// Update running statistics
	err := UpdateRunningStats(runningMean, runningVar, batchMean, batchVar, momentum)
	if err != nil {
		t.Fatalf("UpdateRunningStats failed: %v", err)
	}

	// Retrieve and check updated values
	if err := runningMean.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve running mean: %v", err)
	}
	if err := runningVar.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve running var: %v", err)
	}

	// Check that values were updated correctly
	// running = momentum * running + (1 - momentum) * batch
	expectedMean0 := momentum*0.0 + (1.0-momentum)*1.0
	if math.Abs(float64(runningMean.Data[0]-expectedMean0)) > 1e-6 {
		t.Errorf("Expected running mean[0] = %f, got %f", expectedMean0, runningMean.Data[0])
	}

	expectedVar0 := momentum*1.0 + (1.0-momentum)*0.5
	if math.Abs(float64(runningVar.Data[0]-expectedVar0)) > 1e-6 {
		t.Errorf("Expected running var[0] = %f, got %f", expectedVar0, runningVar.Data[0])
	}

	t.Logf("UpdateRunningStats test passed")
}

func TestInstanceNormForward(t *testing.T) {
	batchSize, channels, height, width := 2, 3, 4, 4
	epsilon := float32(1e-5)

	// Create 4D input data
	inputData := make([]float32, batchSize*channels*height*width)
	for i := range inputData {
		inputData[i] = float32(i) * 0.1
	}

	input, _ := tensor.NewTensor([]int{batchSize, channels, height, width}, inputData)
	defer input.ReleaseGPU()

	// Create parameters (per channel)
	gammaData := make([]float32, channels)
	betaData := make([]float32, channels)
	for i := 0; i < channels; i++ {
		gammaData[i] = 1.0
		betaData[i] = 0.0
	}

	gamma, _ := tensor.NewTensor([]int{channels}, gammaData)
	defer gamma.ReleaseGPU()

	beta, _ := tensor.NewTensor([]int{channels}, betaData)
	defer beta.ReleaseGPU()

	output, err := InstanceNormForward(input, gamma, beta, epsilon)
	if err != nil {
		t.Fatalf("InstanceNormForward failed: %v", err)
	}
	defer output.ReleaseGPU()

	// Check output shape matches input
	if len(output.Shape) != 4 {
		t.Errorf("Expected 4D output, got %dD", len(output.Shape))
	}

	for i, dim := range input.Shape {
		if output.Shape[i] != dim {
			t.Errorf("Output shape mismatch at dimension %d: expected %d, got %d", i, dim, output.Shape[i])
		}
	}

	t.Logf("InstanceNormForward test passed")
}

func TestGroupNormForward(t *testing.T) {
	batchSize, channels, height, width := 2, 4, 3, 3
	numGroups := 2
	epsilon := float32(1e-5)

	// Create 4D input data
	inputData := make([]float32, batchSize*channels*height*width)
	for i := range inputData {
		inputData[i] = float32(i) * 0.1
	}

	input, _ := tensor.NewTensor([]int{batchSize, channels, height, width}, inputData)
	defer input.ReleaseGPU()

	// Create parameters (per channel)
	gammaData := make([]float32, channels)
	betaData := make([]float32, channels)
	for i := 0; i < channels; i++ {
		gammaData[i] = 1.0
		betaData[i] = 0.0
	}

	gamma, _ := tensor.NewTensor([]int{channels}, gammaData)
	defer gamma.ReleaseGPU()

	beta, _ := tensor.NewTensor([]int{channels}, betaData)
	defer beta.ReleaseGPU()

	output, err := GroupNormForward(input, gamma, beta, numGroups, epsilon)
	if err != nil {
		t.Fatalf("GroupNormForward failed: %v", err)
	}
	defer output.ReleaseGPU()

	// Check output shape matches input
	if len(output.Shape) != 4 {
		t.Errorf("Expected 4D output, got %dD", len(output.Shape))
	}

	for i, dim := range input.Shape {
		if output.Shape[i] != dim {
			t.Errorf("Output shape mismatch at dimension %d: expected %d, got %d", i, dim, output.Shape[i])
		}
	}

	t.Logf("GroupNormForward test passed")
}

// Test Gonum compatibility layer
func TestGPUBatchNormLayer(t *testing.T) {
	features := 3
	epsilon := float32(1e-5)
	momentum := float32(0.1)

	layer := NewGPUBatchNormLayer(features, epsilon, momentum)
	defer layer.ReleaseGPU()

	// Create test input
	batchSize := 4
	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	// Test training mode
	layer.SetTraining(true)
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	defer output.ReleaseGPU()

	// Check output shape
	if len(output.Shape) != 2 || output.Shape[0] != batchSize || output.Shape[1] != features {
		t.Errorf("Expected output shape [%d, %d], got %v", batchSize, features, output.Shape)
	}

	// Test inference mode
	layer.SetTraining(false)
	inferenceOutput, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Inference forward pass failed: %v", err)
	}
	defer inferenceOutput.ReleaseGPU()

	t.Logf("GPUBatchNormLayer test passed")
}

func TestGPULayerNormLayer(t *testing.T) {
	features := 3
	epsilon := float32(1e-5)

	layer := NewGPULayerNormLayer(features, epsilon)
	defer layer.ReleaseGPU()

	// Create test input
	batchSize := 4
	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	// Forward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	defer output.ReleaseGPU()

	// Check output shape
	if len(output.Shape) != 2 || output.Shape[0] != batchSize || output.Shape[1] != features {
		t.Errorf("Expected output shape [%d, %d], got %v", batchSize, features, output.Shape)
	}

	t.Logf("GPULayerNormLayer test passed")
}

func TestGonumBatchNormalize(t *testing.T) {
	// Create test matrices
	batchSize, features := 4, 3

	inputData := make([]float64, batchSize*features)
	for i := range inputData {
		inputData[i] = float64(i) * 0.1
	}
	input := mat.NewDense(batchSize, features, inputData)

	// Create parameters
	gamma, beta := CreateBatchNormParams(features)
	runningMean, runningVar := InitializeRunningStats(features)

	// Test training mode
	result := GPUBatchNormalize(input, gamma, beta, runningMean, runningVar, 1e-5, true)

	// Check result shape
	rows, cols := result.Dims()
	if rows != batchSize || cols != features {
		t.Errorf("Expected result dims (%d, %d), got (%d, %d)", batchSize, features, rows, cols)
	}

	t.Logf("GPUBatchNormalize test passed")
}

func TestGonumLayerNormalize(t *testing.T) {
	// Create test matrices
	batchSize, features := 4, 3

	inputData := make([]float64, batchSize*features)
	for i := range inputData {
		inputData[i] = float64(i) * 0.1
	}
	input := mat.NewDense(batchSize, features, inputData)

	// Create parameters
	gamma, beta := CreateLayerNormParams(features)

	// Apply layer normalization
	result := GPULayerNormalize(input, gamma, beta, 1e-5)

	// Check result shape
	rows, cols := result.Dims()
	if rows != batchSize || cols != features {
		t.Errorf("Expected result dims (%d, %d), got (%d, %d)", batchSize, features, rows, cols)
	}

	t.Logf("GPULayerNormalize test passed")
}

// Benchmark tests
func BenchmarkBatchNormForward(b *testing.B) {
	batchSize, features := 64, 512
	epsilon := float32(1e-5)

	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	mean, _ := BatchMean(input)
	defer mean.ReleaseGPU()

	variance, _ := BatchVariance(input, mean)
	defer variance.ReleaseGPU()

	gamma, beta := createTestParameters(features)
	defer gamma.ReleaseGPU()
	defer beta.ReleaseGPU()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		output, _ := BatchNormForward(input, mean, variance, gamma, beta, epsilon)
		output.ReleaseGPU()
	}
}

func BenchmarkLayerNormForward(b *testing.B) {
	batchSize, features := 64, 512
	epsilon := float32(1e-5)

	input := createTestBatchData(batchSize, features)
	defer input.ReleaseGPU()

	gamma, beta := createTestParameters(features)
	defer gamma.ReleaseGPU()
	defer beta.ReleaseGPU()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, _ := LayerNormForward(input, gamma, beta, epsilon)
		result.ReleaseGPU()
	}
}
