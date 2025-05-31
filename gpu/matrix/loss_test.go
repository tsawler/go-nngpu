package matrix

import (
	"math"
	"testing"
)

// Helper function to check if two float32 values are approximately equal
func approxEqualFloat32(a, b, tolerance float32) bool {
	return math.Abs(float64(a-b)) < float64(tolerance)
}

func TestMSELossForward(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{1.0, 2.0, 3.0, 4.0})
	targets := createTestTensor([]int{4}, []float32{1.5, 2.5, 3.5, 4.5})

	loss, err := MSELoss(predictions, targets)
	if err != nil {
		t.Fatalf("MSE loss failed: %v", err)
	}

	// Expected: (0.5^2 + 0.5^2 + 0.5^2 + 0.5^2) / 4 = 0.25
	expected := float32(0.25)
	if !approxEqualFloat32(loss, expected, 1e-6) {
		t.Errorf("MSE loss mismatch. Expected %f, got %f", expected, loss)
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
}

func TestMSELossBackward(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{1.0, 2.0, 3.0, 4.0})
	targets := createTestTensor([]int{4}, []float32{1.5, 2.5, 3.5, 4.5})
	expected := []float32{-0.25, -0.25, -0.25, -0.25} // 2/4 * (pred - target)

	gradients, err := MSELossGradients(predictions, targets)
	if err != nil {
		t.Fatalf("MSE gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	if !approxEqual(gradients.Data, expected, 1e-6) {
		t.Errorf("MSE gradients mismatch. Expected %v, got %v", expected, gradients.Data)
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestBinaryCrossEntropyLossForward(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{0.1, 0.4, 0.6, 0.9})
	targets := createTestTensor([]int{4}, []float32{0.0, 0.0, 1.0, 1.0})

	loss, err := BinaryCrossEntropyLoss(predictions, targets)
	if err != nil {
		t.Fatalf("Binary cross-entropy loss failed: %v", err)
	}

	// Should be a positive value
	if loss <= 0 {
		t.Errorf("Binary cross-entropy loss should be positive, got %f", loss)
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
}

func TestBinaryCrossEntropyLossBackward(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{0.1, 0.4, 0.6, 0.9})
	targets := createTestTensor([]int{4}, []float32{0.0, 0.0, 1.0, 1.0})

	gradients, err := BinaryCrossEntropyLossGradients(predictions, targets)
	if err != nil {
		t.Fatalf("Binary cross-entropy gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	// Check that gradients have reasonable values
	for i, grad := range gradients.Data {
		if math.IsNaN(float64(grad)) || math.IsInf(float64(grad), 0) {
			t.Errorf("Gradient at index %d is NaN or Inf: %f", i, grad)
		}
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestCategoricalCrossEntropyLoss(t *testing.T) {
	// 2 samples, 3 classes
	predictions := createTestTensor([]int{2, 3}, []float32{
		0.2, 0.3, 0.5, // Sample 1: predicted probabilities
		0.1, 0.8, 0.1, // Sample 2: predicted probabilities
	})
	targets := createTestTensor([]int{2, 3}, []float32{
		0.0, 0.0, 1.0, // Sample 1: true class is 2
		0.0, 1.0, 0.0, // Sample 2: true class is 1
	})

	loss, err := CategoricalCrossEntropyLoss(predictions, targets)
	if err != nil {
		t.Fatalf("Categorical cross-entropy loss failed: %v", err)
	}

	if loss <= 0 {
		t.Errorf("Categorical cross-entropy loss should be positive, got %f", loss)
	}

	gradients, err := CategoricalCrossEntropyLossGradients(predictions, targets)
	if err != nil {
		t.Fatalf("Categorical cross-entropy gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestSparseCategoricalCrossEntropyLoss(t *testing.T) {
	// 3 samples, 4 classes
	predictions := createTestTensor([]int{3, 4}, []float32{
		0.1, 0.2, 0.3, 0.4, // Sample 1
		0.4, 0.3, 0.2, 0.1, // Sample 2
		0.25, 0.25, 0.25, 0.25, // Sample 3
	})
	targetIndices := []int{3, 0, 1} // True classes: 3, 0, 1

	loss, err := SparseCategoricalCrossEntropyForward(predictions, targetIndices)
	if err != nil {
		t.Fatalf("Sparse categorical cross-entropy loss failed: %v", err)
	}

	if loss <= 0 {
		t.Errorf("Sparse categorical cross-entropy loss should be positive, got %f", loss)
	}

	gradients, err := SparseCategoricalCrossEntropyBackward(predictions, targetIndices)
	if err != nil {
		t.Fatalf("Sparse categorical cross-entropy gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	// Check that gradients are non-zero only for target classes
	batchSize := predictions.Shape[0]
	numClasses := predictions.Shape[1]

	for b := 0; b < batchSize; b++ {
		targetClass := targetIndices[b]
		for c := 0; c < numClasses; c++ {
			idx := b*numClasses + c
			if c == targetClass {
				// Gradient should be non-zero for target class
				if gradients.Data[idx] == 0.0 {
					t.Errorf("Expected non-zero gradient for target class %d in batch %d", c, b)
				}
			} else {
				// Gradient should be zero for non-target classes
				if gradients.Data[idx] != 0.0 {
					t.Errorf("Expected zero gradient for non-target class %d in batch %d, got %f", c, b, gradients.Data[idx])
				}
			}
		}
	}

	predictions.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestHuberLoss(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{1.0, 2.0, 3.0, 10.0})
	targets := createTestTensor([]int{4}, []float32{1.5, 2.5, 3.5, 4.0})
	delta := float32(1.0)

	loss, err := HuberLoss(predictions, targets, delta)
	if err != nil {
		t.Fatalf("Huber loss failed: %v", err)
	}

	if loss <= 0 {
		t.Errorf("Huber loss should be positive, got %f", loss)
	}

	gradients, err := HuberLossGradients(predictions, targets, delta)
	if err != nil {
		t.Fatalf("Huber gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	// Check that gradients are reasonable
	for i, grad := range gradients.Data {
		if math.IsNaN(float64(grad)) || math.IsInf(float64(grad), 0) {
			t.Errorf("Gradient at index %d is NaN or Inf: %f", i, grad)
		}
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestMAELoss(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{1.0, 2.0, 3.0, 4.0})
	targets := createTestTensor([]int{4}, []float32{1.5, 2.5, 3.5, 4.5})

	loss, err := MAELoss(predictions, targets)
	if err != nil {
		t.Fatalf("MAE loss failed: %v", err)
	}

	// Expected: (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
	expected := float32(0.5)
	if !approxEqualFloat32(loss, expected, 1e-6) {
		t.Errorf("MAE loss mismatch. Expected %f, got %f", expected, loss)
	}

	gradients, err := MAELossGradients(predictions, targets)
	if err != nil {
		t.Fatalf("MAE gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	// Expected gradients: sign(pred - target) / n = [-1, -1, -1, -1] / 4
	expectedGrads := []float32{-0.25, -0.25, -0.25, -0.25}
	if !approxEqual(gradients.Data, expectedGrads, 1e-6) {
		t.Errorf("MAE gradients mismatch. Expected %v, got %v", expectedGrads, gradients.Data)
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestHingeLoss(t *testing.T) {
	predictions := createTestTensor([]int{4}, []float32{0.5, -0.5, 1.5, -1.5})
	targets := createTestTensor([]int{4}, []float32{1.0, 1.0, 1.0, -1.0}) // Binary labels: +1 or -1

	loss, err := HingeLoss(predictions, targets)
	if err != nil {
		t.Fatalf("Hinge loss failed: %v", err)
	}

	if loss < 0 {
		t.Errorf("Hinge loss should be non-negative, got %f", loss)
	}

	gradients, err := HingeLossGradients(predictions, targets)
	if err != nil {
		t.Fatalf("Hinge gradients failed: %v", err)
	}

	if err := gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
	gradients.ReleaseGPU()
}

func TestLossForwardBackward(t *testing.T) {
	predictions := createTestTensor([]int{3}, []float32{1.0, 2.0, 3.0})
	targets := createTestTensor([]int{3}, []float32{1.5, 2.5, 3.5})

	result, err := LossForwardBackward(predictions, targets, MSE)
	if err != nil {
		t.Fatalf("LossForwardBackward failed: %v", err)
	}
	defer result.ReleaseGPU()

	// Check that we got both loss and gradients
	if result.Loss <= 0 {
		t.Errorf("Loss should be positive, got %f", result.Loss)
	}

	if result.Gradients == nil {
		t.Errorf("Gradients should not be nil")
	}

	if err := result.Gradients.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve gradients: %v", err)
	}

	if len(result.Gradients.Data) != 3 {
		t.Errorf("Expected 3 gradient values, got %d", len(result.Gradients.Data))
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
}

func TestInvalidLossInputs(t *testing.T) {
	// Test mismatched shapes
	predictions := createTestTensor([]int{3}, []float32{1.0, 2.0, 3.0})
	targets := createTestTensor([]int{4}, []float32{1.0, 2.0, 3.0, 4.0})

	_, err := MSELoss(predictions, targets)
	if err == nil {
		t.Errorf("Expected error for mismatched shapes")
	}

	// Test invalid Huber loss (missing delta parameter)
	_, err = LossForward(predictions, predictions, Huber)
	if err == nil {
		t.Errorf("Expected error for Huber loss without delta parameter")
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
}

// Benchmark tests
func BenchmarkMSELossForward(b *testing.B) {
	predictions := createTestTensor([]int{1000}, make([]float32, 1000))
	targets := createTestTensor([]int{1000}, make([]float32, 1000))

	// Initialize with random-ish values
	for i := range predictions.Data {
		predictions.Data[i] = float32(i % 100)
		targets.Data[i] = float32((i + 50) % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := MSELoss(predictions, targets)
		if err != nil {
			b.Fatalf("MSE loss failed: %v", err)
		}
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
}

func BenchmarkBinaryCrossEntropyLossForward(b *testing.B) {
	predictions := createTestTensor([]int{1000}, make([]float32, 1000))
	targets := createTestTensor([]int{1000}, make([]float32, 1000))

	// Initialize with valid probability values
	for i := range predictions.Data {
		predictions.Data[i] = float32(i%2)*0.8 + 0.1 // 0.1 or 0.9
		targets.Data[i] = float32(i % 2)             // 0 or 1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := BinaryCrossEntropyLoss(predictions, targets)
		if err != nil {
			b.Fatalf("Binary cross-entropy loss failed: %v", err)
		}
	}

	predictions.ReleaseGPU()
	targets.ReleaseGPU()
}

func BenchmarkSparseCategoricalCrossEntropyForward(b *testing.B) {
	batchSize := 100
	numClasses := 10
	predictions := createTestTensor([]int{batchSize, numClasses}, make([]float32, batchSize*numClasses))
	targetIndices := make([]int, batchSize)

	// Initialize with softmax-like values and random targets
	for i := 0; i < batchSize; i++ {
		sum := float32(0)
		for j := 0; j < numClasses; j++ {
			val := float32(j+1) / float32(numClasses+1)
			predictions.Data[i*numClasses+j] = val
			sum += val
		}
		// Normalize to make it look like softmax output
		for j := 0; j < numClasses; j++ {
			predictions.Data[i*numClasses+j] /= sum
		}
		targetIndices[i] = i % numClasses
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := SparseCategoricalCrossEntropyForward(predictions, targetIndices)
		if err != nil {
			b.Fatalf("Sparse categorical cross-entropy loss failed: %v", err)
		}
	}

	predictions.ReleaseGPU()
}
