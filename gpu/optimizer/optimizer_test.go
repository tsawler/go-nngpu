package optimizer_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/tsawler/go-nngpu/gpu/matrix"
	"github.com/tsawler/go-nngpu/gpu/optimizer"
	"github.com/tsawler/go-nngpu/tensor"
)

// TestSGDOptimizer demonstrates basic SGD usage
func TestSGDOptimizer(t *testing.T) {
	// Create some test parameters (weights for a simple linear layer)
	weights := []float32{0.1, 0.2, 0.3, 0.4}
	bias := []float32{0.0}

	weightTensor, err := tensor.NewTensor([]int{2, 2}, weights)
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}
	defer weightTensor.ReleaseGPU()

	biasTensor, err := tensor.NewTensor([]int{1}, bias)
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}
	defer biasTensor.ReleaseGPU()

	params := []*tensor.Tensor{weightTensor, biasTensor}

	// Create mock gradients
	weightGrads := []float32{0.01, 0.02, 0.03, 0.04}
	biasGrads := []float32{0.01}

	weightGradTensor, err := tensor.NewTensor([]int{2, 2}, weightGrads)
	if err != nil {
		t.Fatalf("Failed to create weight gradient tensor: %v", err)
	}
	defer weightGradTensor.ReleaseGPU()

	biasGradTensor, err := tensor.NewTensor([]int{1}, biasGrads)
	if err != nil {
		t.Fatalf("Failed to create bias gradient tensor: %v", err)
	}
	defer biasGradTensor.ReleaseGPU()

	grads := []*tensor.Tensor{weightGradTensor, biasGradTensor}

	// Create SGD optimizer
	sgdConfig := optimizer.SGDConfig{
		OptimizerConfig: optimizer.OptimizerConfig{
			LearningRate: 0.01,
			WeightDecay:  0.0001,
		},
		Momentum: 0.9,
	}
	
	sgdOpt := optimizer.NewSGD(sgdConfig)
	defer sgdOpt.ReleaseGPU()

	fmt.Printf("Initial weights: %v\n", weights)
	fmt.Printf("Initial bias: %v\n", bias)

	// Perform optimization steps
	for step := 0; step < 5; step++ {
		err := sgdOpt.Step(params, grads)
		if err != nil {
			t.Fatalf("SGD step failed: %v", err)
		}

		// Retrieve updated parameters
		err = weightTensor.RetrieveCPU()
		if err != nil {
			t.Fatalf("Failed to retrieve weights: %v", err)
		}
		err = biasTensor.RetrieveCPU()
		if err != nil {
			t.Fatalf("Failed to retrieve bias: %v", err)
		}

		fmt.Printf("Step %d - Weights: %v, Bias: %v, LR: %f\n", 
			step+1, weightTensor.Data, biasTensor.Data, sgdOpt.GetLearningRate())
	}
}

// TestAdamOptimizer demonstrates Adam optimizer usage
func TestAdamOptimizer(t *testing.T) {
	// Create test parameters
	weights := []float32{0.1, 0.2, 0.3, 0.4}
	weightTensor, err := tensor.NewTensor([]int{2, 2}, weights)
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}
	defer weightTensor.ReleaseGPU()

	params := []*tensor.Tensor{weightTensor}

	// Create gradients with some noise
	rand.Seed(42)
	
	// Create Adam optimizer
	adamConfig := optimizer.AdamConfig{
		OptimizerConfig: optimizer.OptimizerConfig{
			LearningRate: 0.001,
			WeightDecay:  0.01,
		},
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
	
	adamOpt := optimizer.NewAdam(adamConfig)
	defer adamOpt.ReleaseGPU()

	fmt.Printf("Initial weights: %v\n", weights)

	// Simulate training with varying gradients
	for step := range 10 {
		// Create noisy gradients to simulate real training
		noisyGrads := make([]float32, 4)
		for i := range noisyGrads {
			noisyGrads[i] = 0.01 + rand.Float32()*0.02 - 0.01 // Random between 0 and 0.02
		}

		gradTensor, err := tensor.NewTensor([]int{2, 2}, noisyGrads)
		if err != nil {
			t.Fatalf("Failed to create gradient tensor: %v", err)
		}
		
		grads := []*tensor.Tensor{gradTensor}

		err = adamOpt.Step(params, grads)
		if err != nil {
			t.Fatalf("Adam step failed: %v", err)
		}

		// Retrieve updated parameters
		err = weightTensor.RetrieveCPU()
		if err != nil {
			t.Fatalf("Failed to retrieve weights: %v", err)
		}

		fmt.Printf("Step %d - Weights: %v, LR: %f\n", 
			step+1, weightTensor.Data, adamOpt.GetLearningRate())

		gradTensor.ReleaseGPU()
	}
}

// TestLearningRateSchedulers demonstrates various LR scheduling strategies
func TestLearningRateSchedulers(t *testing.T) {
	// Create a simple parameter
	weights := []float32{0.1, 0.2}
	weightTensor, err := tensor.NewTensor([]int{2}, weights)
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}
	defer weightTensor.ReleaseGPU()

	grads := []float32{0.01, 0.02}
	gradTensor, err := tensor.NewTensor([]int{2}, grads)
	if err != nil {
		t.Fatalf("Failed to create gradient tensor: %v", err)
	}
	defer gradTensor.ReleaseGPU()

	params := []*tensor.Tensor{weightTensor}
	gradients := []*tensor.Tensor{gradTensor}

	// Test different schedulers
	testCases := []struct {
		name      string
		scheduler optimizer.LRScheduler
	}{
		{
			name: "Exponential Decay",
			scheduler: optimizer.NewExponentialDecayScheduler(0.1, 0.9, 10),
		},
		{
			name: "Step Decay",
			scheduler: optimizer.NewStepDecayScheduler(0.1, 0.5, 5),
		},
		{
			name: "Cosine Annealing",
			scheduler: optimizer.NewCosineAnnealingScheduler(0.1, 0.001, 20),
		},
		{
			name: "Polynomial Decay",
			scheduler: optimizer.NewPolynomialDecayScheduler(0.1, 0.001, 20, 2.0),
		},
		{
			name: "One Cycle",
			scheduler: optimizer.NewOneCycleLRScheduler(0.1, 20, 0.3, "cos"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create optimizer
			adamConfig := optimizer.AdamConfig{
				OptimizerConfig: optimizer.OptimizerConfig{
					LearningRate: 0.1,
					WeightDecay:  0.0,
				},
				Beta1:   0.9,
				Beta2:   0.999,
				Epsilon: 1e-8,
			}
			
			adamOpt := optimizer.NewAdam(adamConfig)
			defer adamOpt.ReleaseGPU()

			// Set scheduler
			tc.scheduler.SetOptimizer(adamOpt)

			fmt.Printf("\n=== %s ===\n", tc.name)
			
			for step := int64(0); step < 20; step++ {
				err := tc.scheduler.Step(step)
				if err != nil {
					t.Fatalf("Scheduler step failed: %v", err)
				}

				err = adamOpt.Step(params, gradients)
				if err != nil {
					t.Fatalf("Optimizer step failed: %v", err)
				}

				if step%5 == 0 {
					fmt.Printf("Step %d: LR = %.6f\n", step, tc.scheduler.GetLR())
				}
			}
		})
	}
}

// TestWarmupScheduler demonstrates warmup scheduling
func TestWarmupScheduler(t *testing.T) {
	// Create warmup + cosine annealing scheduler
	warmupScheduler := optimizer.NewWarmupCosineScheduler(0.0, 0.1, 0.001, 5, 20)

	// Create optimizer
	sgdConfig := optimizer.SGDConfig{
		OptimizerConfig: optimizer.OptimizerConfig{
			LearningRate: 0.0,
			WeightDecay:  0.0,
		},
		Momentum: 0.0,
	}
	
	sgdOpt := optimizer.NewSGD(sgdConfig)
	defer sgdOpt.ReleaseGPU()

	warmupScheduler.SetOptimizer(sgdOpt)

	fmt.Printf("\n=== Warmup + Cosine Annealing ===\n")
	
	for step := int64(0); step < 25; step++ {
		err := warmupScheduler.Step(step)
		if err != nil {
			t.Fatalf("Warmup scheduler step failed: %v", err)
		}

		fmt.Printf("Step %d: LR = %.6f\n", step, warmupScheduler.GetLR())
	}
}

// TestGradientClipping demonstrates gradient clipping functionality
func TestGradientClipping(t *testing.T) {
	// Create large gradients that need clipping
	largeGrads1 := []float32{10.0, 20.0, 30.0, 40.0}
	largeGrads2 := []float32{50.0, 60.0}

	gradTensor1, err := tensor.NewTensor([]int{2, 2}, largeGrads1)
	if err != nil {
		t.Fatalf("Failed to create gradient tensor 1: %v", err)
	}
	defer gradTensor1.ReleaseGPU()

	gradTensor2, err := tensor.NewTensor([]int{2}, largeGrads2)
	if err != nil {
		t.Fatalf("Failed to create gradient tensor 2: %v", err)
	}
	defer gradTensor2.ReleaseGPU()

	grads := []*tensor.Tensor{gradTensor1, gradTensor2}

	// Compute gradient norm before clipping
	normBefore, err := optimizer.ComputeGradNorm(grads)
	if err != nil {
		t.Fatalf("Failed to compute gradient norm: %v", err)
	}
	fmt.Printf("Gradient norm before clipping: %.4f\n", normBefore)

	// Clip gradients by norm
	maxNorm := float32(5.0)
	actualNorm, err := optimizer.ClipGradsByNorm(grads, maxNorm)
	if err != nil {
		t.Fatalf("Failed to clip gradients by norm: %v", err)
	}
	fmt.Printf("Actual norm before clipping: %.4f\n", actualNorm)

	// Compute gradient norm after clipping
	normAfter, err := optimizer.ComputeGradNorm(grads)
	if err != nil {
		t.Fatalf("Failed to compute gradient norm after clipping: %v", err)
	}
	fmt.Printf("Gradient norm after clipping: %.4f\n", normAfter)

	// Retrieve and print clipped gradients
	err = gradTensor1.RetrieveCPU()
	if err != nil {
		t.Fatalf("Failed to retrieve grad tensor 1: %v", err)
	}
	err = gradTensor2.RetrieveCPU()
	if err != nil {
		t.Fatalf("Failed to retrieve grad tensor 2: %v", err)
	}

	fmt.Printf("Clipped gradients 1: %v\n", gradTensor1.Data)
	fmt.Printf("Clipped gradients 2: %v\n", gradTensor2.Data)
}

// TestOptimizerIntegration demonstrates a complete training-like scenario
func TestOptimizerIntegration(t *testing.T) {
	fmt.Printf("\n=== Complete Training Simulation ===\n")

	// Create a simple 2x2 matrix multiplication as our "model"
	// W * X = Y, where we want to learn W
	
	// Target weights (what we want to learn)
	targetWeights := []float32{2.0, 3.0, 1.0, 4.0}
	
	// Initialize random weights
	rand.Seed(42)
	initWeights := make([]float32, 4)
	for i := range initWeights {
		initWeights[i] = rand.Float32() - 0.5 // Random between -0.5 and 0.5
	}

	weightTensor, err := tensor.NewTensor([]int{2, 2}, initWeights)
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}
	defer weightTensor.ReleaseGPU()

	// Input data
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	inputTensor, err := tensor.NewTensor([]int{2, 2}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.ReleaseGPU()

	// Target output (computed with target weights)
	targetOutput := []float32{8.0, 18.0, 13.0, 30.0} // targetWeights * inputData
	targetTensor, err := tensor.NewTensor([]int{2, 2}, targetOutput)
	if err != nil {
		t.Fatalf("Failed to create target tensor: %v", err)
	}
	defer targetTensor.ReleaseGPU()

	// Create AdamW optimizer with warmup + cosine annealing
	adamConfig := optimizer.AdamWConfig{
		OptimizerConfig: optimizer.OptimizerConfig{
			LearningRate: 0.0, // Will be set by scheduler
			WeightDecay:  0.01,
		},
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
	
	adamOpt := optimizer.NewAdamW(adamConfig)
	defer adamOpt.ReleaseGPU()

	// Create scheduler
	scheduler := optimizer.NewWarmupCosineScheduler(0.0, 0.1, 0.001, 10, 100)
	scheduler.SetOptimizer(adamOpt)

	params := []*tensor.Tensor{weightTensor}

	fmt.Printf("Initial weights: %v\n", initWeights)
	fmt.Printf("Target weights: %v\n", targetWeights)

	// Training loop
	for epoch := int64(0); epoch < 50; epoch++ {
		// Update learning rate
		err := scheduler.Step(epoch)
		if err != nil {
			t.Fatalf("Scheduler step failed: %v", err)
		}

		// Forward pass: compute output = weights * input
		output, err := matrix.MatMul(weightTensor, inputTensor)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		defer output.ReleaseGPU()

		// Compute loss: MSE = mean((output - target)^2)
		diff, err := matrix.Sub(output, targetTensor)
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		defer diff.ReleaseGPU()

		squared, err := matrix.Mul(diff, diff)
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		defer squared.ReleaseGPU()

		// Compute gradients (simplified - in real training you'd use autograd)
		// For MSE loss: dL/dW = 2 * (output - target) * input^T
		inputT, err := matrix.Transpose(inputTensor)
		if err != nil {
			t.Fatalf("Input transpose failed: %v", err)
		}
		defer inputT.ReleaseGPU()

		diffScaled, err := matrix.ScalarMul(diff, 2.0/4.0) // 2.0 from derivative, 1/4 for mean
		if err != nil {
			t.Fatalf("Gradient scaling failed: %v", err)
		}
		defer diffScaled.ReleaseGPU()

		gradients, err := matrix.MatMul(diffScaled, inputT)
		if err != nil {
			t.Fatalf("Gradient computation failed: %v", err)
		}
		defer gradients.ReleaseGPU()

		// Clip gradients
		grads := []*tensor.Tensor{gradients}
		_, err = optimizer.ClipGradsByNorm(grads, 1.0)
		if err != nil {
			t.Fatalf("Gradient clipping failed: %v", err)
		}

		// Optimizer step
		err = adamOpt.Step(params, grads)
		if err != nil {
			t.Fatalf("Optimizer step failed: %v", err)
		}

		// Print progress
		if epoch%10 == 0 {
			err = weightTensor.RetrieveCPU()
			if err != nil {
				t.Fatalf("Failed to retrieve weights: %v", err)
			}
			
			err = squared.RetrieveCPU()
			if err != nil {
				t.Fatalf("Failed to retrieve loss: %v", err)
			}
			
			// Compute mean loss
			var meanLoss float32
			for _, val := range squared.Data {
				meanLoss += val
			}
			meanLoss /= float32(len(squared.Data))

			fmt.Printf("Epoch %d: Loss = %.6f, LR = %.6f, Weights = %v\n", 
				epoch, meanLoss, scheduler.GetLR(), weightTensor.Data)
		}
	}

	// Final results
	err = weightTensor.RetrieveCPU()
	if err != nil {
		t.Fatalf("Failed to retrieve final weights: %v", err)
	}

	fmt.Printf("\nFinal weights: %v\n", weightTensor.Data)
	fmt.Printf("Target weights: %v\n", targetWeights)
	
	// Check if we're close to the target
	for i, learned := range weightTensor.Data {
		target := targetWeights[i]
		diff := learned - target
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.1 { // Allow 10% error
			t.Logf("Warning: Weight %d learned %.4f, target %.4f (diff %.4f)", i, learned, target, diff)
		}
	}
}

// BenchmarkOptimizers compares performance of different optimizers
func BenchmarkOptimizers(b *testing.B) {
	// Setup common data
	weights := make([]float32, 1000)
	grads := make([]float32, 1000)
	for i := range weights {
		weights[i] = rand.Float32()
		grads[i] = rand.Float32() * 0.01
	}

	weightTensor, _ := tensor.NewTensor([]int{1000}, weights)
	defer weightTensor.ReleaseGPU()
	gradTensor, _ := tensor.NewTensor([]int{1000}, grads)
	defer gradTensor.ReleaseGPU()

	params := []*tensor.Tensor{weightTensor}
	gradients := []*tensor.Tensor{gradTensor}

	b.Run("SGD", func(b *testing.B) {
		sgdOpt := optimizer.NewSGD(optimizer.SGDConfig{
			OptimizerConfig: optimizer.OptimizerConfig{LearningRate: 0.01},
			Momentum: 0.9,
		})
		defer sgdOpt.ReleaseGPU()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sgdOpt.Step(params, gradients)
		}
	})

	b.Run("Adam", func(b *testing.B) {
		adamOpt := optimizer.NewAdam(optimizer.AdamConfig{
			OptimizerConfig: optimizer.OptimizerConfig{LearningRate: 0.001},
			Beta1: 0.9,
			Beta2: 0.999,
			Epsilon: 1e-8,
		})
		defer adamOpt.ReleaseGPU()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			adamOpt.Step(params, gradients)
		}
	})

	b.Run("AdamW", func(b *testing.B) {
		adamwOpt := optimizer.NewAdamW(optimizer.AdamWConfig{
			OptimizerConfig: optimizer.OptimizerConfig{LearningRate: 0.001, WeightDecay: 0.01},
			Beta1: 0.9,
			Beta2: 0.999,
			Epsilon: 1e-8,
		})
		defer adamwOpt.ReleaseGPU()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			adamwOpt.Step(params, gradients)
		}
	})
}