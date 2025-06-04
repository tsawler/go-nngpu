package matrix

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tsawler/go-nngpu/tensor"
)

// LossScaleStressTest provides comprehensive testing of loss scaling mechanisms
type LossScaleStressTest struct {
	trainer *MixedPrecisionTrainer
	config  *MixedPrecisionConfig
	results []StressTestResult
}

// StressTestResult records the results of a stress test scenario
type StressTestResult struct {
	Scenario          string
	InitialScale      float32
	FinalScale        float32
	OverflowsDetected int
	StepsSkipped      int
	ScaleUpdates      int
	MaxScale          float32
	MinScale          float32
	StabilityScore    float64 // 0-1, higher is more stable
}

// NewLossScaleStressTest creates a new stress test instance
func NewLossScaleStressTest() *LossScaleStressTest {
	return &LossScaleStressTest{
		results: make([]StressTestResult, 0),
	}
}

// TestOverflowScenarios tests various overflow scenarios to verify adaptive scaling
func (test *LossScaleStressTest) TestOverflowScenarios() ([]StressTestResult, error) {
	scenarios := []struct {
		name          string
		config        *MixedPrecisionConfig
		gradientSizes []float32
	}{
		{
			name: "High Initial Scale - Force Overflow",
			config: &MixedPrecisionConfig{
				Enabled:             true,
				LossScale:           1048576.0, // Very high initial scale
				LossScaleGrowthRate: 2.0,
				LossScaleBackoffRate: 0.5,
				GrowthInterval:      5,
				MaxLossScale:        1048576.0,
				MinLossScale:        1.0,
				SkipOverflowSteps:   true,
			},
			gradientSizes: []float32{100.0, 1000.0, 10000.0, 1.0, 0.1}, // Large gradients first
		},
		{
			name: "Low Initial Scale - Test Growth",
			config: &MixedPrecisionConfig{
				Enabled:             true,
				LossScale:           2.0, // Very low initial scale
				LossScaleGrowthRate: 2.0,
				LossScaleBackoffRate: 0.5,
				GrowthInterval:      3,
				MaxLossScale:        65536.0,
				MinLossScale:        1.0,
				SkipOverflowSteps:   true,
			},
			gradientSizes: []float32{0.001, 0.002, 0.001, 0.003, 0.001}, // Small gradients
		},
		{
			name: "Aggressive Backoff - Rapid Scale Reduction",
			config: &MixedPrecisionConfig{
				Enabled:             true,
				LossScale:           32768.0,
				LossScaleGrowthRate: 1.5,
				LossScaleBackoffRate: 0.1, // Very aggressive backoff
				GrowthInterval:      10,
				MaxLossScale:        65536.0,
				MinLossScale:        0.1,
				SkipOverflowSteps:   true,
			},
			gradientSizes: []float32{50000.0, 100000.0, 1.0, 0.5}, // Huge gradients then normal
		},
		{
			name: "Conservative Growth - Slow Scale Increase",
			config: &MixedPrecisionConfig{
				Enabled:             true,
				LossScale:           8.0,
				LossScaleGrowthRate: 1.1, // Very conservative growth
				LossScaleBackoffRate: 0.8,
				GrowthInterval:      2,
				MaxLossScale:        1024.0,
				MinLossScale:        1.0,
				SkipOverflowSteps:   false, // Don't skip steps
			},
			gradientSizes: []float32{0.01, 0.01, 0.01, 0.01, 0.01}, // Consistent small gradients
		},
	}

	for _, scenario := range scenarios {
		result, err := test.runStressTestScenario(scenario.name, scenario.config, scenario.gradientSizes)
		if err != nil {
			return nil, fmt.Errorf("scenario '%s' failed: %w", scenario.name, err)
		}
		test.results = append(test.results, result)
	}

	return test.results, nil
}

// runStressTestScenario executes a single stress test scenario
func (test *LossScaleStressTest) runStressTestScenario(name string, config *MixedPrecisionConfig, gradientSizes []float32) (StressTestResult, error) {
	trainer, err := NewMixedPrecisionTrainer(config)
	if err != nil {
		return StressTestResult{}, fmt.Errorf("failed to create trainer: %w", err)
	}
	defer trainer.Cleanup()

	result := StressTestResult{
		Scenario:     name,
		InitialScale: trainer.GetCurrentLossScale(),
		MaxScale:     trainer.GetCurrentLossScale(),
		MinScale:     trainer.GetCurrentLossScale(),
	}

	// Simulate training steps with different gradient magnitudes
	for step, gradSize := range gradientSizes {
		// Create gradients with specified magnitude
		gradData := make([]float32, 100) // Small tensor for speed
		for i := range gradData {
			gradData[i] = gradSize * (rand.Float32()*2 - 1) // Random sign
		}

		gradTensor, err := tensor.NewTensor([]int{10, 10}, gradData)
		if err != nil {
			return result, fmt.Errorf("failed to create gradient tensor: %w", err)
		}

		// Scale gradients
		scaledGrads, err := trainer.ScaleGradients(gradTensor)
		if err != nil {
			return result, fmt.Errorf("failed to scale gradients: %w", err)
		}

		// Unscale gradients (this detects overflow)
		_, err = trainer.UnscaleGradients(scaledGrads)
		if err != nil {
			return result, fmt.Errorf("failed to unscale gradients: %w", err)
		}

		// Check for overflow
		if trainer.GetOverflowStatus() {
			result.OverflowsDetected++
		}

		// Check if step should be skipped
		if trainer.ShouldSkipStep() {
			result.StepsSkipped++
		}

		// Update loss scale
		previousScale := trainer.GetCurrentLossScale()
		trainer.UpdateLossScale()
		currentScale := trainer.GetCurrentLossScale()

		// Track scale changes
		if currentScale != previousScale {
			result.ScaleUpdates++
		}

		// Track min/max scales
		if currentScale > result.MaxScale {
			result.MaxScale = currentScale
		}
		if currentScale < result.MinScale {
			result.MinScale = currentScale
		}

		fmt.Printf("    Step %d: GradSize=%.3f, Scale=%.0f, Overflow=%t, Skip=%t\n",
			step+1, gradSize, currentScale, trainer.GetOverflowStatus(), trainer.ShouldSkipStep())
	}

	result.FinalScale = trainer.GetCurrentLossScale()

	// Calculate stability score (fewer overflows and scale changes = more stable)
	totalSteps := float64(len(gradientSizes))
	overflowRate := float64(result.OverflowsDetected) / totalSteps
	updateRate := float64(result.ScaleUpdates) / totalSteps
	result.StabilityScore = 1.0 - (overflowRate*0.5 + updateRate*0.3) // Weighted score

	return result, nil
}

// GenerateExtremePrecisionTestCases creates test cases for extreme precision scenarios
func (test *LossScaleStressTest) GenerateExtremePrecisionTestCases() ([]*tensor.Tensor, []string, error) {
	testCases := []struct {
		name   string
		values []float32
	}{
		{
			"Subnormal Numbers",
			[]float32{1e-38, 5e-39, 1e-40, 1e-45}, // Very small numbers
		},
		{
			"Near Overflow",
			[]float32{65504.0, 65000.0, 32768.0, 16384.0}, // Near float16 max
		},
		{
			"Mixed Magnitudes",
			[]float32{1e-6, 1e3, 1e-4, 1e6, 1e-8}, // Wide range
		},
		{
			"Precision Boundary",
			[]float32{0.00006103515625, 0.00006104, 0.000061, 0.0000610352}, // Around float16 precision limit
		},
		{
			"Zero and Near-Zero",
			[]float32{0.0, 1e-10, -1e-10, 1e-15}, // Testing zero handling
		},
	}

	tensors := make([]*tensor.Tensor, len(testCases))
	names := make([]string, len(testCases))

	for i, testCase := range testCases {
		tensor, err := tensor.NewTensor([]int{len(testCase.values)}, testCase.values)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create tensor for %s: %w", testCase.name, err)
		}
		tensors[i] = tensor
		names[i] = testCase.name
	}

	return tensors, names, nil
}

// TestPrecisionLimits tests the precision limits of float16 conversion
func (test *LossScaleStressTest) TestPrecisionLimits() (map[string]PrecisionTestResult, error) {
	tensors, names, err := test.GenerateExtremePrecisionTestCases()
	if err != nil {
		return nil, err
	}

	results := make(map[string]PrecisionTestResult)
	trainer, err := NewMixedPrecisionTrainer(DefaultMixedPrecisionConfig())
	if err != nil {
		return nil, err
	}
	defer trainer.Cleanup()

	for i, tensor := range tensors {
		name := names[i]
		
		// Convert to float16 and back
		f16Tensor, err := trainer.ConvertTensorToFloat16(tensor)
		if err != nil {
			return nil, fmt.Errorf("conversion failed for %s: %w", name, err)
		}

		if err := tensor.RetrieveCPU(); err != nil {
			return nil, err
		}
		if err := f16Tensor.RetrieveCPU(); err != nil {
			return nil, err
		}

		// Analyze precision loss
		result := PrecisionTestResult{
			TestCase:     name,
			OriginalValues: make([]float32, len(tensor.Data)),
			ConvertedValues: make([]float32, len(f16Tensor.Data)),
			AbsoluteErrors: make([]float64, len(tensor.Data)),
			RelativeErrors: make([]float64, len(tensor.Data)),
		}

		copy(result.OriginalValues, tensor.Data)
		copy(result.ConvertedValues, f16Tensor.Data)

		var maxAbsError, maxRelError, avgAbsError, avgRelError float64
		validCount := 0

		for j := range tensor.Data {
			original := float64(tensor.Data[j])
			converted := float64(f16Tensor.Data[j])
			
			absError := math.Abs(original - converted)
			result.AbsoluteErrors[j] = absError
			
			var relError float64
			if math.Abs(original) > 1e-15 {
				relError = absError / math.Abs(original)
				result.RelativeErrors[j] = relError
				validCount++
				avgRelError += relError
			}

			avgAbsError += absError
			
			if absError > maxAbsError {
				maxAbsError = absError
			}
			if relError > maxRelError {
				maxRelError = relError
			}
		}

		result.MaxAbsoluteError = maxAbsError
		result.MaxRelativeError = maxRelError
		result.AvgAbsoluteError = avgAbsError / float64(len(tensor.Data))
		if validCount > 0 {
			result.AvgRelativeError = avgRelError / float64(validCount)
		}

		results[name] = result
	}

	return results, nil
}

// PrecisionTestResult holds the results of precision testing
type PrecisionTestResult struct {
	TestCase        string
	OriginalValues  []float32
	ConvertedValues []float32
	AbsoluteErrors  []float64
	RelativeErrors  []float64
	MaxAbsoluteError float64
	MaxRelativeError float64
	AvgAbsoluteError float64
	AvgRelativeError float64
}

// RecommendOptimalScale analyzes test results and recommends optimal scaling strategy
func (test *LossScaleStressTest) RecommendOptimalScale(results []StressTestResult) *MixedPrecisionConfig {
	// Analyze results to find the best performing configuration
	bestStability := 0.0
	var bestConfig *MixedPrecisionConfig

	for _, result := range results {
		if result.StabilityScore > bestStability {
			bestStability = result.StabilityScore
			// Create config based on the best performing scenario
			bestConfig = &MixedPrecisionConfig{
				Enabled:             true,
				LossScale:           result.InitialScale,
				LossScaleGrowthRate: 1.5, // Conservative growth
				LossScaleBackoffRate: 0.5, // Moderate backoff
				GrowthInterval:      20,   // Longer intervals for stability
				MaxLossScale:        result.MaxScale,
				MinLossScale:        float32(math.Max(1.0, float64(result.MinScale))),
				SkipOverflowSteps:   true,
			}
		}
	}

	// Apply learned optimizations
	if bestConfig != nil {
		// Adjust based on overall performance patterns
		if bestStability < 0.7 {
			// If stability was low, be more conservative
			bestConfig.LossScaleGrowthRate = 1.2
			bestConfig.GrowthInterval = 50
		}
	}

	return bestConfig
}