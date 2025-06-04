package matrix

import (
	"fmt"

	"github.com/tsawler/go-nngpu/tensor"
	"gonum.org/v1/gonum/mat"
)

// GPUBatchNormLayer represents a batch normalization layer with GPU acceleration and Gonum compatibility
type GPUBatchNormLayer struct {
	Features    int
	Epsilon     float32
	Momentum    float32          // For running statistics update
	Training    bool             // Training vs inference mode
	Gamma       *GPUDense        // Scale parameters
	Beta        *GPUDense        // Shift parameters
	RunningMean *GPUDense        // Running mean for inference
	RunningVar  *GPUDense        // Running variance for inference
	LastInput   *tensor.Tensor   // Stored for backward pass
	LastResult  *BatchNormResult // Stored for backward pass
}

// NewGPUBatchNormLayer creates a new GPU-accelerated batch normalization layer
func NewGPUBatchNormLayer(features int, epsilon, momentum float32) *GPUBatchNormLayer {
	// Initialize gamma to ones (scale)
	gammaData := make([]float64, features)
	for i := range gammaData {
		gammaData[i] = 1.0
	}
	gamma := NewGPUDense(1, features, gammaData)

	// Initialize beta to zeros (shift)
	betaData := make([]float64, features)
	beta := NewGPUDense(1, features, betaData)

	// Initialize running statistics
	runningMeanData := make([]float64, features)
	runningVarData := make([]float64, features)
	for i := range runningVarData {
		runningVarData[i] = 1.0 // Initialize variance to 1
	}
	runningMean := NewGPUDense(1, features, runningMeanData)
	runningVar := NewGPUDense(1, features, runningVarData)

	return &GPUBatchNormLayer{
		Features:    features,
		Epsilon:     epsilon,
		Momentum:    momentum,
		Training:    true,
		Gamma:       gamma,
		Beta:        beta,
		RunningMean: runningMean,
		RunningVar:  runningVar,
	}
}

// SetTraining sets the layer to training or inference mode
func (layer *GPUBatchNormLayer) SetTraining(training bool) {
	layer.Training = training
}

// Forward performs the forward pass of batch normalization
func (layer *GPUBatchNormLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("batch norm layer expects 2D input (batch_size, features), got %dD", len(input.Shape))
	}

	if input.Shape[1] != layer.Features {
		return nil, fmt.Errorf("input features (%d) don't match layer features (%d)", input.Shape[1], layer.Features)
	}

	// Store input for backward pass
	layer.LastInput = input

	// Convert parameters to tensors
	gammaTensor, err := layer.gammaToTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to convert gamma to tensor: %w", err)
	}
	defer gammaTensor.ReleaseGPU()

	betaTensor, err := layer.betaToTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to convert beta to tensor: %w", err)
	}
	defer betaTensor.ReleaseGPU()

	if layer.Training {
		// Training mode: compute batch statistics
		result, err := BatchNormTraining(input, gammaTensor, betaTensor, layer.Epsilon)
		if err != nil {
			return nil, fmt.Errorf("batch norm training forward failed: %w", err)
		}

		// Update running statistics
		runningMeanTensor, err := layer.runningMeanToTensor()
		if err != nil {
			result.ReleaseGPU()
			return nil, fmt.Errorf("failed to convert running mean to tensor: %w", err)
		}
		defer runningMeanTensor.ReleaseGPU()

		runningVarTensor, err := layer.runningVarToTensor()
		if err != nil {
			result.ReleaseGPU()
			return nil, fmt.Errorf("failed to convert running var to tensor: %w", err)
		}
		defer runningVarTensor.ReleaseGPU()

		err = UpdateRunningStats(runningMeanTensor, runningVarTensor, result.Mean, result.Variance, layer.Momentum)
		if err != nil {
			result.ReleaseGPU()
			return nil, fmt.Errorf("failed to update running statistics: %w", err)
		}

		// Update the layer's running statistics
		layer.updateRunningMeanFromTensor(runningMeanTensor)
		layer.updateRunningVarFromTensor(runningVarTensor)

		layer.LastResult = result
		return result.Output, nil

	} else {
		// Inference mode: use running statistics
		runningMeanTensor, err := layer.runningMeanToTensor()
		if err != nil {
			return nil, fmt.Errorf("failed to convert running mean to tensor: %w", err)
		}
		defer runningMeanTensor.ReleaseGPU()

		runningVarTensor, err := layer.runningVarToTensor()
		if err != nil {
			return nil, fmt.Errorf("failed to convert running var to tensor: %w", err)
		}
		defer runningVarTensor.ReleaseGPU()

		output, err := BatchNormInference(input, runningMeanTensor, runningVarTensor, gammaTensor, betaTensor, layer.Epsilon)
		if err != nil {
			return nil, fmt.Errorf("batch norm inference forward failed: %w", err)
		}

		return output, nil
	}
}

// Backward performs the backward pass of batch normalization
func (layer *GPUBatchNormLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error) {
	if !layer.Training {
		return nil, fmt.Errorf("backward pass is only available in training mode")
	}

	if layer.LastInput == nil || layer.LastResult == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	// Convert gamma to tensor
	gammaTensor, err := layer.gammaToTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to convert gamma to tensor: %w", err)
	}
	defer gammaTensor.ReleaseGPU()

	// Compute gradients
	gradients, err := BatchNormBackward(gradOutput, layer.LastInput, layer.LastResult.Mean, layer.LastResult.Variance, gammaTensor, layer.Epsilon)
	if err != nil {
		return nil, fmt.Errorf("batch norm backward failed: %w", err)
	}
	defer gradients.ReleaseGPU()

	// Update gamma and beta gradients (in a real implementation, you'd pass these to an optimizer)
	// TODO: For now, we'll apply a simple gradient descent update
	learningRate := float32(0.001)

	if err := gradients.GradGamma.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve gamma gradients: %w", err)
	}
	if err := gradients.GradBeta.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve beta gradients: %w", err)
	}

	// Update gamma parameters
	for i := 0; i < layer.Features; i++ {
		currentGamma := layer.Gamma.At(0, i)
		layer.Gamma.Set(0, i, currentGamma-float64(learningRate*gradients.GradGamma.Data[i]))
	}

	// Update beta parameters
	for i := 0; i < layer.Features; i++ {
		currentBeta := layer.Beta.At(0, i)
		layer.Beta.Set(0, i, currentBeta-float64(learningRate*gradients.GradBeta.Data[i]))
	}

	return gradients.GradInput, nil
}

// Helper methods for tensor conversion
func (layer *GPUBatchNormLayer) gammaToTensor() (*tensor.Tensor, error) {
	if err := layer.Gamma.tensor.RetrieveCPU(); err != nil {
		return nil, err
	}
	return tensor.NewTensor([]int{layer.Features}, layer.Gamma.tensor.Data)
}

func (layer *GPUBatchNormLayer) betaToTensor() (*tensor.Tensor, error) {
	if err := layer.Beta.tensor.RetrieveCPU(); err != nil {
		return nil, err
	}
	return tensor.NewTensor([]int{layer.Features}, layer.Beta.tensor.Data)
}

func (layer *GPUBatchNormLayer) runningMeanToTensor() (*tensor.Tensor, error) {
	if err := layer.RunningMean.tensor.RetrieveCPU(); err != nil {
		return nil, err
	}
	return tensor.NewTensor([]int{layer.Features}, layer.RunningMean.tensor.Data)
}

func (layer *GPUBatchNormLayer) runningVarToTensor() (*tensor.Tensor, error) {
	if err := layer.RunningVar.tensor.RetrieveCPU(); err != nil {
		return nil, err
	}
	return tensor.NewTensor([]int{layer.Features}, layer.RunningVar.tensor.Data)
}

func (layer *GPUBatchNormLayer) updateRunningMeanFromTensor(t *tensor.Tensor) error {
	if err := t.RetrieveCPU(); err != nil {
		return err
	}
	for i := 0; i < layer.Features; i++ {
		layer.RunningMean.Set(0, i, float64(t.Data[i]))
	}
	return nil
}

func (layer *GPUBatchNormLayer) updateRunningVarFromTensor(t *tensor.Tensor) error {
	if err := t.RetrieveCPU(); err != nil {
		return err
	}
	for i := 0; i < layer.Features; i++ {
		layer.RunningVar.Set(0, i, float64(t.Data[i]))
	}
	return nil
}

// ReleaseGPU releases GPU resources
func (layer *GPUBatchNormLayer) ReleaseGPU() {
	if layer.Gamma != nil {
		layer.Gamma.ReleaseGPU()
	}
	if layer.Beta != nil {
		layer.Beta.ReleaseGPU()
	}
	if layer.RunningMean != nil {
		layer.RunningMean.ReleaseGPU()
	}
	if layer.RunningVar != nil {
		layer.RunningVar.ReleaseGPU()
	}
	if layer.LastResult != nil {
		layer.LastResult.ReleaseGPU()
		layer.LastResult = nil
	}
	layer.LastInput = nil
}

// GPULayerNormLayer represents a layer normalization layer with GPU acceleration
type GPULayerNormLayer struct {
	Features   int
	Epsilon    float32
	Gamma      *GPUDense        // Scale parameters
	Beta       *GPUDense        // Shift parameters
	LastInput  *tensor.Tensor   // Stored for backward pass
	LastResult *BatchNormResult // Stored for backward pass
}

// NewGPULayerNormLayer creates a new GPU-accelerated layer normalization layer
func NewGPULayerNormLayer(features int, epsilon float32) *GPULayerNormLayer {
	// Initialize gamma to ones
	gammaData := make([]float64, features)
	for i := range gammaData {
		gammaData[i] = 1.0
	}
	gamma := NewGPUDense(1, features, gammaData)

	// Initialize beta to zeros
	betaData := make([]float64, features)
	beta := NewGPUDense(1, features, betaData)

	return &GPULayerNormLayer{
		Features: features,
		Epsilon:  epsilon,
		Gamma:    gamma,
		Beta:     beta,
	}
}

// Forward performs the forward pass of layer normalization
func (layer *GPULayerNormLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("layer norm expects 2D input (batch_size, features), got %dD", len(input.Shape))
	}

	if input.Shape[1] != layer.Features {
		return nil, fmt.Errorf("input features (%d) don't match layer features (%d)", input.Shape[1], layer.Features)
	}

	// Store input for backward pass
	layer.LastInput = input

	// Convert parameters to tensors
	gammaTensor, err := layer.gammaToTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to convert gamma to tensor: %w", err)
	}
	defer gammaTensor.ReleaseGPU()

	betaTensor, err := layer.betaToTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to convert beta to tensor: %w", err)
	}
	defer betaTensor.ReleaseGPU()

	// Perform layer normalization
	result, err := LayerNormForward(input, gammaTensor, betaTensor, layer.Epsilon)
	if err != nil {
		return nil, fmt.Errorf("layer norm forward failed: %w", err)
	}

	layer.LastResult = result
	return result.Output, nil
}

// Backward performs the backward pass of layer normalization
func (layer *GPULayerNormLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error) {
	if layer.LastInput == nil || layer.LastResult == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	// Convert gamma to tensor
	gammaTensor, err := layer.gammaToTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to convert gamma to tensor: %w", err)
	}
	defer gammaTensor.ReleaseGPU()

	// Compute gradients
	gradients, err := LayerNormBackward(gradOutput, layer.LastInput, layer.LastResult.Mean, layer.LastResult.Variance, gammaTensor, layer.Epsilon)
	if err != nil {
		return nil, fmt.Errorf("layer norm backward failed: %w", err)
	}
	defer gradients.ReleaseGPU()

	// Simple gradient descent update (in practice, use an optimizer)
	learningRate := float32(0.001)

	if err := gradients.GradGamma.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve gamma gradients: %w", err)
	}
	if err := gradients.GradBeta.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve beta gradients: %w", err)
	}

	// Update parameters
	for i := 0; i < layer.Features; i++ {
		currentGamma := layer.Gamma.At(0, i)
		layer.Gamma.Set(0, i, currentGamma-float64(learningRate*gradients.GradGamma.Data[i]))

		currentBeta := layer.Beta.At(0, i)
		layer.Beta.Set(0, i, currentBeta-float64(learningRate*gradients.GradBeta.Data[i]))
	}

	return gradients.GradInput, nil
}

// Helper methods for tensor conversion
func (layer *GPULayerNormLayer) gammaToTensor() (*tensor.Tensor, error) {
	if err := layer.Gamma.tensor.RetrieveCPU(); err != nil {
		return nil, err
	}
	return tensor.NewTensor([]int{layer.Features}, layer.Gamma.tensor.Data)
}

func (layer *GPULayerNormLayer) betaToTensor() (*tensor.Tensor, error) {
	if err := layer.Beta.tensor.RetrieveCPU(); err != nil {
		return nil, err
	}
	return tensor.NewTensor([]int{layer.Features}, layer.Beta.tensor.Data)
}

// ReleaseGPU releases GPU resources
func (layer *GPULayerNormLayer) ReleaseGPU() {
	if layer.Gamma != nil {
		layer.Gamma.ReleaseGPU()
	}
	if layer.Beta != nil {
		layer.Beta.ReleaseGPU()
	}
	if layer.LastResult != nil {
		layer.LastResult.ReleaseGPU()
		layer.LastResult = nil
	}
	layer.LastInput = nil
}

// Drop-in replacement functions for Gonum operations with batch normalization

// GPUBatchNormalize applies batch normalization to a Gonum matrix
func GPUBatchNormalize(input mat.Matrix, gamma, beta, runningMean, runningVar mat.Matrix, epsilon float32, training bool) *mat.Dense {
	// Convert inputs to GPU tensors
	gpuInput := FromGonum(input)
	defer gpuInput.ReleaseGPU()

	gpuGamma := FromGonum(gamma)
	defer gpuGamma.ReleaseGPU()

	gpuBeta := FromGonum(beta)
	defer gpuBeta.ReleaseGPU()

	var result *tensor.Tensor
	var err error

	if training {
		// Training mode
		gammaTensor, err := tensor.NewTensor([]int{gpuGamma.tensor.Shape[1]}, gpuGamma.tensor.Data)
		if err != nil {
			panic(err)
		}
		betaTensor, err := tensor.NewTensor([]int{gpuBeta.tensor.Shape[1]}, gpuBeta.tensor.Data)
		if err != nil {
			panic(err)
		}

		bnResult, err := BatchNormTraining(gpuInput.tensor, gammaTensor, betaTensor, epsilon)
		if err != nil {
			panic(err)
		}
		defer bnResult.ReleaseGPU()

		result = bnResult.Output
	} else {
		// Inference mode
		gpuRunningMean := FromGonum(runningMean)
		defer gpuRunningMean.ReleaseGPU()

		gpuRunningVar := FromGonum(runningVar)
		defer gpuRunningVar.ReleaseGPU()

		result, err = BatchNormInference(gpuInput.tensor, gpuRunningMean.tensor, gpuRunningVar.tensor, gpuGamma.tensor, gpuBeta.tensor, epsilon)
		if err != nil {
			panic(err)
		}
	}
	defer result.ReleaseGPU()

	// Convert back to Gonum format
	if err := result.RetrieveCPU(); err != nil {
		panic(err)
	}

	rows, cols := result.Shape[0], result.Shape[1]
	data := make([]float64, len(result.Data))
	for i, v := range result.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

// GPULayerNormalize applies layer normalization to a Gonum matrix
func GPULayerNormalize(input, gamma, beta mat.Matrix, epsilon float32) *mat.Dense {
	// Convert inputs to GPU tensors
	gpuInput := FromGonum(input)
	defer gpuInput.ReleaseGPU()

	gpuGamma := FromGonum(gamma)
	defer gpuGamma.ReleaseGPU()

	gpuBeta := FromGonum(beta)
	defer gpuBeta.ReleaseGPU()

	result, err := LayerNormForward(gpuInput.tensor, gpuGamma.tensor, gpuBeta.tensor, epsilon)
	if err != nil {
		panic(err)
	}
	defer result.ReleaseGPU()

	// Convert back to Gonum format
	if err := result.Output.RetrieveCPU(); err != nil {
		panic(err)
	}

	rows, cols := result.Output.Shape[0], result.Output.Shape[1]
	data := make([]float64, len(result.Output.Data))
	for i, v := range result.Output.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

// Utility functions for creating normalized parameters

// CreateBatchNormParams creates initialized gamma and beta parameters for batch normalization
func CreateBatchNormParams(features int) (gamma, beta *mat.Dense) {
	// Gamma (scale) initialized to ones
	gammaData := make([]float64, features)
	for i := range gammaData {
		gammaData[i] = 1.0
	}
	gamma = mat.NewDense(1, features, gammaData)

	// Beta (shift) initialized to zeros
	betaData := make([]float64, features)
	beta = mat.NewDense(1, features, betaData)

	return gamma, beta
}

// CreateLayerNormParams creates initialized gamma and beta parameters for layer normalization
func CreateLayerNormParams(features int) (gamma, beta *mat.Dense) {
	// Same initialization as batch norm
	return CreateBatchNormParams(features)
}

// InitializeRunningStats creates initialized running mean and variance for batch normalization
func InitializeRunningStats(features int) (runningMean, runningVar *mat.Dense) {
	// Running mean initialized to zeros
	runningMeanData := make([]float64, features)
	runningMean = mat.NewDense(1, features, runningMeanData)

	// Running variance initialized to ones
	runningVarData := make([]float64, features)
	for i := range runningVarData {
		runningVarData[i] = 1.0
	}
	runningVar = mat.NewDense(1, features, runningVarData)

	return runningMean, runningVar
}

// Batch processing functions

// BatchGPUBatchNorm applies batch normalization to multiple input matrices
func BatchGPUBatchNorm(inputs []mat.Matrix, gamma, beta, runningMean, runningVar mat.Matrix, epsilon float32, training bool) []*mat.Dense {
	results := make([]*mat.Dense, len(inputs))

	for i, input := range inputs {
		results[i] = GPUBatchNormalize(input, gamma, beta, runningMean, runningVar, epsilon, training)
	}

	return results
}

// BatchGPULayerNorm applies layer normalization to multiple input matrices
func BatchGPULayerNorm(inputs []mat.Matrix, gamma, beta mat.Matrix, epsilon float32) []*mat.Dense {
	results := make([]*mat.Dense, len(inputs))

	for i, input := range inputs {
		results[i] = GPULayerNormalize(input, gamma, beta, epsilon)
	}

	return results
}
