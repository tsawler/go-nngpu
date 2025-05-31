package matrix

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tsawler/go-nngpu/tensor"
	"gonum.org/v1/gonum/mat"
)

// GPUConvLayer represents a convolutional layer with GPU acceleration
type GPUConvLayer struct {
	InputChannels  int
	OutputChannels int
	KernelSize     int // Assuming square kernels for simplicity
	Stride         int
	Padding        int
	Weights        *GPUDense      // Kernel weights as a matrix
	Bias           *GPUDense      // Bias vector
	LastInput      *tensor.Tensor // Stored for backward pass
	LastOutput     *Conv2DResult  // Stored for backward pass
}

// NewGPUConvLayer creates a new GPU-accelerated convolutional layer
func NewGPUConvLayer(inputChannels, outputChannels, kernelSize, stride, padding int) *GPUConvLayer {
	// Initialize weights with Xavier/Glorot initialization
	fanIn := float64(inputChannels * kernelSize * kernelSize)
	fanOut := float64(outputChannels * kernelSize * kernelSize)
	stddev := math.Sqrt(2.0 / (fanIn + fanOut))

	weightsSize := kernelSize * kernelSize * inputChannels * outputChannels
	weightsData := make([]float64, weightsSize)
	for i := range weightsData {
		weightsData[i] = stddev * (rand.Float64()*2 - 1) // Random between -stddev and +stddev
	}

	weights := NewGPUDense(kernelSize*kernelSize*inputChannels, outputChannels, weightsData)

	// Initialize bias to zero
	biasData := make([]float64, outputChannels)
	bias := NewGPUDense(1, outputChannels, biasData)

	return &GPUConvLayer{
		InputChannels:  inputChannels,
		OutputChannels: outputChannels,
		KernelSize:     kernelSize,
		Stride:         stride,
		Padding:        padding,
		Weights:        weights,
		Bias:           bias,
	}
}

// Forward performs the forward pass of the convolutional layer
func (layer *GPUConvLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input must be 4D tensor (batch, height, width, channels)")
	}

	// Store input for backward pass
	layer.LastInput = input

	// Reshape weights to proper kernel format
	kernelShape := []int{layer.KernelSize, layer.KernelSize, layer.InputChannels, layer.OutputChannels}
	kernelData := make([]float32, len(layer.Weights.tensor.Data))
	for i, v := range layer.Weights.tensor.Data {
		kernelData[i] = float32(v)
	}

	kernel, err := tensor.NewTensor(kernelShape, kernelData)
	if err != nil {
		return nil, fmt.Errorf("failed to create kernel tensor: %w", err)
	}
	defer kernel.ReleaseGPU()

	// Perform convolution
	params := Conv2DParams{
		StrideH: layer.Stride,
		StrideW: layer.Stride,
		PadH:    layer.Padding,
		PadW:    layer.Padding,
	}

	result, err := Conv2DForward(input, kernel, params)
	if err != nil {
		return nil, fmt.Errorf("convolution forward failed: %w", err)
	}

	// Store result for backward pass
	layer.LastOutput = result

	// Add bias if provided
	if layer.Bias != nil {
		// TODO: For now, we'll skip bias addition in this simplified implementation
		// In a full implementation, you'd add bias broadcasting
	}

	return result.Output, nil
}

// Backward performs the backward pass of the convolutional layer
func (layer *GPUConvLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error) {
	if layer.LastInput == nil || layer.LastOutput == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	// Reshape weights to proper kernel format
	kernelShape := []int{layer.KernelSize, layer.KernelSize, layer.InputChannels, layer.OutputChannels}
	kernelData := make([]float32, len(layer.Weights.tensor.Data))
	for i, v := range layer.Weights.tensor.Data {
		kernelData[i] = float32(v)
	}

	kernel, err := tensor.NewTensor(kernelShape, kernelData)
	if err != nil {
		return nil, fmt.Errorf("failed to create kernel tensor: %w", err)
	}
	defer kernel.ReleaseGPU()

	params := Conv2DParams{
		StrideH: layer.Stride,
		StrideW: layer.Stride,
		PadH:    layer.Padding,
		PadW:    layer.Padding,
	}

	// Compute gradients with respect to input
	gradInput, err := Conv2DBackwardInput(gradOutput, kernel, layer.LastInput.Shape, params)
	if err != nil {
		return nil, fmt.Errorf("convolution backward input failed: %w", err)
	}

	// Compute gradients with respect to weights (for optimization)
	gradKernel, err := Conv2DBackwardKernel(layer.LastInput, gradOutput, kernelShape, params)
	if err != nil {
		defer gradInput.ReleaseGPU()
		return nil, fmt.Errorf("convolution backward kernel failed: %w", err)
	}
	defer gradKernel.ReleaseGPU()

	// Update weights (simple gradient descent - in practice you'd use an optimizer)
	// This is a simplified update - real implementations would use proper optimizers
	learningRate := float32(0.001)
	if err := gradKernel.RetrieveCPU(); err != nil {
		defer gradInput.ReleaseGPU()
		return nil, fmt.Errorf("failed to retrieve kernel gradients: %w", err)
	}

	// Update weights
	for i, grad := range gradKernel.Data {
		if i < len(layer.Weights.tensor.Data) {
			layer.Weights.tensor.Data[i] -= float32(learningRate) * grad
		}
	}

	return gradInput, nil
}

// ReleaseGPU releases GPU resources
func (layer *GPUConvLayer) ReleaseGPU() {
	if layer.Weights != nil {
		layer.Weights.ReleaseGPU()
	}
	if layer.Bias != nil {
		layer.Bias.ReleaseGPU()
	}
	if layer.LastOutput != nil {
		layer.LastOutput.ReleaseGPU()
		layer.LastOutput = nil
	}
	layer.LastInput = nil
}

// GPUPoolLayer represents a pooling layer with GPU acceleration
type GPUPoolLayer struct {
	PoolType   string // "max" or "avg"
	PoolSize   int
	Stride     int
	Padding    int
	LastInput  *tensor.Tensor
	LastResult *MaxPool2DResult // For max pooling (contains indices)
}

// NewGPUMaxPoolLayer creates a new GPU-accelerated max pooling layer
func NewGPUMaxPoolLayer(poolSize, stride, padding int) *GPUPoolLayer {
	return &GPUPoolLayer{
		PoolType: "max",
		PoolSize: poolSize,
		Stride:   stride,
		Padding:  padding,
	}
}

// NewGPUAvgPoolLayer creates a new GPU-accelerated average pooling layer
func NewGPUAvgPoolLayer(poolSize, stride, padding int) *GPUPoolLayer {
	return &GPUPoolLayer{
		PoolType: "avg",
		PoolSize: poolSize,
		Stride:   stride,
		Padding:  padding,
	}
}

// Forward performs the forward pass of the pooling layer
func (layer *GPUPoolLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input must be 4D tensor (batch, height, width, channels)")
	}

	layer.LastInput = input

	params := Pool2DParams{
		PoolH:   layer.PoolSize,
		PoolW:   layer.PoolSize,
		StrideH: layer.Stride,
		StrideW: layer.Stride,
		PadH:    layer.Padding,
		PadW:    layer.Padding,
	}

	switch layer.PoolType {
	case "max":
		result, err := MaxPool2DForward(input, params)
		if err != nil {
			return nil, fmt.Errorf("max pooling forward failed: %w", err)
		}
		layer.LastResult = result
		return result.Output, nil

	case "avg":
		output, err := AvgPool2DForward(input, params)
		if err != nil {
			return nil, fmt.Errorf("avg pooling forward failed: %w", err)
		}
		return output, nil

	default:
		return nil, fmt.Errorf("unsupported pool type: %s", layer.PoolType)
	}
}

// Backward performs the backward pass of the pooling layer
func (layer *GPUPoolLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error) {
	if layer.LastInput == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	params := Pool2DParams{
		PoolH:   layer.PoolSize,
		PoolW:   layer.PoolSize,
		StrideH: layer.Stride,
		StrideW: layer.Stride,
		PadH:    layer.Padding,
		PadW:    layer.Padding,
	}

	switch layer.PoolType {
	case "max":
		if layer.LastResult == nil {
			return nil, fmt.Errorf("max pooling forward result not available")
		}
		return MaxPool2DBackward(gradOutput, layer.LastResult.Indices, layer.LastInput.Shape, params)

	case "avg":
		return AvgPool2DBackward(gradOutput, layer.LastInput.Shape, params)

	default:
		return nil, fmt.Errorf("unsupported pool type: %s", layer.PoolType)
	}
}

// ReleaseGPU releases GPU resources
func (layer *GPUPoolLayer) ReleaseGPU() {
	if layer.LastResult != nil {
		layer.LastResult.ReleaseGPU()
		layer.LastResult = nil
	}
	layer.LastInput = nil
}

// Utility functions for tensor format conversion

// TensorToGonum converts a 2D tensor to a Gonum Dense matrix
func TensorToGonum(t *tensor.Tensor) (*mat.Dense, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("tensor must be 2D for conversion to Gonum matrix")
	}

	if err := t.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve tensor data from GPU: %w", err)
	}

	rows, cols := t.Shape[0], t.Shape[1]
	data := make([]float64, len(t.Data))
	for i, v := range t.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data), nil
}

// GonumToTensor converts a Gonum Dense matrix to a 2D tensor
func GonumToTensor(m *mat.Dense) (*tensor.Tensor, error) {
	rows, cols := m.Dims()
	data := make([]float32, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = float32(m.At(i, j))
		}
	}

	return tensor.NewTensor([]int{rows, cols}, data)
}

// Flatten4DTo2D flattens a 4D tensor to 2D for compatibility with Gonum operations
func Flatten4DTo2D(t *tensor.Tensor) (*tensor.Tensor, error) {
	if len(t.Shape) != 4 {
		return nil, fmt.Errorf("tensor must be 4D")
	}

	batch := t.Shape[0]
	height := t.Shape[1]
	width := t.Shape[2]
	channels := t.Shape[3]

	// Flatten to (batch, height * width * channels)
	newShape := []int{batch, height * width * channels}

	// Data remains the same, just reshape
	return tensor.NewTensor(newShape, t.Data)
}

// Reshape2DTo4D reshapes a 2D tensor back to 4D
func Reshape2DTo4D(t *tensor.Tensor, targetShape []int) (*tensor.Tensor, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("tensor must be 2D")
	}
	if len(targetShape) != 4 {
		return nil, fmt.Errorf("target shape must be 4D")
	}

	// Verify the total size matches
	originalSize := t.Shape[0] * t.Shape[1]
	targetSize := targetShape[0] * targetShape[1] * targetShape[2] * targetShape[3]

	if originalSize != targetSize {
		return nil, fmt.Errorf("size mismatch: original %d vs target %d", originalSize, targetSize)
	}

	return tensor.NewTensor(targetShape, t.Data)
}

// Convenience functions for common convolution patterns

// GPUConv2DSimple performs a simple 2D convolution with square kernels
func GPUConv2DSimple(input *mat.Dense, kernelSize, stride, padding int, numFilters int) (*mat.Dense, error) {
	// TODO: This is a simplified interface - in practice you'd need more sophisticated tensor handling
	// Convert Gonum matrix to tensor format, perform convolution, and convert back

	// For now, this is a placeholder that demonstrates the interface
	// A full implementation would need proper 4D tensor handling from 2D matrices

	return nil, fmt.Errorf("GPUConv2DSimple: full implementation requires 4D tensor support from 2D matrices")
}

// GPUMaxPool2DSimple performs simple 2D max pooling
func GPUMaxPool2DSimple(input *mat.Dense, poolSize, stride int) (*mat.Dense, error) {
	// Placeholder for simple max pooling interface
	return nil, fmt.Errorf("GPUMaxPool2DSimple: full implementation requires 4D tensor support from 2D matrices")
}

// Batch processing functions

// BatchGPUConv2D performs convolution on multiple input tensors
func BatchGPUConv2D(inputs []*tensor.Tensor, kernel *tensor.Tensor, params Conv2DParams) ([]*Conv2DResult, error) {
	results := make([]*Conv2DResult, len(inputs))

	for i, input := range inputs {
		result, err := Conv2DForward(input, kernel, params)
		if err != nil {
			// Clean up any results that were already computed
			for j := 0; j < i; j++ {
				results[j].ReleaseGPU()
			}
			return nil, fmt.Errorf("failed to process batch item %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// BatchGPUMaxPool2D performs max pooling on multiple input tensors
func BatchGPUMaxPool2D(inputs []*tensor.Tensor, params Pool2DParams) ([]*MaxPool2DResult, error) {
	results := make([]*MaxPool2DResult, len(inputs))

	for i, input := range inputs {
		result, err := MaxPool2DForward(input, params)
		if err != nil {
			// Clean up any results that were already computed
			for j := 0; j < i; j++ {
				results[j].ReleaseGPU()
			}
			return nil, fmt.Errorf("failed to process batch item %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// Debugging and utility functions

// PrintTensorInfo prints information about a tensor's shape and GPU status
func PrintTensorInfo(name string, t *tensor.Tensor) {
	fmt.Printf("%s: Shape=%v, Size=%d, OnGPU=%t\n",
		name, t.Shape, len(t.Data), t.GPUPtr() != nil)
}

// PrintConvParams prints convolution parameters
func PrintConvParams(params Conv2DParams) {
	fmt.Printf("Conv2D Params: Stride=(%d,%d), Pad=(%d,%d)\n",
		params.StrideH, params.StrideW, params.PadH, params.PadW)
}

// PrintPoolParams prints pooling parameters
func PrintPoolParams(params Pool2DParams) {
	fmt.Printf("Pool2D Params: Pool=(%d,%d), Stride=(%d,%d), Pad=(%d,%d)\n",
		params.PoolH, params.PoolW, params.StrideH, params.StrideW, params.PadH, params.PadW)
}
