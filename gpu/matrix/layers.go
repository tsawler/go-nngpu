package matrix

import (
	"math"
	"math/rand"
	
	"github.com/tsawler/go-nngpu/tensor"
)

// LinearLayer represents a fully connected layer
type LinearLayer struct {
	// Parameters
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
	
	// Gradients
	WeightGrad *tensor.Tensor
	BiasGrad   *tensor.Tensor
	
	// Cache for backward pass
	input *tensor.Tensor
	
	// Configuration
	InputSize  int
	OutputSize int
	UseBias    bool
}

// NewLinearLayer creates a new linear layer
func NewLinearLayer(inputSize, outputSize int, useBias bool) *LinearLayer {
	// Initialize weights using Xavier initialization
	weightData := make([]float32, inputSize*outputSize)
	scale := float32(math.Sqrt(2.0 / float64(inputSize+outputSize)))
	for i := range weightData {
		weightData[i] = (rand.Float32()*2 - 1) * scale
	}
	weight, _ := tensor.NewTensor([]int{inputSize, outputSize}, weightData)
	
	var bias *tensor.Tensor
	if useBias {
		biasData := make([]float32, outputSize)
		bias, _ = tensor.NewTensor([]int{outputSize}, biasData)
	}
	
	return &LinearLayer{
		Weight:     weight,
		Bias:       bias,
		InputSize:  inputSize,
		OutputSize: outputSize,
		UseBias:    useBias,
	}
}

// Forward performs the forward pass
func (l *LinearLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	// Cache input for backward pass
	l.input = input
	
	// Compute output = input @ weight + bias
	output, err := MatMul(input, l.Weight)
	if err != nil {
		// Return zero output on error
		outputData := make([]float32, input.Shape[0]*l.OutputSize)
		output, _ = tensor.NewTensor([]int{input.Shape[0], l.OutputSize}, outputData)
		return output
	}
	
	if l.UseBias && l.Bias != nil {
		// Add bias
		output = l.addBias(output, l.Bias)
	}
	
	return output
}

// Backward performs the backward pass
func (l *LinearLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	if l.input == nil {
		panic("Backward called before Forward")
	}
	
	// Compute weight gradient: input.T @ gradOutput
	inputT, _ := Transpose(l.input)
	l.WeightGrad, _ = MatMul(inputT, gradOutput)
	
	// Compute bias gradient: sum(gradOutput, axis=0)
	if l.UseBias && l.Bias != nil {
		l.BiasGrad = l.sumAlongAxis(gradOutput, 0)
	}
	
	// Compute input gradient: gradOutput @ weight.T
	weightT, _ := Transpose(l.Weight)
	gradInput, _ := MatMul(gradOutput, weightT)
	
	return gradInput
}

// GetParameters returns the layer parameters
func (l *LinearLayer) GetParameters() []*tensor.Tensor {
	if l.UseBias && l.Bias != nil {
		return []*tensor.Tensor{l.Weight, l.Bias}
	}
	return []*tensor.Tensor{l.Weight}
}

// GetGradients returns the layer gradients
func (l *LinearLayer) GetGradients() []*tensor.Tensor {
	if l.UseBias && l.BiasGrad != nil {
		return []*tensor.Tensor{l.WeightGrad, l.BiasGrad}
	}
	return []*tensor.Tensor{l.WeightGrad}
}

// ReLULayer represents a ReLU activation layer
type ReLULayer struct {
	// Cache for backward pass
	input  *tensor.Tensor
	output *tensor.Tensor
}

// NewReLULayer creates a new ReLU layer
func NewReLULayer() *ReLULayer {
	return &ReLULayer{}
}

// Forward performs the forward pass
func (r *ReLULayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	r.input = input
	r.output, _ = ReLUForward(input)
	return r.output
}

// Backward performs the backward pass
func (r *ReLULayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	if r.input == nil {
		panic("Backward called before Forward")
	}
	
	// ReLU gradient: gradOutput * (input > 0)
	gradInput, _ := ReLUBackward(gradOutput, r.output)
	return gradInput
}

// GetParameters returns empty slice (no parameters)
func (r *ReLULayer) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// GetGradients returns empty slice (no gradients)
func (r *ReLULayer) GetGradients() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// Conv2DLayer represents a 2D convolution layer
type Conv2DLayer struct {
	// Parameters
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
	
	// Gradients
	WeightGrad *tensor.Tensor
	BiasGrad   *tensor.Tensor
	
	// Cache for backward pass
	input *tensor.Tensor
	
	// Configuration
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
}

// NewConv2DLayer creates a new Conv2D layer
func NewConv2DLayer(inChannels, outChannels, kernelSize, stride, padding int) *Conv2DLayer {
	// Initialize weights
	weightData := make([]float32, outChannels*inChannels*kernelSize*kernelSize)
	scale := float32(math.Sqrt(2.0 / float64(inChannels*kernelSize*kernelSize)))
	for i := range weightData {
		weightData[i] = (rand.Float32()*2 - 1) * scale
	}
	weight, _ := tensor.NewTensor([]int{outChannels, inChannels, kernelSize, kernelSize}, weightData)
	
	// Initialize bias
	biasData := make([]float32, outChannels)
	bias, _ := tensor.NewTensor([]int{outChannels}, biasData)
	
	return &Conv2DLayer{
		Weight:      weight,
		Bias:        bias,
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
	}
}

// Forward performs the forward pass
func (c *Conv2DLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	c.input = input
	
	// Create Conv2D parameters
	params := Conv2DParams{
		StrideH: c.Stride,
		StrideW: c.Stride,
		PadH:    c.Padding,
		PadW:    c.Padding,
	}
	
	// Perform convolution
	result, err := Conv2DForward(input, c.Weight, params)
	if err != nil {
		// Return zero output on error
		return c.createZeroOutput(input)
	}
	
	output := result.Output
	
	// Add bias if present
	if c.Bias != nil {
		output = c.addConvBias(output, c.Bias)
	}
	
	return output
}

// Backward performs the backward pass
func (c *Conv2DLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	if c.input == nil {
		panic("Backward called before Forward")
	}
	
	// TODO: Implement Conv2D backward pass
	// For now, return zero gradients
	gradInputData := make([]float32, len(c.input.Data))
	gradInput, _ := tensor.NewTensor(c.input.Shape, gradInputData)
	
	// Create zero weight gradients
	weightGradData := make([]float32, len(c.Weight.Data))
	c.WeightGrad, _ = tensor.NewTensor(c.Weight.Shape, weightGradData)
	
	// Create zero bias gradients if bias exists
	if c.Bias != nil {
		biasGradData := make([]float32, len(c.Bias.Data))
		c.BiasGrad, _ = tensor.NewTensor(c.Bias.Shape, biasGradData)
	}
	
	return gradInput
}

// GetParameters returns the layer parameters
func (c *Conv2DLayer) GetParameters() []*tensor.Tensor {
	if c.Bias != nil {
		return []*tensor.Tensor{c.Weight, c.Bias}
	}
	return []*tensor.Tensor{c.Weight}
}

// GetGradients returns the layer gradients
func (c *Conv2DLayer) GetGradients() []*tensor.Tensor {
	if c.Bias != nil && c.BiasGrad != nil {
		return []*tensor.Tensor{c.WeightGrad, c.BiasGrad}
	}
	return []*tensor.Tensor{c.WeightGrad}
}

// Helper methods

func (l *LinearLayer) addBias(output, bias *tensor.Tensor) *tensor.Tensor {
	// Add bias to each row
	for i := 0; i < output.Shape[0]; i++ {
		for j := 0; j < output.Shape[1]; j++ {
			output.Data[i*output.Shape[1]+j] += bias.Data[j]
		}
	}
	return output
}

func (l *LinearLayer) sumAlongAxis(input *tensor.Tensor, axis int) *tensor.Tensor {
	if axis == 0 {
		// Sum along rows
		result := make([]float32, input.Shape[1])
		for i := 0; i < input.Shape[0]; i++ {
			for j := 0; j < input.Shape[1]; j++ {
				result[j] += input.Data[i*input.Shape[1]+j]
			}
		}
		output, _ := tensor.NewTensor([]int{input.Shape[1]}, result)
		return output
	}
	return nil
}

func (c *Conv2DLayer) createZeroOutput(input *tensor.Tensor) *tensor.Tensor {
	// Calculate output dimensions
	batch := input.Shape[0]
	height := (input.Shape[2]+2*c.Padding-c.KernelSize)/c.Stride + 1
	width := (input.Shape[3]+2*c.Padding-c.KernelSize)/c.Stride + 1
	
	outputData := make([]float32, batch*c.OutChannels*height*width)
	output, _ := tensor.NewTensor([]int{batch, c.OutChannels, height, width}, outputData)
	return output
}

func (c *Conv2DLayer) addConvBias(output, bias *tensor.Tensor) *tensor.Tensor {
	// Add bias to each channel
	batch := output.Shape[0]
	channels := output.Shape[1]
	height := output.Shape[2]
	width := output.Shape[3]
	
	for b := 0; b < batch; b++ {
		for ch := 0; ch < channels; ch++ {
			biasVal := bias.Data[ch]
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					idx := b*channels*height*width + ch*height*width + h*width + w
					output.Data[idx] += biasVal
				}
			}
		}
	}
	
	return output
}

// Sequential represents a sequential container of layers
type Sequential struct {
	layers []Layer
}

// NewSequential creates a new sequential model
func NewSequential(layers ...Layer) *Sequential {
	return &Sequential{
		layers: layers,
	}
}

// Forward performs forward pass through all layers
func (s *Sequential) Forward(input *tensor.Tensor) *tensor.Tensor {
	output := input
	for _, layer := range s.layers {
		output = layer.Forward(output)
	}
	return output
}

// Backward performs backward pass through all layers
func (s *Sequential) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	grad := gradOutput
	// Backward through layers in reverse order
	for i := len(s.layers) - 1; i >= 0; i-- {
		grad = s.layers[i].Backward(grad)
	}
	return grad
}

// GetParameters returns all parameters from all layers
func (s *Sequential) GetParameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	for _, layer := range s.layers {
		params = append(params, layer.GetParameters()...)
	}
	return params
}

// GetGradients returns all gradients from all layers
func (s *Sequential) GetGradients() []*tensor.Tensor {
	var grads []*tensor.Tensor
	for _, layer := range s.layers {
		grads = append(grads, layer.GetGradients()...)
	}
	return grads
}