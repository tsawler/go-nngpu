package matrix

import (
	"math"
	"math/rand"
	"testing"

	"github.com/tsawler/go-nngpu/tensor"
)

func TestConv2DForward(t *testing.T) {
	// Create a sample input tensor (batch=1, height=6, width=6, channels=1)
	batch, height, width, channels := 1, 6, 6, 1
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)

	// Fill with a simple pattern
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			idx := i*width*channels + j*channels + 0
			inputData[idx] = float32(i + j)
		}
	}

	input, err := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Create a 3x3 edge detection kernel
	kernelH, kernelW := 3, 3
	inputChannels, outputChannels := 1, 1
	kernelData := []float32{
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1,
	}

	kernel, err := tensor.NewTensor([]int{kernelH, kernelW, inputChannels, outputChannels}, kernelData)
	if err != nil {
		t.Fatalf("Failed to create kernel tensor: %v", err)
	}
	defer kernel.ReleaseGPU()

	// Perform convolution
	params := Conv2DParams{
		StrideH: 1,
		StrideW: 1,
		PadH:    1,
		PadW:    1,
	}

	result, err := Conv2DForward(input, kernel, params)
	if err != nil {
		t.Fatalf("Convolution failed: %v", err)
	}
	defer result.ReleaseGPU()

	// Verify output dimensions
	expectedH, expectedW := CalculateConv2DOutputSize(height, width, kernelH, kernelW, params.StrideH, params.StrideW, params.PadH, params.PadW)
	if result.Output.Shape[1] != expectedH || result.Output.Shape[2] != expectedW {
		t.Errorf("Unexpected output dimensions: got %dx%d, expected %dx%d",
			result.Output.Shape[1], result.Output.Shape[2], expectedH, expectedW)
	}

	// Verify the output shape matches input (due to padding)
	if result.Output.Shape[0] != batch || result.Output.Shape[3] != outputChannels {
		t.Errorf("Unexpected output shape: %v", result.Output.Shape)
	}
}

func TestConv2DBackward(t *testing.T) {
	// Create input tensor
	batch, height, width, inputChannels := 1, 4, 4, 2
	inputSize := batch * height * width * inputChannels
	inputData := make([]float32, inputSize)

	rand.Seed(42)
	for i := range inputData {
		inputData[i] = rand.Float32()*2 - 1
	}

	input, err := tensor.NewTensor([]int{batch, height, width, inputChannels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Create kernel
	kernelH, kernelW, outputChannels := 3, 3, 3
	kernelSize := kernelH * kernelW * inputChannels * outputChannels
	kernelData := make([]float32, kernelSize)

	for i := range kernelData {
		kernelData[i] = rand.Float32()*0.2 - 0.1
	}

	kernel, err := tensor.NewTensor([]int{kernelH, kernelW, inputChannels, outputChannels}, kernelData)
	if err != nil {
		t.Fatalf("Failed to create kernel tensor: %v", err)
	}
	defer kernel.ReleaseGPU()

	// Forward pass
	params := Conv2DParams{
		StrideH: 1,
		StrideW: 1,
		PadH:    0,
		PadW:    0,
	}

	forwardResult, err := Conv2DForward(input, kernel, params)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	defer forwardResult.ReleaseGPU()

	// Create gradient tensor
	gradOutputSize := batch * forwardResult.Output.Shape[1] * forwardResult.Output.Shape[2] * outputChannels
	gradOutputData := make([]float32, gradOutputSize)

	for i := range gradOutputData {
		gradOutputData[i] = rand.Float32() * 0.1
	}

	gradOutput, err := tensor.NewTensor(forwardResult.Output.Shape, gradOutputData)
	if err != nil {
		t.Fatalf("Failed to create gradient output tensor: %v", err)
	}
	defer gradOutput.ReleaseGPU()

	// Test backward pass - input gradients
	gradInput, err := Conv2DBackwardInput(gradOutput, kernel, input.Shape, params)
	if err != nil {
		t.Fatalf("Backward input pass failed: %v", err)
	}
	defer gradInput.ReleaseGPU()

	// Verify gradient shapes match original tensors
	for i, dim := range input.Shape {
		if gradInput.Shape[i] != dim {
			t.Errorf("Input gradient shape mismatch at dimension %d: %d vs %d", i, gradInput.Shape[i], dim)
		}
	}

	// Test backward pass - kernel gradients
	gradKernel, err := Conv2DBackwardKernel(input, gradOutput, kernel.Shape, params)
	if err != nil {
		t.Fatalf("Backward kernel pass failed: %v", err)
	}
	defer gradKernel.ReleaseGPU()

	for i, dim := range kernel.Shape {
		if gradKernel.Shape[i] != dim {
			t.Errorf("Kernel gradient shape mismatch at dimension %d: %d vs %d", i, gradKernel.Shape[i], dim)
		}
	}
}

func TestMaxPool2D(t *testing.T) {
	// Create input tensor
	batch, height, width, channels := 1, 4, 4, 2
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)

	// Fill with known values for easier verification
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	input, err := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Test max pooling
	params := Pool2DParams{
		PoolH:   2,
		PoolW:   2,
		StrideH: 2,
		StrideW: 2,
		PadH:    0,
		PadW:    0,
	}

	result, err := MaxPool2DForward(input, params)
	if err != nil {
		t.Fatalf("Max pooling failed: %v", err)
	}
	defer result.ReleaseGPU()

	// Verify output dimensions
	expectedH, expectedW := CalculatePool2DOutputSize(height, width, params.PoolH, params.PoolW,
		params.StrideH, params.StrideW, params.PadH, params.PadW)

	if result.Output.Shape[1] != expectedH || result.Output.Shape[2] != expectedW {
		t.Errorf("Unexpected pooling output dimensions: got %dx%d, expected %dx%d",
			result.Output.Shape[1], result.Output.Shape[2], expectedH, expectedW)
	}

	// Test backward pass
	gradOutputSize := batch * result.Output.Shape[1] * result.Output.Shape[2] * channels
	gradOutputData := make([]float32, gradOutputSize)
	for i := range gradOutputData {
		gradOutputData[i] = 1.0 // Uniform gradient
	}

	gradOutput, err := tensor.NewTensor(result.Output.Shape, gradOutputData)
	if err != nil {
		t.Fatalf("Failed to create gradient output tensor: %v", err)
	}
	defer gradOutput.ReleaseGPU()

	gradInput, err := MaxPool2DBackward(gradOutput, result.Indices, input.Shape, params)
	if err != nil {
		t.Fatalf("Max pooling backward failed: %v", err)
	}
	defer gradInput.ReleaseGPU()

	// Verify gradient input shape
	for i, dim := range input.Shape {
		if gradInput.Shape[i] != dim {
			t.Errorf("Gradient input shape mismatch at dimension %d: %d vs %d", i, gradInput.Shape[i], dim)
		}
	}
}

func TestAvgPool2D(t *testing.T) {
	// Create input tensor
	batch, height, width, channels := 1, 6, 6, 1
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)

	// Fill with constant values for easy average calculation
	for i := range inputData {
		inputData[i] = 4.0
	}

	input, err := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Test average pooling
	params := Pool2DParams{
		PoolH:   2,
		PoolW:   2,
		StrideH: 2,
		StrideW: 2,
		PadH:    0,
		PadW:    0,
	}

	result, err := AvgPool2DForward(input, params)
	if err != nil {
		t.Fatalf("Avg pooling failed: %v", err)
	}
	defer result.ReleaseGPU()

	// Verify output dimensions
	expectedH, expectedW := CalculatePool2DOutputSize(height, width, params.PoolH, params.PoolW,
		params.StrideH, params.StrideW, params.PadH, params.PadW)

	if result.Shape[1] != expectedH || result.Shape[2] != expectedW {
		t.Errorf("Unexpected pooling output dimensions: got %dx%d, expected %dx%d",
			result.Shape[1], result.Shape[2], expectedH, expectedW)
	}

	// Verify the average values (should all be 4.0)
	if err := result.RetrieveCPU(); err != nil {
		t.Fatalf("Failed to retrieve result: %v", err)
	}

	for i, val := range result.Data {
		if math.Abs(float64(val-4.0)) > 1e-6 {
			t.Errorf("Expected average value 4.0, got %f at index %d", val, i)
		}
	}
}

func TestPadding(t *testing.T) {
	// Create a small input tensor
	batch, height, width, channels := 1, 3, 3, 1
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)

	for i := 0; i < inputSize; i++ {
		inputData[i] = float32(i + 1)
	}

	input, err := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Test padding
	padTop, padBottom, padLeft, padRight := 1, 1, 2, 2
	padValue := float32(0.0)

	padded, err := Pad2D(input, padTop, padBottom, padLeft, padRight, padValue)
	if err != nil {
		t.Fatalf("Padding failed: %v", err)
	}
	defer padded.ReleaseGPU()

	// Verify padded dimensions
	expectedH := height + padTop + padBottom
	expectedW := width + padLeft + padRight

	if padded.Shape[1] != expectedH || padded.Shape[2] != expectedW {
		t.Errorf("Unexpected padded dimensions: got %dx%d, expected %dx%d",
			padded.Shape[1], padded.Shape[2], expectedH, expectedW)
	}

	// Test unpadding
	unpadded, err := Unpad2D(padded, padTop, padBottom, padLeft, padRight)
	if err != nil {
		t.Fatalf("Unpadding failed: %v", err)
	}
	defer unpadded.ReleaseGPU()

	// Verify we're back to original dimensions
	for i, dim := range input.Shape {
		if unpadded.Shape[i] != dim {
			t.Errorf("Unpadding failed: dimension %d mismatch %d vs %d", i, unpadded.Shape[i], dim)
		}
	}
}

func TestIm2Col(t *testing.T) {
	// Create a small input tensor for easier verification
	batch, height, width, channels := 1, 3, 3, 2
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)

	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	input, err := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Perform im2col operation
	kernelH, kernelW := 2, 2
	strideH, strideW := 1, 1
	padH, padW := 0, 0

	colResult, err := Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
	if err != nil {
		t.Fatalf("Im2col failed: %v", err)
	}
	defer colResult.ReleaseGPU()

	// Verify expected dimensions
	outputH := (height+2*padH-kernelH)/strideH + 1
	outputW := (width+2*padW-kernelW)/strideW + 1
	expectedRows := batch * outputH * outputW
	expectedCols := kernelH * kernelW * channels

	if colResult.Shape[0] != expectedRows || colResult.Shape[1] != expectedCols {
		t.Errorf("Unexpected im2col dimensions: got %dx%d, expected %dx%d",
			colResult.Shape[0], colResult.Shape[1], expectedRows, expectedCols)
	}

	// Test col2im (inverse operation)
	originalShape := []int{batch, height, width, channels}
	reconstructed, err := Col2Im(colResult, originalShape, kernelH, kernelW, strideH, strideW, padH, padW)
	if err != nil {
		t.Fatalf("Col2im failed: %v", err)
	}
	defer reconstructed.ReleaseGPU()

	// Verify we get back the original shape
	for i, dim := range originalShape {
		if reconstructed.Shape[i] != dim {
			t.Errorf("Col2im shape mismatch at dimension %d: %d vs %d", i, reconstructed.Shape[i], dim)
		}
	}
}

func TestBatchConv2D(t *testing.T) {
	// Create multiple input tensors
	numInputs := 3
	batch, height, width, channels := 1, 4, 4, 1

	inputs := make([]*tensor.Tensor, numInputs)
	for i := 0; i < numInputs; i++ {
		inputSize := batch * height * width * channels
		inputData := make([]float32, inputSize)

		for j := range inputData {
			inputData[j] = float32(i+1) * float32(j+1)
		}

		input, err := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
		if err != nil {
			t.Fatalf("Failed to create input tensor %d: %v", i, err)
		}
		inputs[i] = input
	}

	defer func() {
		for _, input := range inputs {
			if input != nil {
				input.ReleaseGPU()
			}
		}
	}()

	// Create kernel
	kernelH, kernelW := 3, 3
	inputChannels, outputChannels := 1, 2
	kernelSize := kernelH * kernelW * inputChannels * outputChannels
	kernelData := make([]float32, kernelSize)

	for i := range kernelData {
		kernelData[i] = 0.1
	}

	kernel, err := tensor.NewTensor([]int{kernelH, kernelW, inputChannels, outputChannels}, kernelData)
	if err != nil {
		t.Fatalf("Failed to create kernel tensor: %v", err)
	}
	defer kernel.ReleaseGPU()

	// Batch convolution
	params := Conv2DParams{
		StrideH: 1,
		StrideW: 1,
		PadH:    1,
		PadW:    1,
	}

	results, err := BatchGPUConv2D(inputs, kernel, params)
	if err != nil {
		t.Fatalf("Batch convolution failed: %v", err)
	}

	defer func() {
		for _, result := range results {
			if result != nil {
				result.ReleaseGPU()
			}
		}
	}()

	// Verify all results
	if len(results) != numInputs {
		t.Errorf("Expected %d results, got %d", numInputs, len(results))
	}

	for i, result := range results {
		if result == nil || result.Output == nil {
			t.Errorf("Result %d is nil", i)
			continue
		}

		expectedShape := []int{batch, height, width, outputChannels}
		for j, dim := range expectedShape {
			if result.Output.Shape[j] != dim {
				t.Errorf("Result %d shape mismatch at dimension %d: %d vs %d", i, j, result.Output.Shape[j], dim)
			}
		}
	}
}

func TestConvLayerForwardBackward(t *testing.T) {
	// Test the GPUConvLayer
	inputChannels, outputChannels := 2, 4
	kernelSize, stride, padding := 3, 1, 1

	layer := NewGPUConvLayer(inputChannels, outputChannels, kernelSize, stride, padding)
	defer layer.ReleaseGPU()

	// Create input
	batch, height, width := 1, 8, 8
	inputSize := batch * height * width * inputChannels
	inputData := make([]float32, inputSize)

	rand.Seed(123)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	input, err := tensor.NewTensor([]int{batch, height, width, inputChannels}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.ReleaseGPU()

	// Forward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	defer output.ReleaseGPU()

	// Verify output shape
	expectedShape := []int{batch, height, width, outputChannels}
	for i, dim := range expectedShape {
		if output.Shape[i] != dim {
			t.Errorf("Output shape mismatch at dimension %d: %d vs %d", i, output.Shape[i], dim)
		}
	}

	// Create gradient for backward pass
	gradOutputSize := batch * height * width * outputChannels
	gradOutputData := make([]float32, gradOutputSize)
	for i := range gradOutputData {
		gradOutputData[i] = rand.Float32() * 0.01
	}

	gradOutput, err := tensor.NewTensor(expectedShape, gradOutputData)
	if err != nil {
		t.Fatalf("Failed to create gradient output tensor: %v", err)
	}
	defer gradOutput.ReleaseGPU()

	// Backward pass
	gradInput, err := layer.Backward(gradOutput)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}
	defer gradInput.ReleaseGPU()

	// Verify gradient input shape matches original input
	for i, dim := range input.Shape {
		if gradInput.Shape[i] != dim {
			t.Errorf("Gradient input shape mismatch at dimension %d: %d vs %d", i, gradInput.Shape[i], dim)
		}
	}
}

// Benchmarks

func BenchmarkConv2DForward(b *testing.B) {
	// Setup
	batch, height, width, channels := 1, 32, 32, 3
	kernelH, kernelW, outputChannels := 3, 3, 16

	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	input, _ := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	defer input.ReleaseGPU()

	kernelSize := kernelH * kernelW * channels * outputChannels
	kernelData := make([]float32, kernelSize)
	for i := range kernelData {
		kernelData[i] = rand.Float32() * 0.1
	}

	kernel, _ := tensor.NewTensor([]int{kernelH, kernelW, channels, outputChannels}, kernelData)
	defer kernel.ReleaseGPU()

	params := Conv2DParams{StrideH: 1, StrideW: 1, PadH: 1, PadW: 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := Conv2DForward(input, kernel, params)
		if err != nil {
			b.Fatalf("Convolution failed: %v", err)
		}
		result.ReleaseGPU()
	}
}

func BenchmarkMaxPool2D(b *testing.B) {
	// Setup
	batch, height, width, channels := 1, 64, 64, 16
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	input, _ := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	defer input.ReleaseGPU()

	params := Pool2DParams{PoolH: 2, PoolW: 2, StrideH: 2, StrideW: 2, PadH: 0, PadW: 0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := MaxPool2DForward(input, params)
		if err != nil {
			b.Fatalf("Max pooling failed: %v", err)
		}
		result.ReleaseGPU()
	}
}

func BenchmarkIm2Col(b *testing.B) {
	// Setup
	batch, height, width, channels := 1, 32, 32, 8
	inputSize := batch * height * width * channels
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	input, _ := tensor.NewTensor([]int{batch, height, width, channels}, inputData)
	defer input.ReleaseGPU()

	kernelH, kernelW := 3, 3
	strideH, strideW := 1, 1
	padH, padW := 1, 1

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
		if err != nil {
			b.Fatalf("Im2Col failed: %v", err)
		}
		result.ReleaseGPU()
	}
}
