package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	_ "github.com/tsawler/go-nngpu/internal/cgo"
	"github.com/tsawler/go-nngpu/tensor"
)

// GradientMode represents different gradient computation modes
type GradientMode int

const (
	NoGrad GradientMode = iota // No gradient computation
	Grad                       // Compute gradients
)

// OpType represents different operation types for gradient computation
type OpType int

const (
	// Basic operations
	OpAdd OpType = iota
	OpSub
	OpMul
	OpDiv
	OpMatMul
	OpTranspose

	// Activation functions
	OpReLU
	OpSigmoid
	OpTanh
	OpSoftmax
	OpLeakyReLU
	OpELU
	OpSwish
	OpGELU

	// Convolution operations
	OpConv2D
	OpMaxPool2D
	OpAvgPool2D
	OpPad2D

	// Normalization operations
	OpBatchNorm
	OpLayerNorm

	// Loss operations
	OpMSELoss
	OpCrossEntropyLoss
	OpBinaryCrossEntropyLoss

	// Reduction operations
	OpSum
	OpMean
	OpMax
	OpMin

	// Reshape operations
	OpReshape
	OpFlatten
	OpSqueeze
	OpUnsqueeze
)

// GradientTensor wraps a tensor with gradient tracking capabilities
type GradientTensor struct {
	Tensor       *tensor.Tensor
	Gradient     *tensor.Tensor
	RequiresGrad bool
	GradFn       *GradientFunction
	IsLeaf       bool // True if this tensor is a leaf (input/parameter)
}

// GradientFunction represents a backward function for an operation
type GradientFunction struct {
	OpType       OpType
	Inputs       []*GradientTensor
	Outputs      []*GradientTensor
	BackwardFn   func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error)
	SavedTensors []*tensor.Tensor       // Tensors saved for backward pass
	Metadata     map[string]interface{} // Additional metadata for backward pass
}

// ComputationGraph tracks the computational graph for backpropagation
type ComputationGraph struct {
	nodes     []*GradientTensor
	gradMode  GradientMode
	leafNodes []*GradientTensor
}

// Global computation graph
var globalGraph *ComputationGraph

func init() {
	globalGraph = NewComputationGraph()
}

// NewComputationGraph creates a new computation graph
func NewComputationGraph() *ComputationGraph {
	return &ComputationGraph{
		nodes:     make([]*GradientTensor, 0),
		gradMode:  NoGrad,
		leafNodes: make([]*GradientTensor, 0),
	}
}

// NewGradientTensor creates a new gradient tensor
func NewGradientTensor(t *tensor.Tensor, requiresGrad bool) *GradientTensor {
	gt := &GradientTensor{
		Tensor:       t,
		RequiresGrad: requiresGrad,
		IsLeaf:       true,
	}

	if requiresGrad && globalGraph.gradMode == Grad {
		// Initialize gradient tensor with zeros
		gradData := make([]float32, len(t.Data))
		grad, _ := tensor.NewTensor(t.Shape, gradData)
		gt.Gradient = grad

		// Add to computation graph
		globalGraph.addNode(gt)
		globalGraph.leafNodes = append(globalGraph.leafNodes, gt)
	}

	return gt
}

// SetGradientMode sets the global gradient computation mode
func SetGradientMode(mode GradientMode) {
	globalGraph.gradMode = mode
}

// GetGradientMode returns the current gradient computation mode
func GetGradientMode() GradientMode {
	return globalGraph.gradMode
}

// NoGradContext temporarily disables gradient computation
func NoGradContext(fn func()) {
	oldMode := globalGraph.gradMode
	globalGraph.gradMode = NoGrad
	defer func() {
		globalGraph.gradMode = oldMode
	}()
	fn()
}

// addNode adds a node to the computation graph
func (cg *ComputationGraph) addNode(node *GradientTensor) {
	cg.nodes = append(cg.nodes, node)
}

// ClearGraph clears the computation graph
func ClearGraph() {
	globalGraph.nodes = globalGraph.nodes[:0]
	globalGraph.leafNodes = globalGraph.leafNodes[:0]
}

// ZeroGrad zeros all gradients in the computation graph
func ZeroGrad() {
	for _, node := range globalGraph.leafNodes {
		if node.Gradient != nil {
			// Zero out gradient data
			for i := range node.Gradient.Data {
				node.Gradient.Data[i] = 0.0
			}
		}
	}
}

// Backward performs backpropagation from this tensor
func (gt *GradientTensor) Backward() error {
	if !gt.RequiresGrad {
		return fmt.Errorf("tensor does not require gradients")
	}

	if gt.Gradient == nil {
		// Initialize gradient with ones (for scalar output) or appropriate shape
		gradData := make([]float32, len(gt.Tensor.Data))
		if len(gradData) == 1 {
			gradData[0] = 1.0 // Scalar case
		} else {
			// For non-scalar tensors, gradient must be provided externally
			return fmt.Errorf("gradient must be provided for non-scalar tensor")
		}
		var err error
		gt.Gradient, err = tensor.NewTensor(gt.Tensor.Shape, gradData)
		if err != nil {
			return fmt.Errorf("failed to create gradient tensor: %w", err)
		}
	}

	// Perform backward pass through the computational graph
	return gt.backwardRecursive(gt.Gradient)
}

// BackwardWithGradient performs backpropagation with a specific gradient
func (gt *GradientTensor) BackwardWithGradient(grad *tensor.Tensor) error {
	if !gt.RequiresGrad {
		return fmt.Errorf("tensor does not require gradients")
	}

	return gt.backwardRecursive(grad)
}

// backwardRecursive recursively computes gradients through the graph
func (gt *GradientTensor) backwardRecursive(gradOutput *tensor.Tensor) error {
	// If this is a leaf node, accumulate the gradient
	if gt.IsLeaf {
		if gt.Gradient == nil {
			var err error
			gt.Gradient, err = tensor.NewTensor(gt.Tensor.Shape, make([]float32, len(gt.Tensor.Data)))
			if err != nil {
				return fmt.Errorf("failed to create gradient tensor: %w", err)
			}
		}

		// Accumulate gradient (for cases where a variable is used multiple times)
		return AccumulateGradient(gt.Gradient, gradOutput)
	}

	// If this is not a leaf node, compute gradients for inputs
	if gt.GradFn != nil {
		inputGrads, err := gt.GradFn.BackwardFn(gradOutput)
		if err != nil {
			return fmt.Errorf("backward function failed: %w", err)
		}

		// Recursively apply gradients to input tensors
		for i, inputGrad := range inputGrads {
			if i < len(gt.GradFn.Inputs) && inputGrad != nil {
				input := gt.GradFn.Inputs[i]
				if input.RequiresGrad {
					if err := input.backwardRecursive(inputGrad); err != nil {
						return fmt.Errorf("failed to backpropagate to input %d: %w", i, err)
					}
				}
			}
		}
	}

	return nil
}

// AccumulateGradient accumulates gradients using GPU acceleration
func AccumulateGradient(existing, newGrad *tensor.Tensor) error {
	// Ensure both tensors have the same shape
	if len(existing.Shape) != len(newGrad.Shape) {
		return fmt.Errorf("gradient shape mismatch")
	}
	for i, dim := range existing.Shape {
		if dim != newGrad.Shape[i] {
			return fmt.Errorf("gradient shape mismatch at dimension %d", i)
		}
	}

	// Ensure tensors are on GPU
	if err := existing.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move existing gradient to GPU: %w", err)
	}
	if err := newGrad.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move new gradient to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	size := len(existing.Data)

	retCode := C.perform_gradient_accumulate(
		C.GPUPtr(existing.GPUPtr()),
		C.GPUPtr(newGrad.GPUPtr()),
		C.long(size),
		C.DevicePtr(existing.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("GPU gradient accumulation failed (code %d): %s", retCode, errMsg)
	}

	return nil
}

// ClipGradientNorm clips gradients by norm to prevent exploding gradients
func ClipGradientNorm(gradTensors []*GradientTensor, maxNorm float32) error {
	if len(gradTensors) == 0 {
		return nil
	}

	// Calculate total gradient norm
	totalNorm, err := CalculateGradientNorm(gradTensors)
	if err != nil {
		return fmt.Errorf("failed to calculate gradient norm: %w", err)
	}

	if totalNorm <= maxNorm {
		return nil // No clipping needed
	}

	// Calculate clipping factor
	clipFactor := maxNorm / totalNorm

	// Scale all gradients
	for _, gt := range gradTensors {
		if gt.Gradient != nil {
			scaled, err := ScalarMul(gt.Gradient, clipFactor)
			if err != nil {
				return fmt.Errorf("failed to scale gradient: %w", err)
			}

			// Replace gradient
			gt.Gradient.ReleaseGPU()
			gt.Gradient = scaled
		}
	}

	return nil
}

// CalculateGradientNorm calculates the L2 norm of gradients
func CalculateGradientNorm(gradTensors []*GradientTensor) (float32, error) {
	if len(gradTensors) == 0 {
		return 0.0, nil
	}

	// Collect all gradient tensors
	var gradients []*tensor.Tensor
	for _, gt := range gradTensors {
		if gt.Gradient != nil {
			gradients = append(gradients, gt.Gradient)
		}
	}

	if len(gradients) == 0 {
		return 0.0, nil
	}

	return CalculateL2Norm(gradients)
}

// CalculateL2Norm calculates the L2 norm of multiple tensors
func CalculateL2Norm(tensors []*tensor.Tensor) (float32, error) {
	totalSumSquares := float32(0.0)

	for _, t := range tensors {
		// Ensure tensor is on GPU
		if err := t.EnsureGPU(); err != nil {
			return 0, fmt.Errorf("failed to move tensor to GPU: %w", err)
		}

		runtime.LockOSThread()
		var cErr C.CError
		var sumSquares C.float

		retCode := C.perform_tensor_sum_squares(
			C.GPUPtr(t.GPUPtr()),
			C.long(len(t.Data)),
			&sumSquares,
			C.DevicePtr(t.DevicePtr()),
			&cErr,
		)
		runtime.UnlockOSThread()

		if retCode != 0 {
			var errMsg string
			if cErr.message != nil {
				errMsg = C.GoString(cErr.message)
				C.free_c_error_message(cErr.message)
			}
			return 0, fmt.Errorf("GPU sum squares failed (code %d): %s", retCode, errMsg)
		}

		totalSumSquares += float32(sumSquares)
	}

	return totalSumSquares, nil
}

// CreateBackwardFunction creates a backward function for gradient computation
func CreateBackwardFunction(opType OpType, inputs []*GradientTensor, savedTensors []*tensor.Tensor, metadata map[string]interface{}) *GradientFunction {
	gf := &GradientFunction{
		OpType:       opType,
		Inputs:       inputs,
		SavedTensors: savedTensors,
		Metadata:     metadata,
	}

	// Set the appropriate backward function based on operation type
	switch opType {
	case OpAdd:
		gf.BackwardFn = gf.addBackward
	case OpSub:
		gf.BackwardFn = gf.subBackward
	case OpMul:
		gf.BackwardFn = gf.mulBackward
	case OpDiv:
		gf.BackwardFn = gf.divBackward
	case OpMatMul:
		gf.BackwardFn = gf.matmulBackward
	case OpTranspose:
		gf.BackwardFn = gf.transposeBackward
	case OpReLU:
		gf.BackwardFn = gf.reluBackward
	case OpSigmoid:
		gf.BackwardFn = gf.sigmoidBackward
	case OpTanh:
		gf.BackwardFn = gf.tanhBackward
	case OpSoftmax:
		gf.BackwardFn = gf.softmaxBackward
	case OpLeakyReLU:
		gf.BackwardFn = gf.leakyReluBackward
	case OpELU:
		gf.BackwardFn = gf.eluBackward
	case OpSwish:
		gf.BackwardFn = gf.swishBackward
	case OpGELU:
		gf.BackwardFn = gf.geluBackward
	case OpConv2D:
		gf.BackwardFn = gf.conv2dBackward
	case OpMaxPool2D:
		gf.BackwardFn = gf.maxPool2dBackward
	case OpAvgPool2D:
		gf.BackwardFn = gf.avgPool2dBackward
	case OpBatchNorm:
		gf.BackwardFn = gf.batchNormBackward
	case OpLayerNorm:
		gf.BackwardFn = gf.layerNormBackward
	case OpMSELoss:
		gf.BackwardFn = gf.mseLossBackward
	case OpCrossEntropyLoss:
		gf.BackwardFn = gf.crossEntropyLossBackward
	case OpBinaryCrossEntropyLoss:
		gf.BackwardFn = gf.binaryCrossEntropyLossBackward
	case OpSum:
		gf.BackwardFn = gf.sumBackward
	case OpMean:
		gf.BackwardFn = gf.meanBackward
	case OpReshape:
		gf.BackwardFn = gf.reshapeBackward
	default:
		gf.BackwardFn = func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
			return nil, fmt.Errorf("backward function not implemented for operation type %d", opType)
		}
	}

	return gf
}

// Backward functions for different operations

func (gf *GradientFunction) addBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For addition: grad_a = grad_output, grad_b = grad_output
	// But we need to handle broadcasting
	gradA := gradOutput
	gradB := gradOutput

	// Handle shape differences due to broadcasting
	if len(gf.Inputs) >= 2 {
		if len(gf.Inputs[0].Tensor.Shape) != len(gradOutput.Shape) ||
			len(gf.Inputs[1].Tensor.Shape) != len(gradOutput.Shape) {
			// Need to handle broadcasting - sum over broadcasted dimensions
			var err error
			gradA, err = gf.handleBroadcastBackward(gradOutput, gf.Inputs[0].Tensor.Shape)
			if err != nil {
				return nil, err
			}
			gradB, err = gf.handleBroadcastBackward(gradOutput, gf.Inputs[1].Tensor.Shape)
			if err != nil {
				return nil, err
			}
		}
	}

	return []*tensor.Tensor{gradA, gradB}, nil
}

func (gf *GradientFunction) subBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For subtraction: grad_a = grad_output, grad_b = -grad_output
	gradA := gradOutput

	// Create negative gradient for second input
	gradB, err := ScalarMul(gradOutput, -1.0)
	if err != nil {
		return nil, fmt.Errorf("failed to create negative gradient: %w", err)
	}

	// Handle broadcasting if needed
	if len(gf.Inputs) >= 2 {
		if len(gf.Inputs[0].Tensor.Shape) != len(gradOutput.Shape) {
			gradA, err = gf.handleBroadcastBackward(gradOutput, gf.Inputs[0].Tensor.Shape)
			if err != nil {
				return nil, err
			}
		}
		if len(gf.Inputs[1].Tensor.Shape) != len(gradOutput.Shape) {
			gradB, err = gf.handleBroadcastBackward(gradB, gf.Inputs[1].Tensor.Shape)
			if err != nil {
				return nil, err
			}
		}
	}

	return []*tensor.Tensor{gradA, gradB}, nil
}

func (gf *GradientFunction) mulBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For element-wise multiplication: grad_a = grad_output * b, grad_b = grad_output * a
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for multiplication backward")
	}

	a := gf.SavedTensors[0] // First input
	b := gf.SavedTensors[1] // Second input

	gradA, err := Mul(gradOutput, b)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient for first input: %w", err)
	}

	gradB, err := Mul(gradOutput, a)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient for second input: %w", err)
	}

	// Handle broadcasting if needed
	if len(gf.Inputs) >= 2 {
		if len(gf.Inputs[0].Tensor.Shape) != len(gradA.Shape) {
			gradA, err = gf.handleBroadcastBackward(gradA, gf.Inputs[0].Tensor.Shape)
			if err != nil {
				return nil, err
			}
		}
		if len(gf.Inputs[1].Tensor.Shape) != len(gradB.Shape) {
			gradB, err = gf.handleBroadcastBackward(gradB, gf.Inputs[1].Tensor.Shape)
			if err != nil {
				return nil, err
			}
		}
	}

	return []*tensor.Tensor{gradA, gradB}, nil
}

func (gf *GradientFunction) divBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For division: grad_a = grad_output / b, grad_b = -grad_output * a / (b * b)
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for division backward")
	}

	a := gf.SavedTensors[0] // First input
	b := gf.SavedTensors[1] // Second input

	gradA, err := Div(gradOutput, b)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient for first input: %w", err)
	}

	// grad_b = -grad_output * a / (b * b)
	numerator, err := Mul(gradOutput, a)
	if err != nil {
		return nil, fmt.Errorf("failed to compute numerator for second gradient: %w", err)
	}

	bSquared, err := Mul(b, b)
	if err != nil {
		return nil, fmt.Errorf("failed to compute b squared: %w", err)
	}

	gradB, err := Div(numerator, bSquared)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient for second input: %w", err)
	}

	gradB, err = ScalarMul(gradB, -1.0)
	if err != nil {
		return nil, fmt.Errorf("failed to negate second gradient: %w", err)
	}

	return []*tensor.Tensor{gradA, gradB}, nil
}

func (gf *GradientFunction) matmulBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For matrix multiplication C = A * B:
	// grad_A = grad_output * B^T
	// grad_B = A^T * grad_output
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for matmul backward")
	}

	a := gf.SavedTensors[0] // First input
	b := gf.SavedTensors[1] // Second input

	// Compute grad_A = grad_output * B^T
	bT, err := Transpose(b)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose B: %w", err)
	}
	defer bT.ReleaseGPU()

	gradA, err := MatMul(gradOutput, bT)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient for first input: %w", err)
	}

	// Compute grad_B = A^T * grad_output
	aT, err := Transpose(a)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose A: %w", err)
	}
	defer aT.ReleaseGPU()

	gradB, err := MatMul(aT, gradOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient for second input: %w", err)
	}

	return []*tensor.Tensor{gradA, gradB}, nil
}

func (gf *GradientFunction) transposeBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For transpose: grad_input = grad_output^T
	gradInput, err := Transpose(gradOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose gradient: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) reluBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For ReLU: use saved activation output
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for ReLU backward")
	}

	activationOutput := gf.SavedTensors[0]

	gradInput, err := ReLUBackward(gradOutput, activationOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute ReLU backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) sigmoidBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for Sigmoid backward")
	}

	activationOutput := gf.SavedTensors[0]

	gradInput, err := SigmoidBackward(gradOutput, activationOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Sigmoid backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) tanhBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for Tanh backward")
	}

	activationOutput := gf.SavedTensors[0]

	gradInput, err := TanhBackward(gradOutput, activationOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Tanh backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) softmaxBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 1 {
		return nil, fmt.Errorf("insufficient saved tensors for Softmax backward")
	}

	activationOutput := gf.SavedTensors[0]

	gradInput, err := SoftmaxBackward(gradOutput, activationOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Softmax backward: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) mseLossBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for MSE loss backward")
	}

	predictions := gf.SavedTensors[0]
	targets := gf.SavedTensors[1]

	gradInput, err := MSELossGradients(predictions, targets)
	if err != nil {
		return nil, fmt.Errorf("failed to compute MSE loss gradients: %w", err)
	}

	// Scale by incoming gradient
	if gradOutput != nil {
		gradInput, err = Mul(gradInput, gradOutput)
		if err != nil {
			return nil, fmt.Errorf("failed to scale gradient: %w", err)
		}
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) crossEntropyLossBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gf.SavedTensors) < 2 {
		return nil, fmt.Errorf("insufficient saved tensors for cross-entropy loss backward")
	}

	predictions := gf.SavedTensors[0]
	targets := gf.SavedTensors[1]

	gradInput, err := CategoricalCrossEntropyLossGradients(predictions, targets)
	if err != nil {
		return nil, fmt.Errorf("failed to compute cross-entropy loss gradients: %w", err)
	}

	// Scale by incoming gradient
	if gradOutput != nil {
		gradInput, err = Mul(gradInput, gradOutput)
		if err != nil {
			return nil, fmt.Errorf("failed to scale gradient: %w", err)
		}
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) sumBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For sum reduction: broadcast gradient back to original shape
	if len(gf.Inputs) < 1 {
		return nil, fmt.Errorf("no input tensors for sum backward")
	}

	originalShape := gf.Inputs[0].Tensor.Shape
	gradInput, err := gf.handleBroadcastBackward(gradOutput, originalShape)
	if err != nil {
		return nil, fmt.Errorf("failed to broadcast gradient for sum: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) meanBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For mean reduction: broadcast gradient and scale by 1/N
	if len(gf.Inputs) < 1 {
		return nil, fmt.Errorf("no input tensors for mean backward")
	}

	originalShape := gf.Inputs[0].Tensor.Shape

	// Calculate the number of elements that were averaged
	originalSize := 1
	for _, dim := range originalShape {
		originalSize *= dim
	}

	currentSize := 1
	for _, dim := range gradOutput.Shape {
		currentSize *= dim
	}

	scale := float32(currentSize) / float32(originalSize)

	// Scale gradient
	scaledGrad, err := ScalarMul(gradOutput, scale)
	if err != nil {
		return nil, fmt.Errorf("failed to scale gradient for mean: %w", err)
	}

	// Broadcast back to original shape
	gradInput, err := gf.handleBroadcastBackward(scaledGrad, originalShape)
	if err != nil {
		return nil, fmt.Errorf("failed to broadcast gradient for mean: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

func (gf *GradientFunction) reshapeBackward(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
	// For reshape: reshape gradient back to original shape
	if len(gf.Inputs) < 1 {
		return nil, fmt.Errorf("no input tensors for reshape backward")
	}

	originalShape := gf.Inputs[0].Tensor.Shape

	// Reshape gradient back to original shape
	gradData := make([]float32, len(gradOutput.Data))
	copy(gradData, gradOutput.Data)

	gradInput, err := tensor.NewTensor(originalShape, gradData)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape gradient: %w", err)
	}

	return []*tensor.Tensor{gradInput}, nil
}

// handleBroadcastBackward handles gradient broadcasting for operations that support it
func (gf *GradientFunction) handleBroadcastBackward(grad *tensor.Tensor, targetShape []int) (*tensor.Tensor, error) {
	// If shapes are already the same, no broadcasting needed
	if len(grad.Shape) == len(targetShape) {
		match := true
		for i, dim := range grad.Shape {
			if dim != targetShape[i] {
				match = false
				break
			}
		}
		if match {
			return grad, nil
		}
	}

	// For now, implement a simple case: sum over extra dimensions
	// This handles the most common broadcasting cases

	// If target has fewer dimensions, sum over leading dimensions
	result := grad
	for len(result.Shape) > len(targetShape) {
		var err error
		result, err = SumAlongAxis(result, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to sum along axis: %w", err)
		}
	}

	// If dimensions don't match, sum over dimensions of size 1 in target
	for i := len(result.Shape) - 1; i >= 0; i-- {
		if i < len(targetShape) && targetShape[i] == 1 && result.Shape[i] > 1 {
			var err error
			result, err = SumAlongAxis(result, i)
			if err != nil {
				return nil, fmt.Errorf("failed to sum along axis %d: %w", i, err)
			}
		}
	}

	return result, nil
}

// SumAlongAxis sums a tensor along a specific axis
func SumAlongAxis(input *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	if axis < 0 || axis >= len(input.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with %d dimensions", axis, len(input.Shape))
	}

	// Calculate output shape
	outputShape := make([]int, 0, len(input.Shape)-1)
	for i, dim := range input.Shape {
		if i != axis {
			outputShape = append(outputShape, dim)
		}
	}

	// If result would be scalar, make it 1D with size 1
	if len(outputShape) == 0 {
		outputShape = []int{1}
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	outputData := make([]float32, outputSize)
	output, err := tensor.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Ensure tensors are on GPU
	if err := input.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move input to GPU: %w", err)
	}
	if err := output.EnsureGPU(); err != nil {
		return nil, fmt.Errorf("failed to move output to GPU: %w", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var cErr C.CError
	retCode := C.perform_sum_along_axis(
		C.GPUPtr(input.GPUPtr()),
		C.int(axis),
		C.long(len(input.Shape)),
		(*C.long)(unsafe.Pointer(&input.Shape[0])),
		C.GPUPtr(output.GPUPtr()),
		C.long(len(outputShape)),
		(*C.long)(unsafe.Pointer(&outputShape[0])),
		C.DevicePtr(input.DevicePtr()),
		&cErr,
	)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return nil, fmt.Errorf("GPU sum along axis failed (code %d): %s", retCode, errMsg)
	}

	return output, nil
}

// Helper functions for gradient-aware operations

// GradAdd performs gradient-aware addition
func GradAdd(a, b *GradientTensor) (*GradientTensor, error) {
	// Perform the forward operation
	result, err := Add(a.Tensor, b.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward add failed: %w", err)
	}

	// Create result gradient tensor
	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		IsLeaf:       false,
	}

	// Set up backward function if gradients are required
	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpAdd, []*GradientTensor{a, b}, []*tensor.Tensor{a.Tensor, b.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradSub performs gradient-aware subtraction
func GradSub(a, b *GradientTensor) (*GradientTensor, error) {
	result, err := Sub(a.Tensor, b.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward sub failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpSub, []*GradientTensor{a, b}, []*tensor.Tensor{a.Tensor, b.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradMul performs gradient-aware element-wise multiplication
func GradMul(a, b *GradientTensor) (*GradientTensor, error) {
	result, err := Mul(a.Tensor, b.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward mul failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpMul, []*GradientTensor{a, b}, []*tensor.Tensor{a.Tensor, b.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradDiv performs gradient-aware element-wise division
func GradDiv(a, b *GradientTensor) (*GradientTensor, error) {
	result, err := Div(a.Tensor, b.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward div failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpDiv, []*GradientTensor{a, b}, []*tensor.Tensor{a.Tensor, b.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradMatMul performs gradient-aware matrix multiplication
func GradMatMul(a, b *GradientTensor) (*GradientTensor, error) {
	result, err := MatMul(a.Tensor, b.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward matmul failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpMatMul, []*GradientTensor{a, b}, []*tensor.Tensor{a.Tensor, b.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradTranspose performs gradient-aware transpose
func GradTranspose(a *GradientTensor) (*GradientTensor, error) {
	result, err := Transpose(a.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward transpose failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpTranspose, []*GradientTensor{a}, []*tensor.Tensor{a.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradReLU performs gradient-aware ReLU activation
func GradReLU(a *GradientTensor) (*GradientTensor, error) {
	result, err := ReLUForward(a.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward ReLU failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		// Save the activation output for backward pass
		resultGT.GradFn = CreateBackwardFunction(OpReLU, []*GradientTensor{a}, []*tensor.Tensor{result}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradSigmoid performs gradient-aware Sigmoid activation
func GradSigmoid(a *GradientTensor) (*GradientTensor, error) {
	result, err := SigmoidForward(a.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward Sigmoid failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpSigmoid, []*GradientTensor{a}, []*tensor.Tensor{result}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradTanh performs gradient-aware Tanh activation
func GradTanh(a *GradientTensor) (*GradientTensor, error) {
	result, err := TanhForward(a.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward Tanh failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpTanh, []*GradientTensor{a}, []*tensor.Tensor{result}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradSoftmax performs gradient-aware Softmax activation
func GradSoftmax(a *GradientTensor) (*GradientTensor, error) {
	result, err := SoftmaxForward(a.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward Softmax failed: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpSoftmax, []*GradientTensor{a}, []*tensor.Tensor{result}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradMSELoss performs gradient-aware MSE loss computation
func GradMSELoss(predictions, targets *GradientTensor) (*GradientTensor, error) {
	loss, err := MSELoss(predictions.Tensor, targets.Tensor)
	if err != nil {
		return nil, fmt.Errorf("forward MSE loss failed: %w", err)
	}

	// Create scalar tensor for loss
	lossData := []float32{loss}
	lossTensor, err := tensor.NewTensor([]int{1}, lossData)
	if err != nil {
		return nil, fmt.Errorf("failed to create loss tensor: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       lossTensor,
		RequiresGrad: predictions.RequiresGrad || targets.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpMSELoss, []*GradientTensor{predictions, targets},
			[]*tensor.Tensor{predictions.Tensor, targets.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradSum performs gradient-aware sum reduction
func GradSum(a *GradientTensor) (*GradientTensor, error) {
	// Compute sum of all elements
	sum := float32(0.0)
	if err := a.Tensor.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve tensor data: %w", err)
	}

	for _, val := range a.Tensor.Data {
		sum += val
	}

	// Create scalar result
	resultData := []float32{sum}
	result, err := tensor.NewTensor([]int{1}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create sum tensor: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpSum, []*GradientTensor{a}, []*tensor.Tensor{a.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradMean performs gradient-aware mean reduction
func GradMean(a *GradientTensor) (*GradientTensor, error) {
	// Compute mean of all elements
	sum := float32(0.0)
	if err := a.Tensor.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve tensor data: %w", err)
	}

	for _, val := range a.Tensor.Data {
		sum += val
	}

	mean := sum / float32(len(a.Tensor.Data))

	// Create scalar result
	resultData := []float32{mean}
	result, err := tensor.NewTensor([]int{1}, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create mean tensor: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpMean, []*GradientTensor{a}, []*tensor.Tensor{a.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// GradReshape performs gradient-aware tensor reshaping
func GradReshape(a *GradientTensor, newShape []int) (*GradientTensor, error) {
	// Verify that the total number of elements matches
	oldSize := 1
	for _, dim := range a.Tensor.Shape {
		oldSize *= dim
	}

	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if oldSize != newSize {
		return nil, fmt.Errorf("cannot reshape tensor from size %d to size %d", oldSize, newSize)
	}

	// Create new tensor with same data but different shape
	newData := make([]float32, len(a.Tensor.Data))
	copy(newData, a.Tensor.Data)

	result, err := tensor.NewTensor(newShape, newData)
	if err != nil {
		return nil, fmt.Errorf("failed to create reshaped tensor: %w", err)
	}

	resultGT := &GradientTensor{
		Tensor:       result,
		RequiresGrad: a.RequiresGrad,
		IsLeaf:       false,
	}

	if resultGT.RequiresGrad && globalGraph.gradMode == Grad {
		resultGT.GradFn = CreateBackwardFunction(OpReshape, []*GradientTensor{a}, []*tensor.Tensor{a.Tensor}, nil)
		globalGraph.addNode(resultGT)
	}

	return resultGT, nil
}

// DetachGradient creates a new GradientTensor that doesn't require gradients
func DetachGradient(a *GradientTensor) *GradientTensor {
	return &GradientTensor{
		Tensor:       a.Tensor,
		RequiresGrad: false,
		IsLeaf:       true,
	}
}

// GradientCheckpoint implements gradient checkpointing to save memory
func GradientCheckpoint(fn func(*GradientTensor) (*GradientTensor, error), input *GradientTensor) (*GradientTensor, error) {
	// Disable gradients during forward pass
	oldMode := globalGraph.gradMode
	globalGraph.gradMode = NoGrad

	// Run forward pass without tracking gradients
	output, err := fn(input)
	if err != nil {
		globalGraph.gradMode = oldMode
		return nil, err
	}

	// Restore gradient mode
	globalGraph.gradMode = oldMode

	if !input.RequiresGrad || globalGraph.gradMode == NoGrad {
		return output, nil
	}

	// Create a special gradient function that will recompute the forward pass during backward
	output.RequiresGrad = true
	output.IsLeaf = false
	output.GradFn = &GradientFunction{
		OpType: OpReshape, // Reuse reshape op type as placeholder
		Inputs: []*GradientTensor{input},
		BackwardFn: func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error) {
			// Recompute forward pass with gradients enabled
			oldMode := globalGraph.gradMode
			globalGraph.gradMode = Grad

			recomputedOutput, err := fn(input)
			if err != nil {
				globalGraph.gradMode = oldMode
				return nil, err
			}

			globalGraph.gradMode = oldMode

			// Now compute backward pass
			err = recomputedOutput.backwardRecursive(gradOutput)
			if err != nil {
				return nil, err
			}

			return []*tensor.Tensor{input.Gradient}, nil
		},
	}

	globalGraph.addNode(output)
	return output, nil
}
