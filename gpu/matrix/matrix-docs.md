# matrix
--
    import "."


## Usage

#### func  AccumulateGradient

```go
func AccumulateGradient(existing, newGrad *tensor.Tensor) error
```
AccumulateGradient accumulates gradients using GPU acceleration

#### func  ActivationBackward

```go
func ActivationBackward(gradOutput, activationOutput *tensor.Tensor, activationType ActivationType, alpha float32) (*tensor.Tensor, error)
```
ActivationBackward computes the gradient of an activation function

#### func  ActivationForward

```go
func ActivationForward(input *tensor.Tensor, activationType ActivationType, alpha float32) (*tensor.Tensor, error)
```
ActivationForward applies an activation function to the input tensor

#### func  Add

```go
func Add(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
Add performs element-wise matrix addition C = A + B on the GPU

#### func  AvgPool2D

```go
func AvgPool2D(input *tensor.Tensor, poolSize, stride, padding int) (*tensor.Tensor, error)
```
AvgPool2D performs 2D average pooling with default parameters

#### func  AvgPool2DBackward

```go
func AvgPool2DBackward(gradOutput *tensor.Tensor, inputShape []int, params Pool2DParams) (*tensor.Tensor, error)
```
AvgPool2DBackward performs 2D average pooling backward pass

#### func  AvgPool2DForward

```go
func AvgPool2DForward(input *tensor.Tensor, params Pool2DParams) (*tensor.Tensor, error)
```
AvgPool2DForward performs 2D average pooling forward pass

#### func  BatchActivationBackward

```go
func BatchActivationBackward(gradOutputs, activationOutputs []*tensor.Tensor, activationType ActivationType, alpha float32) ([]*tensor.Tensor, error)
```
BatchActivationBackward computes gradients for multiple tensors

#### func  BatchActivationForward

```go
func BatchActivationForward(inputs []*tensor.Tensor, activationType ActivationType, alpha float32) ([]*tensor.Tensor, error)
```
BatchActivationForward applies activation function to multiple tensors

#### func  BatchGPUAdd

```go
func BatchGPUAdd(operations []struct{ A, B mat.Matrix }) []*mat.Dense
```
BatchGPUAdd performs multiple element-wise additions efficiently

#### func  BatchGPUBatchNorm

```go
func BatchGPUBatchNorm(inputs []mat.Matrix, gamma, beta, runningMean, runningVar mat.Matrix, epsilon float32, training bool) []*mat.Dense
```
BatchGPUBatchNorm applies batch normalization to multiple input matrices

#### func  BatchGPUCholesky

```go
func BatchGPUCholesky(matrices []mat.Matrix) []*mat.Dense
```
BatchGPUCholesky performs multiple Cholesky decompositions efficiently

#### func  BatchGPUInverse

```go
func BatchGPUInverse(matrices []mat.Matrix) []*mat.Dense
```
BatchGPUInverse performs multiple matrix inversions efficiently

#### func  BatchGPULayerNorm

```go
func BatchGPULayerNorm(inputs []mat.Matrix, gamma, beta mat.Matrix, epsilon float32) []*mat.Dense
```
BatchGPULayerNorm applies layer normalization to multiple input matrices

#### func  BatchGPUMatMul

```go
func BatchGPUMatMul(operations []struct{ A, B mat.Matrix }) []*mat.Dense
```
BatchGPUMatMul keeps matrices on GPU for multiple operations

#### func  BatchGPUSparseMatMul

```go
func BatchGPUSparseMatMul(operations []struct{ A, B interface{} }) []*mat.Dense
```
BatchGPUSparseMatMul performs multiple sparse matrix multiplications efficiently

#### func  BatchGPUSparseMatVec

```go
func BatchGPUSparseMatVec(matrices []*GPUSparse, vectors [][]float64) [][]float64
```
BatchGPUSparseMatVec performs multiple sparse matrix-vector multiplications

#### func  BatchMean

```go
func BatchMean(input *tensor.Tensor) (*tensor.Tensor, error)
```
BatchMean computes the mean across the batch dimension

#### func  BatchNormForward

```go
func BatchNormForward(input, mean, variance, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, error)
```
BatchNormForward performs batch normalization forward pass

#### func  BatchNormInference

```go
func BatchNormInference(input, runningMean, runningVar, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, error)
```
BatchNormInference performs batch normalization in inference mode using running
statistics

#### func  BatchVariance

```go
func BatchVariance(input, mean *tensor.Tensor) (*tensor.Tensor, error)
```
BatchVariance computes the variance across the batch dimension

#### func  BinaryCrossEntropyLoss

```go
func BinaryCrossEntropyLoss(predictions, targets *tensor.Tensor) (float32, error)
```
BinaryCrossEntropyLoss computes Binary Cross-Entropy loss

#### func  BinaryCrossEntropyLossGradients

```go
func BinaryCrossEntropyLossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error)
```
BinaryCrossEntropyLossGradients computes Binary Cross-Entropy gradients

#### func  CalculateConv2DOutputSize

```go
func CalculateConv2DOutputSize(inputH, inputW, kernelH, kernelW, strideH, strideW, padH, padW int) (int, int)
```
CalculateConv2DOutputSize calculates the output dimensions for 2D convolution

#### func  CalculateGradientNorm

```go
func CalculateGradientNorm(gradTensors []*GradientTensor) (float32, error)
```
CalculateGradientNorm calculates the L2 norm of gradients

#### func  CalculateL2Norm

```go
func CalculateL2Norm(tensors []*tensor.Tensor) (float32, error)
```
CalculateL2Norm calculates the L2 norm of multiple tensors

#### func  CalculatePool2DOutputSize

```go
func CalculatePool2DOutputSize(inputH, inputW, poolH, poolW, strideH, strideW, padH, padW int) (int, int)
```
CalculatePool2DOutputSize calculates the output dimensions for 2D pooling

#### func  CategoricalCrossEntropyLoss

```go
func CategoricalCrossEntropyLoss(predictions, targets *tensor.Tensor) (float32, error)
```
CategoricalCrossEntropyLoss computes Categorical Cross-Entropy loss

#### func  CategoricalCrossEntropyLossGradients

```go
func CategoricalCrossEntropyLossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error)
```
CategoricalCrossEntropyLossGradients computes Categorical Cross-Entropy
gradients

#### func  Cholesky

```go
func Cholesky(A *tensor.Tensor) (*tensor.Tensor, error)
```
Cholesky performs Cholesky decomposition using the Accelerate framework Returns
the lower triangular matrix L such that A = L * L^T

#### func  ClearGraph

```go
func ClearGraph()
```
ClearGraph clears the computation graph

#### func  ClipGradientNorm

```go
func ClipGradientNorm(gradTensors []*GradientTensor, maxNorm float32) error
```
ClipGradientNorm clips gradients by norm to prevent exploding gradients

#### func  Col2Im

```go
func Col2Im(input *tensor.Tensor, outputShape []int, kernelH, kernelW, strideH, strideW, padH, padW int) (*tensor.Tensor, error)
```
Col2Im performs the col2im operation (inverse of im2col)

#### func  Conv2DBackwardInput

```go
func Conv2DBackwardInput(gradOutput, kernel *tensor.Tensor, inputShape []int, params Conv2DParams) (*tensor.Tensor, error)
```
Conv2DBackwardInput computes gradients with respect to input

#### func  Conv2DBackwardKernel

```go
func Conv2DBackwardKernel(input, gradOutput *tensor.Tensor, kernelShape []int, params Conv2DParams) (*tensor.Tensor, error)
```
Conv2DBackwardKernel computes gradients with respect to kernel

#### func  CreateBatchNormParams

```go
func CreateBatchNormParams(features int) (gamma, beta *mat.Dense)
```
CreateBatchNormParams creates initialized gamma and beta parameters for batch
normalization

#### func  CreateLayerNormParams

```go
func CreateLayerNormParams(features int) (gamma, beta *mat.Dense)
```
CreateLayerNormParams creates initialized gamma and beta parameters for layer
normalization

#### func  DenseToSparse

```go
func DenseToSparse(A *tensor.Tensor, threshold float32) (*tensor.SparseTensor, error)
```
DenseToSparse converts a dense matrix to sparse format using GPU acceleration

#### func  Determinant

```go
func Determinant(A *tensor.Tensor) (float32, error)
```
Determinant computes the matrix determinant using the Accelerate framework

#### func  Div

```go
func Div(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
Div performs element-wise matrix division C = A ⊘ B on the GPU

#### func  DropoutBackward

```go
func DropoutBackward(gradOutput, mask *tensor.Tensor, probability float32) (*tensor.Tensor, error)
```
DropoutBackward applies dropout mask during backward pass

#### func  DropoutForward

```go
func DropoutForward(input, output, mask *tensor.Tensor, probability float32, seed uint32) error
```
DropoutForward applies dropout during forward pass

#### func  ELUBackward

```go
func ELUBackward(gradOutput, activationOutput *tensor.Tensor, alpha float32) (*tensor.Tensor, error)
```
ELUBackward computes ELU gradient

#### func  ELUForward

```go
func ELUForward(input *tensor.Tensor, alpha float32) (*tensor.Tensor, error)
```
ELUForward applies ELU (Exponential Linear Unit) activation function

#### func  Flatten4DTo2D

```go
func Flatten4DTo2D(t *tensor.Tensor) (*tensor.Tensor, error)
```
Flatten4DTo2D flattens a 4D tensor to 2D for compatibility with Gonum operations

#### func  GELUBackward

```go
func GELUBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error)
```
GELUBackward computes GELU gradient

#### func  GELUForward

```go
func GELUForward(input *tensor.Tensor) (*tensor.Tensor, error)
```
GELUForward applies GELU (Gaussian Error Linear Unit) activation function

#### func  GPUAdd

```go
func GPUAdd(a, b mat.Matrix) *mat.Dense
```
GPUAdd is a drop-in replacement for gonum's matrix addition

#### func  GPUBatchNormalize

```go
func GPUBatchNormalize(input mat.Matrix, gamma, beta, runningMean, runningVar mat.Matrix, epsilon float32, training bool) *mat.Dense
```
GPUBatchNormalize applies batch normalization to a Gonum matrix

#### func  GPUCholesky

```go
func GPUCholesky(a mat.Matrix) *mat.Dense
```
GPUCholesky is a drop-in replacement for gonum's Cholesky decomposition

#### func  GPUConv2DSimple

```go
func GPUConv2DSimple(input *mat.Dense, kernelSize, stride, padding int, numFilters int) (*mat.Dense, error)
```
GPUConv2DSimple performs a simple 2D convolution with square kernels

#### func  GPUDeterminant

```go
func GPUDeterminant(a mat.Matrix) float64
```
GPUDeterminant is a drop-in replacement for gonum's matrix determinant

#### func  GPUDivElem

```go
func GPUDivElem(a, b mat.Matrix) *mat.Dense
```
GPUDivElem is a drop-in replacement for gonum's element-wise division

#### func  GPUInverse

```go
func GPUInverse(a mat.Matrix) *mat.Dense
```
GPUInverse is a drop-in replacement for gonum's matrix inverse

#### func  GPULayerNormalize

```go
func GPULayerNormalize(input, gamma, beta mat.Matrix, epsilon float32) *mat.Dense
```
GPULayerNormalize applies layer normalization to a Gonum matrix

#### func  GPUMatMul

```go
func GPUMatMul(a, b mat.Matrix) *mat.Dense
```
GPUMatMul is a drop-in replacement for gonum's matrix multiplication

#### func  GPUMaxPool2DSimple

```go
func GPUMaxPool2DSimple(input *mat.Dense, poolSize, stride int) (*mat.Dense, error)
```
GPUMaxPool2DSimple performs simple 2D max pooling

#### func  GPUMulElem

```go
func GPUMulElem(a, b mat.Matrix) *mat.Dense
```
GPUMulElem is a drop-in replacement for gonum's element-wise multiplication

#### func  GPUSparseMatMul

```go
func GPUSparseMatMul(a, b interface{}) *mat.Dense
```
GPUSparseMatMul is a drop-in replacement for matrix multiplication with sparse
matrices

#### func  GPUSparseToDense

```go
func GPUSparseToDense(gs *GPUSparse) *mat.Dense
```
GPUSparseToDense converts a sparse matrix to dense format

#### func  GPUSub

```go
func GPUSub(a, b mat.Matrix) *mat.Dense
```
GPUSub is a drop-in replacement for gonum's matrix subtraction

#### func  GetGradientStats

```go
func GetGradientStats(gradTensors []*GradientTensor) (map[string]float32, error)
```
GetGradientStats returns statistics about gradients for debugging

#### func  GonumToTensor

```go
func GonumToTensor(m *mat.Dense) (*tensor.Tensor, error)
```
GonumToTensor converts a Gonum Dense matrix to a 2D tensor

#### func  GradientScaleAndClip

```go
func GradientScaleAndClip(gradTensors []*GradientTensor, scale float32, maxNorm float32) error
```
GradientScaleAndClip combines gradient scaling and clipping in one operation

#### func  GroupNormForward

```go
func GroupNormForward(input, gamma, beta *tensor.Tensor, numGroups int, epsilon float32) (*tensor.Tensor, error)
```
GroupNormForward performs group normalization forward pass

#### func  HingeLoss

```go
func HingeLoss(predictions, targets *tensor.Tensor) (float32, error)
```
HingeLoss computes Hinge loss

#### func  HingeLossGradients

```go
func HingeLossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error)
```
HingeLossGradients computes Hinge loss gradients

#### func  HuberLoss

```go
func HuberLoss(predictions, targets *tensor.Tensor, delta float32) (float32, error)
```
HuberLoss computes Huber loss with specified delta

#### func  HuberLossGradients

```go
func HuberLossGradients(predictions, targets *tensor.Tensor, delta float32) (*tensor.Tensor, error)
```
HuberLossGradients computes Huber loss gradients

#### func  Im2Col

```go
func Im2Col(input *tensor.Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) (*tensor.Tensor, error)
```
Im2Col performs the im2col operation for efficient convolution implementation

#### func  InitializeRunningStats

```go
func InitializeRunningStats(features int) (runningMean, runningVar *mat.Dense)
```
InitializeRunningStats creates initialized running mean and variance for batch
normalization

#### func  InstanceNormForward

```go
func InstanceNormForward(input, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, error)
```
InstanceNormForward performs instance normalization forward pass

#### func  Inverse

```go
func Inverse(A *tensor.Tensor) (*tensor.Tensor, error)
```
Inverse computes the matrix inverse using the Accelerate framework

#### func  IsSparseWorthwhile

```go
func IsSparseWorthwhile(rows, cols, nnz int) bool
```
IsSparseWorthwhile determines if using sparse format is beneficial

#### func  LayerNormForwardBackward

```go
func LayerNormForwardBackward(input, gamma, beta, gradOutput *tensor.Tensor, epsilon float32) (*BatchNormResult, *BatchNormGradients, error)
```
LayerNormForwardBackward performs both forward and backward passes for layer
normalization

#### func  LeakyReLUBackward

```go
func LeakyReLUBackward(gradOutput, activationOutput *tensor.Tensor, alpha float32) (*tensor.Tensor, error)
```
LeakyReLUBackward computes Leaky ReLU gradient

#### func  LeakyReLUForward

```go
func LeakyReLUForward(input *tensor.Tensor, alpha float32) (*tensor.Tensor, error)
```
LeakyReLUForward applies Leaky ReLU activation function

#### func  LossBackward

```go
func LossBackward(predictions, targets *tensor.Tensor, lossType LossType, params ...float32) (*tensor.Tensor, error)
```
LossBackward computes the backward pass (gradients) of a loss function

#### func  LossForward

```go
func LossForward(predictions, targets *tensor.Tensor, lossType LossType, params ...float32) (float32, error)
```
LossForward computes the forward pass of a loss function

#### func  MAELoss

```go
func MAELoss(predictions, targets *tensor.Tensor) (float32, error)
```
MAELoss computes Mean Absolute Error loss

#### func  MAELossGradients

```go
func MAELossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error)
```
MAELossGradients computes MAE gradients

#### func  MSELoss

```go
func MSELoss(predictions, targets *tensor.Tensor) (float32, error)
```
MSELoss computes Mean Squared Error loss

#### func  MSELossGradients

```go
func MSELossGradients(predictions, targets *tensor.Tensor) (*tensor.Tensor, error)
```
MSELossGradients computes MSE gradients

#### func  MatMul

```go
func MatMul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
MatMul performs matrix multiplication C = A * B on the GPU

#### func  MaxPool2DBackward

```go
func MaxPool2DBackward(gradOutput, indices *tensor.Tensor, inputShape []int, params Pool2DParams) (*tensor.Tensor, error)
```
MaxPool2DBackward performs 2D max pooling backward pass

#### func  Mul

```go
func Mul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
Mul performs element-wise matrix multiplication (Hadamard product) C = A ⊙ B on
the GPU

#### func  NoGradContext

```go
func NoGradContext(fn func())
```
NoGradContext temporarily disables gradient computation

#### func  Pad2D

```go
func Pad2D(input *tensor.Tensor, padTop, padBottom, padLeft, padRight int, padValue float32) (*tensor.Tensor, error)
```
Pad2D adds padding to a 2D tensor

#### func  PrintConvParams

```go
func PrintConvParams(params Conv2DParams)
```
PrintConvParams prints convolution parameters

#### func  PrintPoolParams

```go
func PrintPoolParams(params Pool2DParams)
```
PrintPoolParams prints pooling parameters

#### func  PrintTensorInfo

```go
func PrintTensorInfo(name string, t *tensor.Tensor)
```
PrintTensorInfo prints information about a tensor's shape and GPU status

#### func  ReLUBackward

```go
func ReLUBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error)
```
ReLUBackward computes ReLU gradient

#### func  ReLUForward

```go
func ReLUForward(input *tensor.Tensor) (*tensor.Tensor, error)
```
ReLUForward applies ReLU activation function

#### func  ReflectionPad2D

```go
func ReflectionPad2D(input *tensor.Tensor, padding int) (*tensor.Tensor, error)
```
ReflectionPad2D adds reflection padding to a tensor

#### func  Reshape2DTo4D

```go
func Reshape2DTo4D(t *tensor.Tensor, targetShape []int) (*tensor.Tensor, error)
```
Reshape2DTo4D reshapes a 2D tensor back to 4D

#### func  SaveForBackward

```go
func SaveForBackward(tensors ...*tensor.Tensor) []*tensor.Tensor
```
SaveForBackward saves tensors that will be needed during the backward pass

#### func  ScalarAdd

```go
func ScalarAdd(A *tensor.Tensor, scalar float32) (*tensor.Tensor, error)
```
ScalarAdd performs scalar addition C = A + scalar on the GPU

#### func  ScalarMul

```go
func ScalarMul(A *tensor.Tensor, scalar float32) (*tensor.Tensor, error)
```
ScalarMul performs scalar multiplication C = A * scalar on the GPU

#### func  SetGradientMode

```go
func SetGradientMode(mode GradientMode)
```
SetGradientMode sets the global gradient computation mode

#### func  SigmoidBackward

```go
func SigmoidBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error)
```
SigmoidBackward computes Sigmoid gradient

#### func  SigmoidForward

```go
func SigmoidForward(input *tensor.Tensor) (*tensor.Tensor, error)
```
SigmoidForward applies Sigmoid activation function

#### func  SoftmaxBackward

```go
func SoftmaxBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error)
```
SoftmaxBackward computes Softmax gradient

#### func  SoftmaxForward

```go
func SoftmaxForward(input *tensor.Tensor) (*tensor.Tensor, error)
```
SoftmaxForward applies Softmax activation function

#### func  SparseAdd

```go
func SparseAdd(A, B *tensor.SparseTensor) (*tensor.SparseTensor, error)
```
SparseAdd performs element-wise addition of sparse matrices

#### func  SparseCategoricalCrossEntropyBackward

```go
func SparseCategoricalCrossEntropyBackward(predictions *tensor.Tensor, targetIndices []int) (*tensor.Tensor, error)
```
SparseCategoricalCrossEntropyBackward computes gradients for sparse categorical
cross-entropy

#### func  SparseCategoricalCrossEntropyForward

```go
func SparseCategoricalCrossEntropyForward(predictions *tensor.Tensor, targetIndices []int) (float32, error)
```
SparseCategoricalCrossEntropyForward computes sparse categorical cross-entropy
with integer targets

#### func  SparseMatMul

```go
func SparseMatMul(A, B interface{}) (*tensor.Tensor, error)
```
SparseMatMul performs sparse matrix multiplication C = A * B Supports
sparse-dense, dense-sparse, and sparse-sparse multiplication

#### func  SparseMatVec

```go
func SparseMatVec(A *tensor.SparseTensor, x *tensor.Tensor) (*tensor.Tensor, error)
```
SparseMatVec performs sparse matrix-vector multiplication

#### func  SparseScalarMul

```go
func SparseScalarMul(A *tensor.SparseTensor, scalar float32) (*tensor.SparseTensor, error)
```
SparseScalarMul performs scalar multiplication on a sparse matrix

#### func  SparseToDense

```go
func SparseToDense(A *tensor.SparseTensor) (*tensor.Tensor, error)
```
SparseToDense converts a sparse matrix to dense format using GPU acceleration

#### func  SparseTranspose

```go
func SparseTranspose(A *tensor.SparseTensor) (*tensor.SparseTensor, error)
```
SparseTranspose computes the transpose of a sparse matrix

#### func  Sub

```go
func Sub(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
Sub performs element-wise matrix subtraction C = A - B on the GPU

#### func  SumAlongAxis

```go
func SumAlongAxis(input *tensor.Tensor, axis int) (*tensor.Tensor, error)
```
SumAlongAxis sums a tensor along a specific axis

#### func  SwishBackward

```go
func SwishBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error)
```
SwishBackward computes Swish gradient

#### func  SwishForward

```go
func SwishForward(input *tensor.Tensor) (*tensor.Tensor, error)
```
SwishForward applies Swish activation function (x * sigmoid(x))

#### func  TanhBackward

```go
func TanhBackward(gradOutput, activationOutput *tensor.Tensor) (*tensor.Tensor, error)
```
TanhBackward computes Tanh gradient

#### func  TanhForward

```go
func TanhForward(input *tensor.Tensor) (*tensor.Tensor, error)
```
TanhForward applies Tanh activation function

#### func  TensorToGonum

```go
func TensorToGonum(t *tensor.Tensor) (*mat.Dense, error)
```
TensorToGonum converts a 2D tensor to a Gonum Dense matrix

#### func  Transpose

```go
func Transpose(A *tensor.Tensor) (*tensor.Tensor, error)
```
Transpose performs matrix transpose on the GPU

#### func  Unpad2D

```go
func Unpad2D(input *tensor.Tensor, padTop, padBottom, padLeft, padRight int) (*tensor.Tensor, error)
```
Unpad2D removes padding from a 2D tensor (crop operation)

#### func  UpdateCreateBackwardFunction

```go
func UpdateCreateBackwardFunction()
```
Update the CreateBackwardFunction to include the new operations

#### func  UpdateRunningStats

```go
func UpdateRunningStats(runningMean, runningVar, batchMean, batchVar *tensor.Tensor, momentum float32) error
```
UpdateRunningStats updates running mean and variance for batch normalization

#### func  ZeroGrad

```go
func ZeroGrad()
```
ZeroGrad zeros all gradients in the computation graph

#### func  ZeroPad2D

```go
func ZeroPad2D(input *tensor.Tensor, padding int) (*tensor.Tensor, error)
```
ZeroPad2D adds zero padding to a tensor

#### func  ZeroTensorGPU

```go
func ZeroTensorGPU(t *tensor.Tensor) error
```

#### type ActivationType

```go
type ActivationType int
```

ActivationType represents different activation function types

```go
const (
	ReLU ActivationType = iota
	Sigmoid
	Tanh
	Softmax
	LeakyReLU
	ELU
	Swish
	GELU
)
```

#### func (ActivationType) String

```go
func (at ActivationType) String() string
```
String returns string representation of activation type

#### type BatchNormGradients

```go
type BatchNormGradients struct {
	GradInput *tensor.Tensor // Gradients w.r.t. input
	GradGamma *tensor.Tensor // Gradients w.r.t. gamma (scale parameter)
	GradBeta  *tensor.Tensor // Gradients w.r.t. beta (shift parameter)
}
```

BatchNormGradients contains gradients from batch normalization backward pass

#### func  BatchNormBackward

```go
func BatchNormBackward(gradOutput, input, mean, variance, gamma *tensor.Tensor, epsilon float32) (*BatchNormGradients, error)
```
BatchNormBackward performs batch normalization backward pass

#### func  BatchNormForwardBackward

```go
func BatchNormForwardBackward(input, mean, variance, gamma, beta, gradOutput *tensor.Tensor, epsilon float32) (*tensor.Tensor, *BatchNormGradients, error)
```
BatchNormForwardBackward performs both forward and backward passes efficiently

#### func  LayerNormBackward

```go
func LayerNormBackward(gradOutput, input, mean, variance, gamma *tensor.Tensor, epsilon float32) (*BatchNormGradients, error)
```
LayerNormBackward performs layer normalization backward pass

#### func (*BatchNormGradients) ReleaseGPU

```go
func (bng *BatchNormGradients) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the batch norm gradients

#### type BatchNormResult

```go
type BatchNormResult struct {
	Output   *tensor.Tensor // Normalized output
	Mean     *tensor.Tensor // Computed mean (for backward pass)
	Variance *tensor.Tensor // Computed variance (for backward pass)
}
```

BatchNormResult contains the result of batch normalization and auxiliary data

#### func  BatchNormTraining

```go
func BatchNormTraining(input, gamma, beta *tensor.Tensor, epsilon float32) (*BatchNormResult, error)
```
BatchNormTraining performs batch normalization in training mode, computing batch
statistics

#### func  LayerNormForward

```go
func LayerNormForward(input, gamma, beta *tensor.Tensor, epsilon float32) (*BatchNormResult, error)
```
LayerNormForward performs layer normalization forward pass

#### func (*BatchNormResult) ReleaseGPU

```go
func (bnr *BatchNormResult) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the batch norm result

#### type BatchNormType

```go
type BatchNormType int
```

BatchNormType represents different normalization types

```go
const (
	BatchNorm BatchNormType = iota
	LayerNorm
	InstanceNorm
	GroupNorm
)
```

#### func (BatchNormType) String

```go
func (bnt BatchNormType) String() string
```
String returns string representation of batch norm type

#### type CachedBatch

```go
type CachedBatch struct {
}
```

CachedBatch represents a cached batch of data

#### type CachedTensor

```go
type CachedTensor struct {
}
```

CachedTensor represents a cached tensor with metadata

#### type Checkpoint

```go
type Checkpoint struct {
	// Metadata
	Version        string    `json:"version"`
	Timestamp      time.Time `json:"timestamp"`
	Epoch          int       `json:"epoch"`
	Step           int64     `json:"step"`
	Loss           float32   `json:"loss"`
	ValidationLoss float32   `json:"validation_loss"`
	LearningRate   float32   `json:"learning_rate"`

	// Model info
	ModelName   string                 `json:"model_name"`
	ModelConfig map[string]interface{} `json:"model_config"`

	// Training config
	TrainingConfig *TrainingConfig `json:"training_config"`

	// File paths
	ModelPath     string `json:"model_path"`
	OptimizerPath string `json:"optimizer_path"`
	SchedulerPath string `json:"scheduler_path"`
	MetricsPath   string `json:"metrics_path"`

	// Checksums for verification
	ModelChecksum     string `json:"model_checksum"`
	OptimizerChecksum string `json:"optimizer_checksum"`
	SchedulerChecksum string `json:"scheduler_checksum"`

	// Additional metadata
	Platform        string `json:"platform"`
	GPUMemoryUsage  int64  `json:"gpu_memory_usage"`
	TotalParams     int64  `json:"total_params"`
	TrainableParams int64  `json:"trainable_params"`
}
```

Checkpoint represents a training checkpoint

#### type CheckpointConfig

```go
type CheckpointConfig struct {
	BaseDir        string
	MaxCheckpoints int
	SaveOptimizer  bool
	SaveScheduler  bool
	Compression    bool
	ChecksumVerify bool
	SaveInterval   int  // Save every N epochs
	SaveBest       bool // Save best validation loss
	SaveLast       bool // Always save last checkpoint
}
```

CheckpointConfig contains configuration for checkpoint management

#### type CheckpointManager

```go
type CheckpointManager struct {
}
```

CheckpointManager handles saving and loading training checkpoints

#### func  NewCheckpointManager

```go
func NewCheckpointManager(config CheckpointConfig) (*CheckpointManager, error)
```
NewCheckpointManager creates a new checkpoint manager

#### func (*CheckpointManager) ExportCheckpoint

```go
func (cm *CheckpointManager) ExportCheckpoint(checkpointPath, exportPath string) error
```
ExportCheckpoint exports a checkpoint to a portable format

#### func (*CheckpointManager) GetBestCheckpoint

```go
func (cm *CheckpointManager) GetBestCheckpoint() (*Checkpoint, error)
```
GetBestCheckpoint returns the checkpoint with the best validation loss

#### func (*CheckpointManager) GetLatestCheckpoint

```go
func (cm *CheckpointManager) GetLatestCheckpoint() (*Checkpoint, error)
```
GetLatestCheckpoint returns the most recent checkpoint

#### func (*CheckpointManager) ImportCheckpoint

```go
func (cm *CheckpointManager) ImportCheckpoint(importPath, checkpointName string) error
```
ImportCheckpoint imports a checkpoint from a portable format

#### func (*CheckpointManager) ListCheckpoints

```go
func (cm *CheckpointManager) ListCheckpoints() ([]*Checkpoint, error)
```
ListCheckpoints returns a list of available checkpoints

#### func (*CheckpointManager) LoadCheckpoint

```go
func (cm *CheckpointManager) LoadCheckpoint(checkpointPath string, trainer *Trainer, model TrainableModel) (*Checkpoint, error)
```
LoadCheckpoint loads a training checkpoint

#### func (*CheckpointManager) SaveCheckpoint

```go
func (cm *CheckpointManager) SaveCheckpoint(trainer *Trainer, model TrainableModel, name string) (*Checkpoint, error)
```
SaveCheckpoint saves a training checkpoint

#### type ComputationGraph

```go
type ComputationGraph struct {
}
```

ComputationGraph tracks the computational graph for backpropagation

#### func  NewComputationGraph

```go
func NewComputationGraph() *ComputationGraph
```
NewComputationGraph creates a new computation graph

#### type Conv2DParams

```go
type Conv2DParams struct {
	StrideH int // Stride in height dimension
	StrideW int // Stride in width dimension
	PadH    int // Padding in height dimension
	PadW    int // Padding in width dimension
}
```

Conv2DParams represents parameters for 2D convolution

#### type Conv2DResult

```go
type Conv2DResult struct {
	Output *tensor.Tensor
}
```

Conv2DResult contains the result of a convolution operation and any auxiliary
data

#### func  BatchGPUConv2D

```go
func BatchGPUConv2D(inputs []*tensor.Tensor, kernel *tensor.Tensor, params Conv2DParams) ([]*Conv2DResult, error)
```
BatchGPUConv2D performs convolution on multiple input tensors

#### func  Conv2D

```go
func Conv2D(input, kernel *tensor.Tensor, stride, padding int) (*Conv2DResult, error)
```
Conv2D performs a simple 2D convolution with default parameters

#### func  Conv2DForward

```go
func Conv2DForward(input, kernel *tensor.Tensor, params Conv2DParams) (*Conv2DResult, error)
```
Conv2DForward performs 2D convolution forward pass Input tensor shape: [batch,
height, width, channels] Kernel tensor shape: [kernel_height, kernel_width,
input_channels, output_channels] Output tensor shape: [batch, output_height,
output_width, output_channels]

#### func (*Conv2DResult) ReleaseGPU

```go
func (cr *Conv2DResult) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the convolution result

#### type DataLoader

```go
type DataLoader interface {
	GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
	BatchCount() int
	Shuffle() error
	Reset() error
	SetBatchSize(batchSize int)
	GetDatasetSize() int
}
```

DataLoader interface for efficient data loading

#### type DataLoaderConfig

```go
type DataLoaderConfig struct {
	BatchSize      int
	Shuffle        bool
	DropLast       bool
	NumWorkers     int
	PrefetchFactor int
	CacheSize      int
	MaxMemoryUsage int64
	MemoryPool     *GPUMemoryPool
	TensorCache    *TensorCache
}
```

DataLoaderConfig contains configuration for data loader

#### type DataLoaderStats

```go
type DataLoaderStats struct {
	TotalBatches      int64
	CacheHits         int64
	CacheMisses       int64
	LoadTime          time.Duration
	AverageLoadTime   time.Duration
	PrefetchHits      int64
	MemoryUsage       int64
	WorkerUtilization []float32
}
```

DataLoaderStats tracks data loading performance

#### type DataWorker

```go
type DataWorker struct {
}
```

DataWorker handles async data loading

#### type Dataset

```go
type Dataset interface {
	GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error)
	Len() int
	GetShape() ([]int, []int) // input shape, target shape
}
```

Dataset interface for data sources

#### type EigenDecomposition

```go
type EigenDecomposition struct {
	Eigenvalues  *tensor.Tensor // Vector of eigenvalues
	Eigenvectors *tensor.Tensor // Matrix of eigenvectors (column-wise)
}
```

EigenDecomposition represents the result of eigenvalue decomposition

#### func  Eigen

```go
func Eigen(A *tensor.Tensor) (*EigenDecomposition, error)
```
Eigen performs eigenvalue decomposition for symmetric matrices using the
Accelerate framework

#### func (*EigenDecomposition) ReleaseGPU

```go
func (eigen *EigenDecomposition) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the eigenvalue decomposition

#### type FileDataset

```go
type FileDataset struct {
}
```

FileDataset loads data from files on demand

#### func  NewFileDataset

```go
func NewFileDataset(inputPaths, targetPaths []string, inputShape, targetShape []int) (*FileDataset, error)
```
NewFileDataset creates a new file-based dataset

#### func (*FileDataset) GetItem

```go
func (ds *FileDataset) GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error)
```
GetItem loads and returns a single item from files

#### func (*FileDataset) GetShape

```go
func (ds *FileDataset) GetShape() ([]int, []int)
```
GetShape returns the shape of input and target tensors

#### func (*FileDataset) Len

```go
func (ds *FileDataset) Len() int
```
Len returns the size of the dataset

#### func (*FileDataset) SetTransform

```go
func (ds *FileDataset) SetTransform(transform func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error))
```
SetTransform sets a transformation function for the data

#### type GPUBatchNormLayer

```go
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
```

GPUBatchNormLayer represents a batch normalization layer with GPU acceleration
and Gonum compatibility

#### func  NewGPUBatchNormLayer

```go
func NewGPUBatchNormLayer(features int, epsilon, momentum float32) *GPUBatchNormLayer
```
NewGPUBatchNormLayer creates a new GPU-accelerated batch normalization layer

#### func (*GPUBatchNormLayer) Backward

```go
func (layer *GPUBatchNormLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error)
```
Backward performs the backward pass of batch normalization

#### func (*GPUBatchNormLayer) Forward

```go
func (layer *GPUBatchNormLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs the forward pass of batch normalization

#### func (*GPUBatchNormLayer) ReleaseGPU

```go
func (layer *GPUBatchNormLayer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*GPUBatchNormLayer) SetTraining

```go
func (layer *GPUBatchNormLayer) SetTraining(training bool)
```
SetTraining sets the layer to training or inference mode

#### type GPUConvLayer

```go
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
```

GPUConvLayer represents a convolutional layer with GPU acceleration

#### func  NewGPUConvLayer

```go
func NewGPUConvLayer(inputChannels, outputChannels, kernelSize, stride, padding int) *GPUConvLayer
```
NewGPUConvLayer creates a new GPU-accelerated convolutional layer

#### func (*GPUConvLayer) Backward

```go
func (layer *GPUConvLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error)
```
Backward performs the backward pass of the convolutional layer

#### func (*GPUConvLayer) Forward

```go
func (layer *GPUConvLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs the forward pass of the convolutional layer

#### func (*GPUConvLayer) ReleaseGPU

```go
func (layer *GPUConvLayer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### type GPUDense

```go
type GPUDense struct {
}
```

GPUDense wraps a tensor to implement gonum's mat.Matrix interface

#### func  FromGonum

```go
func FromGonum(m mat.Matrix) *GPUDense
```
FromGonum converts a gonum matrix to GPUDense

#### func  NewGPUDense

```go
func NewGPUDense(rows, cols int, data []float64) *GPUDense
```
NewGPUDense creates a new GPU-backed matrix compatible with Gonum

#### func  NewGPUDenseFromTensor

```go
func NewGPUDenseFromTensor(t *tensor.Tensor) *GPUDense
```

#### func (*GPUDense) Add

```go
func (g *GPUDense) Add(a, b mat.Matrix)
```
GPU-accelerated element-wise addition

#### func (*GPUDense) AddScalar

```go
func (g *GPUDense) AddScalar(scalar float64)
```
GPU-accelerated scalar addition

#### func (*GPUDense) At

```go
func (g *GPUDense) At(i, j int) float64
```

#### func (*GPUDense) Cholesky

```go
func (g *GPUDense) Cholesky() *GPUDense
```
Cholesky performs Cholesky decomposition and returns the lower triangular matrix
L

#### func (*GPUDense) Det

```go
func (g *GPUDense) Det() float64
```
Det computes the matrix determinant using GPU acceleration

#### func (*GPUDense) Dims

```go
func (g *GPUDense) Dims() (r, c int)
```
Implement mat.Matrix interface

#### func (*GPUDense) DivElem

```go
func (g *GPUDense) DivElem(a, b mat.Matrix)
```
GPU-accelerated element-wise division

#### func (*GPUDense) Eigen

```go
func (g *GPUDense) Eigen() (*GPUDense, *GPUDense)
```
Eigen performs eigenvalue decomposition for symmetric matrices

#### func (*GPUDense) GetTensor

```go
func (g *GPUDense) GetTensor() *tensor.Tensor
```
GetTensor returns the underlying tensor.Tensor

#### func (*GPUDense) Inverse

```go
func (g *GPUDense) Inverse() *GPUDense
```
Inverse computes the matrix inverse using GPU acceleration

#### func (*GPUDense) LU

```go
func (g *GPUDense) LU() (*GPUDense, *GPUDense, []int)
```
LU performs LU decomposition and returns L, U matrices and pivot indices

#### func (*GPUDense) Mul

```go
func (g *GPUDense) Mul(a, b mat.Matrix)
```
GPU-accelerated matrix multiplication

#### func (*GPUDense) MulElem

```go
func (g *GPUDense) MulElem(a, b mat.Matrix)
```
GPU-accelerated element-wise multiplication (Hadamard product)

#### func (*GPUDense) QR

```go
func (g *GPUDense) QR() (*GPUDense, *GPUDense)
```
QR performs QR decomposition and returns Q, R matrices

#### func (*GPUDense) ReleaseGPU

```go
func (g *GPUDense) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*GPUDense) SVD

```go
func (g *GPUDense) SVD() (*GPUDense, *GPUDense, *GPUDense)
```
SVD performs Singular Value Decomposition

#### func (*GPUDense) Scale

```go
func (g *GPUDense) Scale(scalar float64)
```
GPU-accelerated scalar multiplication

#### func (*GPUDense) Set

```go
func (g *GPUDense) Set(i, j int, v float64)
```
Implement mat.Mutable interface for setting values

#### func (*GPUDense) Sub

```go
func (g *GPUDense) Sub(a, b mat.Matrix)
```
GPU-accelerated element-wise subtraction

#### func (*GPUDense) T

```go
func (g *GPUDense) T() mat.Matrix
```

#### func (*GPUDense) ToGonum

```go
func (g *GPUDense) ToGonum() *mat.Dense
```
ToGonum converts back to a standard gonum Dense matrix

#### type GPUEigenDecomposition

```go
type GPUEigenDecomposition struct {
	Eigenvalues  *mat.VecDense
	Eigenvectors *mat.Dense
}
```

GPUEigenDecomposition represents eigenvalue decomposition results in Gonum
format

#### func  BatchGPUEigen

```go
func BatchGPUEigen(matrices []mat.Matrix) []*GPUEigenDecomposition
```
BatchGPUEigen performs multiple eigenvalue decompositions efficiently

#### func  GPUEigen

```go
func GPUEigen(a mat.Matrix) *GPUEigenDecomposition
```
GPUEigen is a drop-in replacement for gonum's eigenvalue decomposition

#### type GPULUDecomposition

```go
type GPULUDecomposition struct {
	L            *mat.Dense
	U            *mat.Dense
	PivotIndices []int
}
```

GPULUDecomposition represents LU decomposition results in Gonum format

#### func  GPULU

```go
func GPULU(a mat.Matrix) *GPULUDecomposition
```
GPULU is a drop-in replacement for gonum's LU decomposition

#### type GPULayerNormLayer

```go
type GPULayerNormLayer struct {
	Features   int
	Epsilon    float32
	Gamma      *GPUDense        // Scale parameters
	Beta       *GPUDense        // Shift parameters
	LastInput  *tensor.Tensor   // Stored for backward pass
	LastResult *BatchNormResult // Stored for backward pass
}
```

GPULayerNormLayer represents a layer normalization layer with GPU acceleration

#### func  NewGPULayerNormLayer

```go
func NewGPULayerNormLayer(features int, epsilon float32) *GPULayerNormLayer
```
NewGPULayerNormLayer creates a new GPU-accelerated layer normalization layer

#### func (*GPULayerNormLayer) Backward

```go
func (layer *GPULayerNormLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error)
```
Backward performs the backward pass of layer normalization

#### func (*GPULayerNormLayer) Forward

```go
func (layer *GPULayerNormLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs the forward pass of layer normalization

#### func (*GPULayerNormLayer) ReleaseGPU

```go
func (layer *GPULayerNormLayer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### type GPUMemoryPool

```go
type GPUMemoryPool struct {
}
```

GPUMemoryPool manages GPU memory allocation and reuse

#### func  NewGPUMemoryPool

```go
func NewGPUMemoryPool(maxMemory int64) (*GPUMemoryPool, error)
```
NewGPUMemoryPool creates a new GPU memory pool

#### func (*GPUMemoryPool) Allocate

```go
func (p *GPUMemoryPool) Allocate(size int64) (unsafe.Pointer, error)
```
Allocate allocates memory from the pool

#### func (*GPUMemoryPool) Free

```go
func (p *GPUMemoryPool) Free(ptr unsafe.Pointer) error
```
Free returns memory to the pool for reuse

#### func (*GPUMemoryPool) GetStats

```go
func (p *GPUMemoryPool) GetStats() MemoryStats
```
GetStats returns memory pool statistics

#### func (*GPUMemoryPool) GetUsage

```go
func (p *GPUMemoryPool) GetUsage() int64
```
GetUsage returns current memory usage

#### func (*GPUMemoryPool) Release

```go
func (p *GPUMemoryPool) Release(ptr unsafe.Pointer) error
```
Release permanently frees memory and releases GPU resources

#### func (*GPUMemoryPool) ReleaseAll

```go
func (p *GPUMemoryPool) ReleaseAll()
```
ReleaseAll releases all memory pool resources

#### type GPUPoolLayer

```go
type GPUPoolLayer struct {
	PoolType   string // "max" or "avg"
	PoolSize   int
	Stride     int
	Padding    int
	LastInput  *tensor.Tensor
	LastResult *MaxPool2DResult // For max pooling (contains indices)
}
```

GPUPoolLayer represents a pooling layer with GPU acceleration

#### func  NewGPUAvgPoolLayer

```go
func NewGPUAvgPoolLayer(poolSize, stride, padding int) *GPUPoolLayer
```
NewGPUAvgPoolLayer creates a new GPU-accelerated average pooling layer

#### func  NewGPUMaxPoolLayer

```go
func NewGPUMaxPoolLayer(poolSize, stride, padding int) *GPUPoolLayer
```
NewGPUMaxPoolLayer creates a new GPU-accelerated max pooling layer

#### func (*GPUPoolLayer) Backward

```go
func (layer *GPUPoolLayer) Backward(gradOutput *tensor.Tensor) (*tensor.Tensor, error)
```
Backward performs the backward pass of the pooling layer

#### func (*GPUPoolLayer) Forward

```go
func (layer *GPUPoolLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs the forward pass of the pooling layer

#### func (*GPUPoolLayer) ReleaseGPU

```go
func (layer *GPUPoolLayer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### type GPUQRDecomposition

```go
type GPUQRDecomposition struct {
	Q *mat.Dense
	R *mat.Dense
}
```

GPUQRDecomposition represents QR decomposition results in Gonum format

#### func  BatchGPUQR

```go
func BatchGPUQR(matrices []mat.Matrix) []*GPUQRDecomposition
```
BatchGPUQR performs multiple QR decompositions efficiently

#### func  GPUQR

```go
func GPUQR(a mat.Matrix) *GPUQRDecomposition
```
GPUQR is a drop-in replacement for gonum's QR decomposition

#### type GPUSVDDecomposition

```go
type GPUSVDDecomposition struct {
	U  *mat.Dense
	S  *mat.VecDense
	VT *mat.Dense
}
```

GPUSVDDecomposition represents SVD results in Gonum format

#### func  BatchGPUSVD

```go
func BatchGPUSVD(matrices []mat.Matrix) []*GPUSVDDecomposition
```
BatchGPUSVD performs multiple SVD decompositions efficiently

#### func  GPUSVD

```go
func GPUSVD(a mat.Matrix) *GPUSVDDecomposition
```
GPUSVD is a drop-in replacement for gonum's SVD

#### type GPUSparse

```go
type GPUSparse struct {
}
```

GPUSparse wraps a sparse tensor to implement gonum's mat.Matrix interface

#### func  BatchGPUSparseAdd

```go
func BatchGPUSparseAdd(operations []struct{ A, B *GPUSparse }) []*GPUSparse
```
BatchGPUSparseAdd performs multiple sparse matrix additions efficiently

#### func  BatchGPUSparseScalarMul

```go
func BatchGPUSparseScalarMul(matrices []*GPUSparse, scalars []float64) []*GPUSparse
```
BatchGPUSparseScalarMul performs multiple sparse scalar multiplications
efficiently

#### func  FromGonumSparse

```go
func FromGonumSparse(rows, cols int, data []float64, threshold float64) *GPUSparse
```
FromGonumSparse converts a gonum sparse matrix to GPUSparse Note: This is a
placeholder - in a full implementation, you'd support various gonum sparse
matrix types when they become available

#### func  GPUDenseToSparse

```go
func GPUDenseToSparse(m mat.Matrix, threshold float64) *GPUSparse
```
GPUDenseToSparse converts a dense matrix to sparse format

#### func  NewGPUSparse

```go
func NewGPUSparse(rows, cols int, rowIndices, colIndices []int32, values []float32) *GPUSparse
```
NewGPUSparse creates a new GPU-backed sparse matrix compatible with Gonum

#### func  NewGPUSparseFromDense

```go
func NewGPUSparseFromDense(m mat.Matrix, threshold float64) *GPUSparse
```
NewGPUSparseFromDense creates a sparse matrix from a dense gonum matrix

#### func (*GPUSparse) At

```go
func (gs *GPUSparse) At(i, j int) float64
```

#### func (*GPUSparse) ConvertToCSC

```go
func (gs *GPUSparse) ConvertToCSC() error
```
ConvertToCSC converts the sparse matrix to CSC format

#### func (*GPUSparse) ConvertToCSR

```go
func (gs *GPUSparse) ConvertToCSR() error
```
ConvertToCSR converts the sparse matrix to CSR format

#### func (*GPUSparse) Dims

```go
func (gs *GPUSparse) Dims() (r, c int)
```
Implement mat.Matrix interface

#### func (*GPUSparse) GetDensity

```go
func (gs *GPUSparse) GetDensity() float64
```
GetDensity returns the density (sparsity ratio) of the matrix

#### func (*GPUSparse) GetFormat

```go
func (gs *GPUSparse) GetFormat() tensor.SparseFormat
```
GetFormat returns the current storage format

#### func (*GPUSparse) GetNNZ

```go
func (gs *GPUSparse) GetNNZ() int
```
GetNNZ returns the number of non-zero elements

#### func (*GPUSparse) GetSparseTensor

```go
func (gs *GPUSparse) GetSparseTensor() *tensor.SparseTensor
```
GetSparseTensor returns the underlying sparse tensor (for advanced users)

#### func (*GPUSparse) ReleaseGPU

```go
func (gs *GPUSparse) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*GPUSparse) SparseAdd

```go
func (gs *GPUSparse) SparseAdd(other *GPUSparse) *GPUSparse
```
SparseAdd performs sparse matrix addition

#### func (*GPUSparse) SparseMatVec

```go
func (gs *GPUSparse) SparseMatVec(x []float64) []float64
```
SparseMatVec performs sparse matrix-vector multiplication

#### func (*GPUSparse) SparseMul

```go
func (gs *GPUSparse) SparseMul(a, b interface{}) *mat.Dense
```
SparseMul performs sparse matrix multiplication

#### func (*GPUSparse) SparseScalarMul

```go
func (gs *GPUSparse) SparseScalarMul(scalar float64) *GPUSparse
```
SparseScalarMul performs scalar multiplication

#### func (*GPUSparse) T

```go
func (gs *GPUSparse) T() mat.Matrix
```

#### func (*GPUSparse) ToDense

```go
func (gs *GPUSparse) ToDense() *mat.Dense
```
ToDense converts the sparse matrix to a dense gonum matrix

#### type GradientFunction

```go
type GradientFunction struct {
	OpType       OpType
	Inputs       []*GradientTensor
	Outputs      []*GradientTensor
	BackwardFn   func(gradOutput *tensor.Tensor) ([]*tensor.Tensor, error)
	SavedTensors []*tensor.Tensor       // Tensors saved for backward pass
	Metadata     map[string]interface{} // Additional metadata for backward pass
}
```

GradientFunction represents a backward function for an operation

#### func  CreateBackwardFunction

```go
func CreateBackwardFunction(opType OpType, inputs []*GradientTensor, savedTensors []*tensor.Tensor, metadata map[string]interface{}) *GradientFunction
```
CreateBackwardFunction creates a backward function for gradient computation

#### type GradientMode

```go
type GradientMode int
```

GradientMode represents different gradient computation modes

```go
const (
	NoGrad GradientMode = iota // No gradient computation
	Grad                       // Compute gradients
)
```

#### func  GetGradientMode

```go
func GetGradientMode() GradientMode
```
GetGradientMode returns the current gradient computation mode

#### type GradientTensor

```go
type GradientTensor struct {
	Tensor       *tensor.Tensor
	Gradient     *tensor.Tensor
	RequiresGrad bool
	GradFn       *GradientFunction
	IsLeaf       bool // True if this tensor is a leaf (input/parameter)
}
```

GradientTensor wraps a tensor with gradient tracking capabilities

#### func  CloneGradientTensor

```go
func CloneGradientTensor(gt *GradientTensor) (*GradientTensor, error)
```
CloneGradientTensor creates a deep copy of a gradient tensor

#### func  DetachGradient

```go
func DetachGradient(a *GradientTensor) *GradientTensor
```
DetachGradient creates a new GradientTensor that doesn't require gradients

#### func  GradAdd

```go
func GradAdd(a, b *GradientTensor) (*GradientTensor, error)
```
GradAdd performs gradient-aware addition

#### func  GradBatchNorm

```go
func GradBatchNorm(input, gamma, beta *GradientTensor, epsilon float32) (*GradientTensor, error)
```
BatchNorm performs gradient-aware batch normalization

#### func  GradCat

```go
func GradCat(tensors []*GradientTensor, dim int) (*GradientTensor, error)
```
Concatenation along a specific dimension

#### func  GradConv2D

```go
func GradConv2D(input, kernel *GradientTensor, params Conv2DParams) (*GradientTensor, error)
```
Conv2D performs gradient-aware 2D convolution

#### func  GradCrossEntropyLoss

```go
func GradCrossEntropyLoss(predictions, targets *GradientTensor) (*GradientTensor, error)
```
Cross-entropy loss with built-in softmax

#### func  GradDiv

```go
func GradDiv(a, b *GradientTensor) (*GradientTensor, error)
```
GradDiv performs gradient-aware element-wise division

#### func  GradDropout

```go
func GradDropout(input *GradientTensor, probability float32, training bool, seed uint32) (*GradientTensor, error)
```
Dropout performs gradient-aware dropout for regularization

#### func  GradFlatten

```go
func GradFlatten(input *GradientTensor, startDim int) (*GradientTensor, error)
```
Flatten reshapes a multi-dimensional tensor to 2D

#### func  GradMSELoss

```go
func GradMSELoss(predictions, targets *GradientTensor) (*GradientTensor, error)
```
GradMSELoss performs gradient-aware MSE loss computation

#### func  GradMatMul

```go
func GradMatMul(a, b *GradientTensor) (*GradientTensor, error)
```
GradMatMul performs gradient-aware matrix multiplication

#### func  GradMaxPool2D

```go
func GradMaxPool2D(input *GradientTensor, params Pool2DParams) (*GradientTensor, error)
```
MaxPool2D performs gradient-aware 2D max pooling

#### func  GradMean

```go
func GradMean(a *GradientTensor) (*GradientTensor, error)
```
GradMean performs gradient-aware mean reduction

#### func  GradMul

```go
func GradMul(a, b *GradientTensor) (*GradientTensor, error)
```
GradMul performs gradient-aware element-wise multiplication

#### func  GradReLU

```go
func GradReLU(a *GradientTensor) (*GradientTensor, error)
```
GradReLU performs gradient-aware ReLU activation

#### func  GradReshape

```go
func GradReshape(a *GradientTensor, newShape []int) (*GradientTensor, error)
```
GradReshape performs gradient-aware tensor reshaping

#### func  GradSigmoid

```go
func GradSigmoid(a *GradientTensor) (*GradientTensor, error)
```
GradSigmoid performs gradient-aware Sigmoid activation

#### func  GradSoftmax

```go
func GradSoftmax(a *GradientTensor) (*GradientTensor, error)
```
GradSoftmax performs gradient-aware Softmax activation

#### func  GradSparseCrossEntropyLoss

```go
func GradSparseCrossEntropyLoss(predictions *GradientTensor, targetIndices []int) (*GradientTensor, error)
```
Sparse cross-entropy loss for integer targets

#### func  GradSqueeze

```go
func GradSqueeze(input *GradientTensor, dim int) (*GradientTensor, error)
```
Squeeze removes dimensions of size 1

#### func  GradSub

```go
func GradSub(a, b *GradientTensor) (*GradientTensor, error)
```
GradSub performs gradient-aware subtraction

#### func  GradSum

```go
func GradSum(a *GradientTensor) (*GradientTensor, error)
```
GradSum performs gradient-aware sum reduction

#### func  GradTanh

```go
func GradTanh(a *GradientTensor) (*GradientTensor, error)
```
GradTanh performs gradient-aware Tanh activation

#### func  GradTranspose

```go
func GradTranspose(a *GradientTensor) (*GradientTensor, error)
```
GradTranspose performs gradient-aware transpose

#### func  GradUnsqueeze

```go
func GradUnsqueeze(input *GradientTensor, dim int) (*GradientTensor, error)
```
Unsqueeze adds a dimension of size 1

#### func  GradientCheckpoint

```go
func GradientCheckpoint(fn func(*GradientTensor) (*GradientTensor, error), input *GradientTensor) (*GradientTensor, error)
```
GradientCheckpoint implements gradient checkpointing to save memory

#### func  Linear

```go
func Linear(input, weight, bias *GradientTensor) (*GradientTensor, error)
```
Linear performs a linear transformation: output = input * weight + bias

#### func  NewGradientTensor

```go
func NewGradientTensor(t *tensor.Tensor, requiresGrad bool) *GradientTensor
```
NewGradientTensor creates a new gradient tensor

#### func (*GradientTensor) Backward

```go
func (gt *GradientTensor) Backward() error
```
Backward performs backpropagation from this tensor

#### func (*GradientTensor) BackwardWithGradient

```go
func (gt *GradientTensor) BackwardWithGradient(grad *tensor.Tensor) error
```
BackwardWithGradient performs backpropagation with a specific gradient

#### type InMemoryDataset

```go
type InMemoryDataset struct {
}
```

InMemoryDataset is a simple dataset that holds all data in memory

#### func  NewInMemoryDataset

```go
func NewInMemoryDataset(inputs, targets []*tensor.Tensor) (*InMemoryDataset, error)
```
NewInMemoryDataset creates a new in-memory dataset

#### func (*InMemoryDataset) GetItem

```go
func (ds *InMemoryDataset) GetItem(index int) (*tensor.Tensor, *tensor.Tensor, error)
```
GetItem returns a single item from the dataset

#### func (*InMemoryDataset) GetShape

```go
func (ds *InMemoryDataset) GetShape() ([]int, []int)
```
GetShape returns the shape of input and target tensors

#### func (*InMemoryDataset) Len

```go
func (ds *InMemoryDataset) Len() int
```
Len returns the size of the dataset

#### type LRUList

```go
type LRUList struct {
}
```

LRUList implements a doubly-linked list for LRU cache

#### func  NewLRUList

```go
func NewLRUList() *LRUList
```
NewLRUList creates a new LRU list

#### func (*LRUList) AddToFront

```go
func (lru *LRUList) AddToFront(key string) *LRUNode
```
AddToFront adds a node to the front of the list

#### func (*LRUList) MoveToFront

```go
func (lru *LRUList) MoveToFront(node *LRUNode)
```
MoveToFront moves a node to the front

#### func (*LRUList) Remove

```go
func (lru *LRUList) Remove(node *LRUNode)
```
Remove removes a node from the list

#### func (*LRUList) RemoveLast

```go
func (lru *LRUList) RemoveLast() string
```
RemoveLast removes and returns the key of the last node

#### type LRUNode

```go
type LRUNode struct {
}
```

LRUNode represents a node in the LRU list

#### type LUDecomposition

```go
type LUDecomposition struct {
	L            *tensor.Tensor // Lower triangular matrix
	U            *tensor.Tensor // Upper triangular matrix
	PivotIndices []int          // Pivot indices for row swaps
}
```

LUDecomposition represents the result of LU decomposition

#### func  LU

```go
func LU(A *tensor.Tensor) (*LUDecomposition, error)
```
LU performs LU decomposition using the Accelerate framework

#### func (*LUDecomposition) ReleaseGPU

```go
func (lu *LUDecomposition) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the LU decomposition

#### type LoadRequest

```go
type LoadRequest struct {
}
```

LoadRequest represents a data loading request

#### type LossResult

```go
type LossResult struct {
	Loss      float32
	Gradients *tensor.Tensor
}
```

LossResult contains the computed loss value and gradients

#### func  LossForwardBackward

```go
func LossForwardBackward(predictions, targets *tensor.Tensor, lossType LossType, params ...float32) (*LossResult, error)
```
LossForwardBackward computes both forward and backward passes efficiently

#### func  SparseCategoricalCrossEntropyForwardBackward

```go
func SparseCategoricalCrossEntropyForwardBackward(predictions *tensor.Tensor, targetIndices []int) (*LossResult, error)
```
SparseCategoricalCrossEntropyForwardBackward computes both forward and backward
passes for sparse categorical cross-entropy

#### func (*LossResult) ReleaseGPU

```go
func (lr *LossResult) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the loss result

#### type LossType

```go
type LossType int
```

LossType represents different loss function types

```go
const (
	MSE LossType = iota
	BinaryCrossEntropy
	CategoricalCrossEntropy
	SparseCategoricalCrossEntropy
	Huber
	MAE
	Hinge
)
```

#### func (LossType) String

```go
func (lt LossType) String() string
```
String returns string representation of loss type

#### type MaxPool2DResult

```go
type MaxPool2DResult struct {
	Output  *tensor.Tensor
	Indices *tensor.Tensor // Stores indices of max elements for backward pass
}
```

MaxPool2DResult contains the result of max pooling and indices for backward pass

#### func  BatchGPUMaxPool2D

```go
func BatchGPUMaxPool2D(inputs []*tensor.Tensor, params Pool2DParams) ([]*MaxPool2DResult, error)
```
BatchGPUMaxPool2D performs max pooling on multiple input tensors

#### func  MaxPool2D

```go
func MaxPool2D(input *tensor.Tensor, poolSize, stride, padding int) (*MaxPool2DResult, error)
```
MaxPool2D performs 2D max pooling with default parameters

#### func  MaxPool2DForward

```go
func MaxPool2DForward(input *tensor.Tensor, params Pool2DParams) (*MaxPool2DResult, error)
```
MaxPool2DForward performs 2D max pooling forward pass

#### func (*MaxPool2DResult) ReleaseGPU

```go
func (mpr *MaxPool2DResult) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the max pooling result

#### type MemoryBlock

```go
type MemoryBlock struct {
}
```

MemoryBlock represents a block of GPU memory

#### type MemoryDataLoader

```go
type MemoryDataLoader struct {
}
```

MemoryDataLoader loads data efficiently with memory optimization

#### func  NewMemoryDataLoader

```go
func NewMemoryDataLoader(dataset Dataset, config DataLoaderConfig) (*MemoryDataLoader, error)
```
NewMemoryDataLoader creates a new memory-efficient data loader

#### func (*MemoryDataLoader) BatchCount

```go
func (dl *MemoryDataLoader) BatchCount() int
```
BatchCount returns the number of batches in the dataset

#### func (*MemoryDataLoader) GetBatch

```go
func (dl *MemoryDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
```
GetBatch returns a batch of data

#### func (*MemoryDataLoader) GetDatasetSize

```go
func (dl *MemoryDataLoader) GetDatasetSize() int
```
GetDatasetSize returns the size of the dataset

#### func (*MemoryDataLoader) GetStats

```go
func (dl *MemoryDataLoader) GetStats() DataLoaderStats
```
GetStats returns data loader statistics

#### func (*MemoryDataLoader) Reset

```go
func (dl *MemoryDataLoader) Reset() error
```
Reset resets the data loader state

#### func (*MemoryDataLoader) SetBatchSize

```go
func (dl *MemoryDataLoader) SetBatchSize(batchSize int)
```
SetBatchSize sets the batch size

#### func (*MemoryDataLoader) Shuffle

```go
func (dl *MemoryDataLoader) Shuffle() error
```
Shuffle shuffles the dataset indices

#### func (*MemoryDataLoader) Stop

```go
func (dl *MemoryDataLoader) Stop()
```
Stop stops the data loader and cleans up resources

#### type MemoryProfiler

```go
type MemoryProfiler struct {
}
```

MemoryProfiler provides memory usage profiling

#### func  NewMemoryProfiler

```go
func NewMemoryProfiler(interval time.Duration, maxSamples int) *MemoryProfiler
```
NewMemoryProfiler creates a new memory profiler

#### func (*MemoryProfiler) GetPeakUsage

```go
func (mp *MemoryProfiler) GetPeakUsage() (gpuPeak, cpuPeak int64)
```
GetPeakUsage returns peak GPU and CPU usage

#### func (*MemoryProfiler) GetSamples

```go
func (mp *MemoryProfiler) GetSamples() []MemorySample
```
GetSamples returns all collected samples

#### func (*MemoryProfiler) Start

```go
func (mp *MemoryProfiler) Start(memPool *GPUMemoryPool, cache *TensorCache)
```
Start begins memory profiling

#### func (*MemoryProfiler) Stop

```go
func (mp *MemoryProfiler) Stop()
```
Stop ends memory profiling

#### type MemorySample

```go
type MemorySample struct {
	Timestamp   time.Time
	GPUUsage    int64
	CPUUsage    int64
	CacheHits   int64
	CacheMisses int64
}
```

MemorySample represents a memory usage sample

#### type MemoryStats

```go
type MemoryStats struct {
	TotalAllocated     int64
	TotalFreed         int64
	PeakUsage          int64
	AllocationCount    int64
	FreeCount          int64
	CacheHits          int64
	CacheMisses        int64
	FragmentationRatio float32
}
```

MemoryStats tracks memory pool statistics

#### type OpType

```go
type OpType int
```

OpType represents different operation types for gradient computation

```go
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
```

#### type OptimizerType

```go
type OptimizerType int
```

OptimizerType represents different optimizer types (matching your existing ones)

```go
const (
	OptimizerSGD OptimizerType = iota
	OptimizerAdam
	OptimizerAdamW
	OptimizerRMSprop
)
```

#### type Pool2DParams

```go
type Pool2DParams struct {
	PoolH   int // Pool kernel height
	PoolW   int // Pool kernel width
	StrideH int // Stride in height dimension
	StrideW int // Stride in width dimension
	PadH    int // Padding in height dimension
	PadW    int // Padding in width dimension
}
```

Pool2DParams represents parameters for 2D pooling operations

#### type PrefetchedBatch

```go
type PrefetchedBatch struct {
}
```

PrefetchedBatch represents a prefetched batch

#### type QRDecomposition

```go
type QRDecomposition struct {
	Q *tensor.Tensor // Orthogonal matrix
	R *tensor.Tensor // Upper triangular matrix
}
```

QRDecomposition represents the result of QR decomposition

#### func  QR

```go
func QR(A *tensor.Tensor) (*QRDecomposition, error)
```
QR performs QR decomposition using the Accelerate framework

#### func (*QRDecomposition) ReleaseGPU

```go
func (qr *QRDecomposition) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the QR decomposition

#### type SVDDecomposition

```go
type SVDDecomposition struct {
	U  *tensor.Tensor // Left singular vectors (m x m)
	S  *tensor.Tensor // Singular values (min(m,n))
	VT *tensor.Tensor // Right singular vectors transposed (n x n)
}
```

SVDDecomposition represents the result of Singular Value Decomposition

#### func  SVD

```go
func SVD(A *tensor.Tensor) (*SVDDecomposition, error)
```
SVD performs Singular Value Decomposition using the Accelerate framework

#### func (*SVDDecomposition) ReleaseGPU

```go
func (svd *SVDDecomposition) ReleaseGPU()
```
ReleaseGPU releases GPU resources for the SVD decomposition

#### type SchedulerType

```go
type SchedulerType int
```

SchedulerType represents different learning rate scheduler types

```go
const (
	NoScheduler SchedulerType = iota
	StepLR
	ExponentialLR
	CosineAnnealingLR
	PolynomialLR
	WarmupLR
)
```

#### type SparseMatrixInfo

```go
type SparseMatrixInfo struct {
	Rows        int
	Cols        int
	NNZ         int
	Density     float64
	Format      tensor.SparseFormat
	MemoryUsage int64 // Estimated memory usage in bytes
}
```

SparseMatrixInfo provides information about a sparse matrix

#### func  GetSparseMatrixInfo

```go
func GetSparseMatrixInfo(gs *GPUSparse) SparseMatrixInfo
```
GetSparseMatrixInfo returns detailed information about a sparse matrix

#### func (SparseMatrixInfo) String

```go
func (info SparseMatrixInfo) String() string
```
String returns a string representation of sparse matrix info

#### type TensorCache

```go
type TensorCache struct {
}
```

TensorCache provides caching for frequently used tensors

#### func  NewTensorCache

```go
func NewTensorCache(maxSize int) *TensorCache
```
NewTensorCache creates a new tensor cache

#### func (*TensorCache) Clear

```go
func (tc *TensorCache) Clear()
```
Clear removes all items from cache

#### func (*TensorCache) Get

```go
func (tc *TensorCache) Get(key string, shape []int, createFn func() (*tensor.Tensor, error)) (*tensor.Tensor, error)
```
Get retrieves a tensor from cache or creates a new one

#### func (*TensorCache) GetStats

```go
func (tc *TensorCache) GetStats() (hits, misses int64, hitRatio float32)
```
GetStats returns cache statistics

#### func (*TensorCache) Put

```go
func (tc *TensorCache) Put(key string, tensor *tensor.Tensor, shape []int)
```
Put adds a tensor to the cache

#### type TrainableModel

```go
type TrainableModel interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*GradientTensor
	ZeroGrad()
	SetTraining(training bool)
	GetName() string
}
```

Model interface for training

#### type Trainer

```go
type Trainer struct {
	Config  *TrainingConfig
	State   *TrainingState
	Metrics *TrainingMetrics
	Model   TrainableModel

	// Callbacks
	OnEpochStart    func(epoch int) error
	OnEpochEnd      func(epoch int, metrics *TrainingMetrics) error
	OnBatchStart    func(epoch, batch int) error
	OnBatchEnd      func(epoch, batch int, loss float32) error
	OnValidation    ValidationCallback
	OnTrainingStart func() error
	OnTrainingEnd   func(metrics *TrainingMetrics) error
}
```

Trainer manages memory-efficient training

#### func  NewTrainer

```go
func NewTrainer(config *TrainingConfig, model TrainableModel) (*Trainer, error)
```
NewTrainer creates a new memory-efficient trainer

#### func (*Trainer) GetMetrics

```go
func (t *Trainer) GetMetrics() *TrainingMetrics
```
GetMetrics returns current training metrics

#### func (*Trainer) GetState

```go
func (t *Trainer) GetState() *TrainingState
```
GetState returns current training state

#### func (*Trainer) LoadCheckpoint

```go
func (t *Trainer) LoadCheckpoint(filepath string) error
```
LoadCheckpoint loads a training state from checkpoint

#### func (*Trainer) Release

```go
func (t *Trainer) Release()
```
Release releases GPU resources used by the trainer

#### func (*Trainer) SaveCheckpoint

```go
func (t *Trainer) SaveCheckpoint(filepath string) error
```
SaveCheckpoint saves the current training state

#### func (*Trainer) Train

```go
func (t *Trainer) Train(trainLoader, validLoader DataLoader) error
```
Train performs memory-efficient training

#### type TrainingConfig

```go
type TrainingConfig struct {
	// Memory management
	GradientAccumulationSteps int   // Number of mini-batches to accumulate gradients
	MaxMemoryUsage            int64 // Maximum GPU memory usage in bytes (0 = no limit)
	EnableGradientCheckpoint  bool  // Enable gradient checkpointing to save memory
	EnableMixedPrecision      bool  // Enable mixed precision training

	// Training parameters
	LearningRate     float32
	WeightDecay      float32
	GradientClipping float32 // Max gradient norm (0 = no clipping)
	EpochCount       int
	BatchSize        int
	ValidationFreq   int // Validate every N batches (0 = no validation)

	// Optimizer settings
	OptimizerType OptimizerType
	Beta1         float32 // For Adam/AdamW
	Beta2         float32 // For Adam/AdamW
	Epsilon       float32 // For Adam/AdamW
	Momentum      float32 // For SGD with momentum

	// Scheduler settings
	SchedulerType     SchedulerType
	SchedulerGamma    float32 // Decay factor for step/exponential schedulers
	SchedulerStepSize int     // Step size for step scheduler
	WarmupSteps       int     // Number of warmup steps
	TotalSteps        int     // Total training steps (for cosine annealing)

	// Memory optimization
	ClearCacheFreq   int  // Clear GPU cache every N batches
	PrefetchBatches  int  // Number of batches to prefetch (0 = disable)
	AsyncDataLoading bool // Enable async data loading
}
```

TrainingConfig contains configuration for memory-efficient training

#### type TrainingMetrics

```go
type TrainingMetrics struct {
	Losses           []float32
	ValidationLosses []float32
	LearningRates    []float32
	GradientNorms    []float32
	BatchTimes       []time.Duration
	MemoryUsages     []int64
	Epochs           []int

	// Statistics
	BestValidationLoss float32
	BestEpoch          int
	TotalTrainingTime  time.Duration
	AverageEpochTime   time.Duration
}
```

TrainingMetrics tracks training progress and performance

#### type TrainingState

```go
type TrainingState struct {
	Epoch          int
	Batch          int
	Step           int
	Loss           float32
	ValidationLoss float32
	LearningRate   float32
	GradientNorm   float32
	MemoryUsage    int64
	BatchTime      time.Duration

	// Accumulated gradients
	AccumulatedGrads  map[string]*tensor.Tensor
	AccumulationCount int

	// Optimizer state
	OptimizerInstance optimizer.Optimizer
	SchedulerInstance optimizer.LRScheduler

	// Memory management
	MemoryPool  *GPUMemoryPool
	TensorCache *TensorCache
}
```

TrainingState tracks the current state of training

#### type ValidationCallback

```go
type ValidationCallback func(epoch int, batch int, validationLoss float32) error
```

ValidationCallback is called during validation
