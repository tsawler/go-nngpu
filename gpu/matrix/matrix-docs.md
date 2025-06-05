# matrix
--
    import "."


## Usage

```go
const (
	OptimizedOpAdd = iota
	OptimizedOpSubtract
	OptimizedOpMultiply
	OptimizedOpDivide
	OptimizedOpMaximum
	OptimizedOpMinimum
	OptimizedOpPower
)
```
Optimized elementwise operation types

```go
const (
	OptimizedOpSum = iota
	OptimizedOpMax
	OptimizedOpMin
)
```
Optimized reduction operation types

#### func  AccumulateGradient

```go
func AccumulateGradient(existing, newGrad *tensor.Tensor) error
```
AccumulateGradient accumulates gradients using GPU acceleration

#### func  AccumulateGradientsAdvanced

```go
func AccumulateGradientsAdvanced(gradients []*GradientTensor, layerNames []string, weights []float32) error
```
AccumulateGradientsAdvanced performs advanced gradient accumulation

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

#### func  AddAutodiffBreakpoint

```go
func AddAutodiffBreakpoint(tensor *GradientTensor)
```
AddBreakpoint adds a breakpoint for a specific tensor

#### func  AddToAutodiffWatchList

```go
func AddToAutodiffWatchList(tensor *GradientTensor)
```
AddToWatchList adds a tensor to the watch list

#### func  AllocateAlignedBuffer

```go
func AllocateAlignedBuffer(device unsafe.Pointer, size int, alignment int) (unsafe.Pointer, error)
```
AllocateAlignedBuffer allocates an aligned GPU buffer using the optimized C
functions

#### func  ApplyAutodiffConfig

```go
func ApplyAutodiffConfig(config *AutodiffConfig)
```
ApplyAutodiffConfig applies the given configuration

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

#### func  ClearAutodiffDebugLog

```go
func ClearAutodiffDebugLog()
```
ClearDebugLog clears the debug log buffer

#### func  ClearGPUCache

```go
func ClearGPUCache()
```
ClearGPUCache clears GPU cache (placeholder for demo)

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

#### func  ClipGradientsAdaptive

```go
func ClipGradientsAdaptive(gradients []*GradientTensor, targetNorm float32, adaptationRate float32) (float32, error)
```
ClipGradientsAdaptive performs adaptive gradient clipping

#### func  CoalescedMemoryCopy

```go
func CoalescedMemoryCopy(device unsafe.Pointer, srcPtr, dstPtr unsafe.Pointer, size int, srcStride, dstStride int) error
```
CoalescedMemoryCopy performs optimized memory copy with coalescing

#### func  Col2Im

```go
func Col2Im(input *tensor.Tensor, outputShape []int, kernelH, kernelW, strideH, strideW, padH, padW int) (*tensor.Tensor, error)
```
Col2Im performs the col2im operation (inverse of im2col)

#### func  CompileOptimizedKernel

```go
func CompileOptimizedKernel(source string, options *KernelCompilationOptions) (unsafe.Pointer, error)
```
CompileOptimizedKernel compiles a kernel using the global cache

#### func  CompressGradients

```go
func CompressGradients(gradients []*GradientTensor) error
```
CompressGradients compresses gradients using the configured method

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

#### func  CreateCausalMask

```go
func CreateCausalMask(rows, cols int) (*tensor.Tensor, error)
```
CreateCausalMask creates a causal attention mask

#### func  CreateLayerNormParams

```go
func CreateLayerNormParams(features int) (gamma, beta *mat.Dense)
```
CreateLayerNormParams creates initialized gamma and beta parameters for layer
normalization

#### func  CreateOptimizedIntermediateTensor

```go
func CreateOptimizedIntermediateTensor(shape []int, operation string) (*tensor.Tensor, error)
```
CreateOptimizedIntermediateTensor creates an optimized intermediate tensor

#### func  DecompressGradient

```go
func DecompressGradient(compressed *CompressedTensor) (*tensor.Tensor, error)
```
DecompressGradient decompresses a compressed gradient tensor

#### func  DenseToSparse

```go
func DenseToSparse(A *tensor.Tensor, threshold float32) (*tensor.SparseTensor, error)
```
DenseToSparse converts a dense matrix to sparse format using GPU acceleration

#### func  DetectMemoryFragmentation

```go
func DetectMemoryFragmentation() float32
```
DetectMemoryFragmentation detects memory fragmentation

#### func  Determinant

```go
func Determinant(A *tensor.Tensor) (float32, error)
```
Determinant computes the matrix determinant using the Accelerate framework

#### func  DisableAdvancedGradientAccumulation

```go
func DisableAdvancedGradientAccumulation()
```
DisableAdvancedGradientAccumulation disables advanced gradient accumulation

#### func  DisableAdvancedOptimizer

```go
func DisableAdvancedOptimizer()
```
DisableAdvancedOptimizer disables the advanced optimizer

#### func  DisableAutodiffDebugging

```go
func DisableAutodiffDebugging()
```
DisableDebugging disables autodiff debugging

#### func  DisableAutodiffProfiling

```go
func DisableAutodiffProfiling()
```
DisableProfiling disables autodiff profiling

#### func  DisableAutomaticCodeGeneration

```go
func DisableAutomaticCodeGeneration()
```
DisableAutomaticCodeGeneration disables automatic code generation

#### func  DisableGlobalFusion

```go
func DisableGlobalFusion()
```
DisableGlobalFusion disables global fusion optimizations

#### func  DisableGradientAnalysis

```go
func DisableGradientAnalysis()
```
DisableGradientAnalysis disables gradient analysis

#### func  DisableGradientNoise

```go
func DisableGradientNoise()
```
DisableGradientNoise disables gradient noise injection

#### func  DisableGradientSmoothing

```go
func DisableGradientSmoothing()
```
DisableGradientSmoothing disables gradient smoothing

#### func  DisableHigherOrderAutodiff

```go
func DisableHigherOrderAutodiff()
```
DisableHigherOrderAutodiff disables higher-order automatic differentiation

#### func  DisableMemoryEfficientAutodiff

```go
func DisableMemoryEfficientAutodiff()
```
DisableMemoryEfficientAutodiff disables memory-efficient autodiff

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

#### func  EnableAdvancedGradientAccumulation

```go
func EnableAdvancedGradientAccumulation(accumulationType AccumulationType, steps int, threshold float32)
```
EnableAdvancedGradientAccumulation enables advanced gradient accumulation

#### func  EnableAdvancedOptimizer

```go
func EnableAdvancedOptimizer()
```
EnableAdvancedOptimizer enables the advanced optimizer

#### func  EnableAutodiffDebugging

```go
func EnableAutodiffDebugging(logLevel DebugLogLevel)
```
EnableDebugging enables autodiff debugging

#### func  EnableAutodiffProfiling

```go
func EnableAutodiffProfiling()
```
EnableProfiling enables autodiff profiling

#### func  EnableAutomaticCodeGeneration

```go
func EnableAutomaticCodeGeneration()
```
EnableAutomaticCodeGeneration enables automatic code generation

#### func  EnableGlobalFusion

```go
func EnableGlobalFusion()
```
EnableGlobalFusion enables global fusion optimizations

#### func  EnableGradientAnalysis

```go
func EnableGradientAnalysis()
```
EnableGradientAnalysis enables gradient analysis

#### func  EnableGradientNoise

```go
func EnableGradientNoise(noiseType NoiseType, schedule *NoiseSchedule)
```
EnableGradientNoise enables gradient noise injection

#### func  EnableGradientSmoothing

```go
func EnableGradientSmoothing(smoothingType SmoothingType, windowSize int, alpha float32)
```
EnableGradientSmoothing enables gradient smoothing

#### func  EnableHigherOrderAutodiff

```go
func EnableHigherOrderAutodiff(maxOrder int)
```
EnableHigherOrderAutodiff enables higher-order automatic differentiation

#### func  EnableMemoryEfficientAutodiff

```go
func EnableMemoryEfficientAutodiff(config *MemoryEfficientConfig)
```
EnableMemoryEfficientAutodiff enables memory-efficient autodiff globally

#### func  EstimateMemoryUsage

```go
func EstimateMemoryUsage(config DataLoaderConfig, inputShape, targetShape []int) int64
```
EstimateMemoryUsage estimates memory usage for a data loader configuration

#### func  Flatten4DTo2D

```go
func Flatten4DTo2D(t *tensor.Tensor) (*tensor.Tensor, error)
```
Flatten4DTo2D flattens a 4D tensor to 2D for compatibility with Gonum operations

#### func  FloatMP16ToFloat32

```go
func FloatMP16ToFloat32(h FloatMP16) float32
```
FloatMP16ToFloat32 converts float16 to float32

#### func  FlushGPUCache

```go
func FlushGPUCache(device unsafe.Pointer) error
```
FlushGPUCache flushes GPU cache to ensure optimal access patterns

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

#### func  GetAutodiffDebugLog

```go
func GetAutodiffDebugLog() string
```
GetDebugLog returns the accumulated debug log

#### func  GetAutodiffStats

```go
func GetAutodiffStats() map[string]interface{}
```
GetAutodiffStats returns comprehensive statistics about the autodiff system

#### func  GetCombinedMemoryStats

```go
func GetCombinedMemoryStats(state *TrainingState) (poolUsage, tensorUsage, peakUsage int64)
```
GetCombinedMemoryStats returns memory statistics from both the memory pool and
tensor tracking

#### func  GetGlobalFusionStats

```go
func GetGlobalFusionStats() map[string]*FusionStats
```
GetGlobalFusionStats returns global fusion statistics

#### func  GetGradientStats

```go
func GetGradientStats(gradTensors []*GradientTensor) (map[string]float32, error)
```
GetGradientStats returns statistics about gradients for debugging

#### func  GetGradientUtilitiesStats

```go
func GetGradientUtilitiesStats() map[string]interface{}
```
GetGradientUtilitiesStats returns comprehensive statistics

#### func  GetKernelCacheStats

```go
func GetKernelCacheStats() (hitRate float64, entries int, sizeBytes int64, hitCount int64, missCount int64)
```
GetKernelCacheStats returns global kernel cache statistics

#### func  GetMemoryEfficientStats

```go
func GetMemoryEfficientStats() map[string]interface{}
```
GetMemoryEfficientStats returns comprehensive statistics

#### func  GetMemoryOptimizationStats

```go
func GetMemoryOptimizationStats() map[string]interface{}
```
GetMemoryOptimizationStats returns memory optimization statistics

#### func  GetPhase8CStatus

```go
func GetPhase8CStatus() map[string]bool
```
GetPhase8CStatus returns the implementation status of Phase 8C

#### func  GetSharedMemoryUsageStats

```go
func GetSharedMemoryUsageStats() map[string]interface{}
```
GetSharedMemoryUsageStats returns statistics about shared memory usage

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

#### func  GreaterThanScalar

```go
func GreaterThanScalar(t *tensor.Tensor, scalar float32) (*tensor.Tensor, error)
```
GreaterThanScalar creates a binary mask tensor where elements > scalar are 1.0,
else 0.0

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

#### func  InPlaceOperation

```go
func InPlaceOperation(tensor *GradientTensor, operation func(*tensor.Tensor) error) error
```
InPlaceOperation performs an operation in-place to save memory

#### func  InitializeGlobalBufferReuseManager

```go
func InitializeGlobalBufferReuseManager(memoryPool *GPUMemoryPool)
```
InitializeGlobalBufferReuseManager initializes the global buffer reuse manager

#### func  InitializeKernelCache

```go
func InitializeKernelCache(device unsafe.Pointer)
```
InitializeKernelCache initializes the global kernel cache

#### func  InitializeMemoryOptimizationSuite

```go
func InitializeMemoryOptimizationSuite(device unsafe.Pointer, config *OptimizationConfig)
```
InitializeMemoryOptimizationSuite initializes the global memory optimization
suite

#### func  InitializeMemoryOptimizers

```go
func InitializeMemoryOptimizers()
```
InitializeMemoryOptimizers initializes global memory optimizers

#### func  InitializePhase8C

```go
func InitializePhase8C(device unsafe.Pointer) error
```
InitializePhase8C initializes all Phase 8C components with a unified interface

#### func  InitializePhase8CWithDefaults

```go
func InitializePhase8CWithDefaults() error
```
InitializePhase8CWithDefaults initializes Phase 8C with default configuration

#### func  InitializeRunningStats

```go
func InitializeRunningStats(features int) (runningMean, runningVar *mat.Dense)
```
InitializeRunningStats creates initialized running mean and variance for batch
normalization

#### func  InjectGradientNoise

```go
func InjectGradientNoise(gradients []*GradientTensor, noiseConfig *NoiseSchedule, step int64) error
```
InjectGradientNoise injects controlled noise into gradients

#### func  InspectGradientTensor

```go
func InspectGradientTensor(tensor *GradientTensor) string
```
InspectTensor provides detailed inspection of a gradient tensor

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

#### func  IsPhase8CInitialized

```go
func IsPhase8CInitialized() bool
```
IsPhase8CInitialized returns true if Phase 8C has been initialized

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

#### func  LayerNormForwardWithStats

```go
func LayerNormForwardWithStats(input, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error)
```
LayerNormForwardWithStats computes layer normalization and returns normalized
output and statistics

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

#### func  LogAutodiffDebug

```go
func LogAutodiffDebug(level DebugLogLevel, message string, args ...interface{})
```
LogDebug logs a debug message with the specified level

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

#### func  OptimizeBatchedTransfer

```go
func OptimizeBatchedTransfer(tensors []*tensor.Tensor, toGPU bool, operation string) error
```
OptimizeBatchedTransfer optimizes a batch of tensor transfers using the global
optimizer

#### func  OptimizeComputationGraph

```go
func OptimizeComputationGraph(graph *ComputationGraph) error
```
OptimizeComputationGraph applies advanced optimizations to a computation graph

#### func  OptimizeMemoryUsage

```go
func OptimizeMemoryUsage() error
```
OptimizeMemoryUsage performs global memory optimization

#### func  OptimizeTensorTransfer

```go
func OptimizeTensorTransfer(t *tensor.Tensor, toGPU bool, operation string) error
```
OptimizeTensorTransfer optimizes a single tensor transfer using the global
optimizer

#### func  OptimizedAdd

```go
func OptimizedAdd(a, b *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedAdd performs optimized element-wise addition with broadcasting

#### func  OptimizedBatchMatMul

```go
func OptimizedBatchMatMul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedBatchMatMul performs optimized batch matrix multiplication

#### func  OptimizedConv1x1

```go
func OptimizedConv1x1(input, weight *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedConv1x1 performs optimized 1x1 convolution (pointwise convolution)

#### func  OptimizedDepthwiseConv

```go
func OptimizedDepthwiseConv(input, kernel *tensor.Tensor, strideH, strideW, padH, padW int) (*tensor.Tensor, error)
```
OptimizedDepthwiseConv performs optimized depthwise convolution

#### func  OptimizedDivide

```go
func OptimizedDivide(a, b *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedDivide performs optimized element-wise division with broadcasting

#### func  OptimizedElementwiseBinaryOp

```go
func OptimizedElementwiseBinaryOp(a, b *tensor.Tensor, opType int) (*tensor.Tensor, error)
```
OptimizedElementwiseBinaryOp performs optimized elementwise binary operations
with broadcasting

#### func  OptimizedGEMM

```go
func OptimizedGEMM(A, B *tensor.Tensor, alpha, beta float32) (*tensor.Tensor, error)
```
OptimizedGEMM performs optimized General Matrix Multiplication with tiling and
shared memory

#### func  OptimizedLayerNorm

```go
func OptimizedLayerNorm(input, gamma, beta *tensor.Tensor, epsilon float32) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error)
```
OptimizedLayerNorm performs optimized layer normalization

#### func  OptimizedMax

```go
func OptimizedMax(input *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedMax performs optimized maximum reduction

#### func  OptimizedMin

```go
func OptimizedMin(input *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedMin performs optimized minimum reduction

#### func  OptimizedMultiply

```go
func OptimizedMultiply(a, b *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedMultiply performs optimized element-wise multiplication with
broadcasting

#### func  OptimizedReduce

```go
func OptimizedReduce(input *tensor.Tensor, opType int) (*tensor.Tensor, error)
```
OptimizedReduce performs optimized reduction operations

#### func  OptimizedSoftmax

```go
func OptimizedSoftmax(input *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedSoftmax performs numerically stable optimized softmax

#### func  OptimizedSubtract

```go
func OptimizedSubtract(a, b *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedSubtract performs optimized element-wise subtraction with broadcasting

#### func  OptimizedSum

```go
func OptimizedSum(input *tensor.Tensor) (*tensor.Tensor, error)
```
OptimizedSum performs optimized sum reduction

#### func  OptimizedTensorAlloc

```go
func OptimizedTensorAlloc(shape []int, operation string) (*tensor.Tensor, error)
```
OptimizedTensorAlloc allocates a tensor using the optimized memory pool

#### func  Pad2D

```go
func Pad2D(input *tensor.Tensor, padTop, padBottom, padLeft, padRight int, padValue float32) (*tensor.Tensor, error)
```
Pad2D adds padding to a 2D tensor

#### func  Phase8CComplete

```go
func Phase8CComplete() bool
```
Phase8CComplete returns true if all Phase 8C components are implemented

#### func  Phase8CFeatures

```go
func Phase8CFeatures() []string
```
Phase8CFeatures returns a list of all Phase 8C features

#### func  PrecompileCommonKernels

```go
func PrecompileCommonKernels() error
```
PrecompileCommonKernels precompiles frequently used kernels

#### func  PrefetchGPUData

```go
func PrefetchGPUData(device unsafe.Pointer, bufferPtr unsafe.Pointer, size, offset int) error
```
PrefetchGPUData prefetches data to GPU cache for optimal access patterns

#### func  PrintAdvancedOptimizationStats

```go
func PrintAdvancedOptimizationStats()
```
PrintAdvancedOptimizationStats prints detailed optimization statistics

#### func  PrintConvParams

```go
func PrintConvParams(params Conv2DParams)
```
PrintConvParams prints convolution parameters

#### func  PrintGradientUtilitiesStats

```go
func PrintGradientUtilitiesStats()
```
PrintGradientUtilitiesStats prints detailed gradient utilities statistics

#### func  PrintMemoryEfficientStats

```go
func PrintMemoryEfficientStats()
```
PrintMemoryEfficientStats prints detailed memory efficiency statistics

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

#### func  ProfiledBackwardOperation

```go
func ProfiledBackwardOperation(opType OpType, backwardOp func() error) error
```
ProfiledBackwardOperation wraps a backward operation with profiling

#### func  ProfiledOperation

```go
func ProfiledOperation(opType OpType, operation func() error) error
```
ProfiledOperation wraps an operation with profiling

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

#### func  ReduceSum

```go
func ReduceSum(t *tensor.Tensor, axes []int, keepDims bool) (*tensor.Tensor, error)
```
ReduceSum reduces a tensor by summing along specified axes

#### func  ReflectionPad2D

```go
func ReflectionPad2D(input *tensor.Tensor, padding int) (*tensor.Tensor, error)
```
ReflectionPad2D adds reflection padding to a tensor

#### func  RegisterAnomalyCallback

```go
func RegisterAnomalyCallback(callback AnomalyCallback)
```
RegisterAnomalyCallback registers a callback for anomaly detection

#### func  RegisterFusedOperation

```go
func RegisterFusedOperation(name string, operation FusedOperation)
```
RegisterFusedOperation registers a custom fused operation

#### func  RegisterOptimizationTechnique

```go
func RegisterOptimizationTechnique(technique OptimizationTechnique)
```
RegisterOptimizationTechnique registers a new optimization technique

#### func  ReleaseOptimizedBuffer

```go
func ReleaseOptimizedBuffer(buffer unsafe.Pointer) error
```
ReleaseOptimizedBuffer releases a GPU buffer using optimized C functions

#### func  RemoveAutodiffBreakpoint

```go
func RemoveAutodiffBreakpoint(tensor *GradientTensor)
```
RemoveBreakpoint removes a breakpoint for a specific tensor

#### func  ResetGradientUtilities

```go
func ResetGradientUtilities()
```
ResetGradientUtilities resets all gradient utilities to their initial state

#### func  ResetPhase8C

```go
func ResetPhase8C()
```
ResetPhase8C cleans up and resets all Phase 8C components (for testing)

#### func  Reshape

```go
func Reshape(t *tensor.Tensor, newShape []int) (*tensor.Tensor, error)
```
Reshape reshapes a tensor to new dimensions

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

#### func  SetGlobalFusionMemoryThreshold

```go
func SetGlobalFusionMemoryThreshold(threshold int64)
```
SetGlobalFusionMemoryThreshold sets the global memory threshold for fusion

#### func  SetGlobalFusionPerformanceThreshold

```go
func SetGlobalFusionPerformanceThreshold(threshold float32)
```
SetGlobalFusionPerformanceThreshold sets the global performance threshold for
fusion

#### func  SetGlobalMatrixDevice

```go
func SetGlobalMatrixDevice(device unsafe.Pointer)
```
SetGlobalMatrixDevice sets the global device for matrix operations

#### func  SetGradientMode

```go
func SetGradientMode(mode GradientMode)
```
SetGradientMode sets the global gradient computation mode

#### func  SetLayerPriority

```go
func SetLayerPriority(layerName string, priority float32)
```
SetLayerPriority sets the priority weight for a specific layer

#### func  SetSharedMemoryLimits

```go
func SetSharedMemoryLimits(maxSharedMemory, bankSize, warpSize int)
```
SetSharedMemoryLimits configures shared memory limits for optimization

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

#### func  SmoothGradients

```go
func SmoothGradients(gradients []*GradientTensor, layerNames []string) error
```
SmoothGradients applies smoothing to gradients

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

#### func  SwapTensorToCPU

```go
func SwapTensorToCPU(t *tensor.Tensor) error
```
SwapTensorToCPU swaps a tensor from GPU to CPU memory

#### func  SwapTensorToGPU

```go
func SwapTensorToGPU(t *tensor.Tensor) error
```
SwapTensorToGPU swaps a tensor from CPU to GPU memory

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

#### func  TrackMemoryAccess

```go
func TrackMemoryAccess(ptr unsafe.Pointer, accessType AccessType)
```
TrackMemoryAccess tracks access to allocated memory

#### func  TrackMemoryAllocation

```go
func TrackMemoryAllocation(ptr unsafe.Pointer, size int64, purpose string, tensor *tensor.Tensor)
```
TrackMemoryAllocation tracks a memory allocation

#### func  TrackMemoryDeallocation

```go
func TrackMemoryDeallocation(ptr unsafe.Pointer)
```
TrackMemoryDeallocation tracks a memory deallocation

#### func  TrackTensorAllocation

```go
func TrackTensorAllocation(t *tensor.Tensor, canEvict bool)
```
TrackTensorAllocation tracks a tensor allocation

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

#### func  UpdateTensorAccess

```go
func UpdateTensorAccess(t *tensor.Tensor)
```
UpdateTensorAccess updates tensor access information

#### func  ValidateDataLoaderConfig

```go
func ValidateDataLoaderConfig(config DataLoaderConfig) error
```
ValidateDataLoaderConfig validates data loader configuration

#### func  ValidateGradients

```go
func ValidateGradients(model func(*tensor.Tensor) (*tensor.Tensor, error), input *tensor.Tensor, epsilon float32) error
```
ValidateGradients performs gradient validation using finite differences

#### func  ValidatePhase8CImplementation

```go
func ValidatePhase8CImplementation() bool
```
ValidatePhase8CImplementation validates that all Phase 8C features are working
correctly

#### func  WithMemoryEfficientContext

```go
func WithMemoryEfficientContext(config *MemoryEfficientConfig, fn func() error) error
```
WithMemoryEfficientContext executes a function with a specific memory-efficient
context

#### func  WithMixedPrecision

```go
func WithMixedPrecision(lossScaling float32, fn func() error) error
```
WithMixedPrecision executes a function with mixed precision enabled

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

#### type AccessOrder

```go
type AccessOrder int
```

AccessOrder defines how tensor elements are accessed

```go
const (
	AccessOrderRowMajor AccessOrder = iota
	AccessOrderColumnMajor
	AccessOrderTiled
	AccessOrderBlocked
	AccessOrderStrided
	AccessOrderRandom
)
```

#### type AccessPattern

```go
type AccessPattern struct {
	OperationType   string
	TensorShapes    [][]int
	AccessOrder     AccessOrder
	StridePattern   []int
	AccessFrequency int64
	LastAccess      time.Time
	IsCoalesced     bool
	BankConflicts   int
}
```

AccessPattern describes how memory is accessed during an operation

#### func  AnalyzeMemoryAccessPattern

```go
func AnalyzeMemoryAccessPattern(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*AccessPattern, error)
```
AnalyzeMemoryAccessPattern analyzes memory access patterns using the global
optimizer

#### type AccessPatternOptimizer

```go
type AccessPatternOptimizer struct {
}
```

AccessPatternOptimizer optimizes based on access patterns

#### func (*AccessPatternOptimizer) GetHint

```go
func (apo *AccessPatternOptimizer) GetHint(bufferName string) int
```
GetHint returns optimization hint based on patterns

#### func (*AccessPatternOptimizer) UpdatePattern

```go
func (apo *AccessPatternOptimizer) UpdatePattern(bufferName string, accessType int)
```
UpdatePattern updates access pattern statistics

#### type AccessRecord

```go
type AccessRecord struct {
	BufferName string
	AccessType int // 0: CPU read, 1: CPU write, 2: GPU read, 3: GPU write
	Timestamp  int64
	Size       int
}
```

AccessRecord tracks memory access patterns

#### type AccessType

```go
type AccessType int
```

AccessType represents types of memory access

```go
const (
	ReadAccess AccessType = iota
	WriteAccess
	ReadWriteAccess
)
```

#### type AccumulationBuffer

```go
type AccumulationBuffer struct {
}
```

AccumulationBuffer stores accumulated gradients

#### type AccumulationType

```go
type AccumulationType int
```

AccumulationType represents different accumulation strategies

```go
const (
	SimpleAccumulation AccumulationType = iota
	WeightedAccumulation
	AdaptiveAccumulation
	PriorityAccumulation
)
```

#### type AcquisitionFunction

```go
type AcquisitionFunction int
```

AcquisitionFunction defines acquisition function types

```go
const (
	ExpectedImprovement AcquisitionFunction = iota
	UpperConfidenceBound
	ProbabilityOfImprovement
)
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

#### type AdaptiveMemoryManager

```go
type AdaptiveMemoryManager struct {
}
```

AdaptiveMemoryManager automatically adjusts memory settings based on usage
patterns

#### func  NewAdaptiveMemoryManager

```go
func NewAdaptiveMemoryManager() *AdaptiveMemoryManager
```
NewAdaptiveMemoryManager creates a new adaptive memory manager

#### func (*AdaptiveMemoryManager) AdaptMemorySettings

```go
func (amm *AdaptiveMemoryManager) AdaptMemorySettings()
```
AdaptMemorySettings adapts memory settings based on usage patterns

#### func (*AdaptiveMemoryManager) RecordMemoryUsage

```go
func (amm *AdaptiveMemoryManager) RecordMemoryUsage(usage int64)
```
RecordMemoryUsage records current memory usage

#### func (*AdaptiveMemoryManager) RecordPerformance

```go
func (amm *AdaptiveMemoryManager) RecordPerformance(performance float32)
```
RecordPerformance records performance metric

#### type AdaptivePrecisionManager

```go
type AdaptivePrecisionManager struct {
}
```

AdaptivePrecisionManager automatically selects optimal precision based on
operation characteristics

#### func  NewAdaptivePrecisionManager

```go
func NewAdaptivePrecisionManager(config *MixedPrecisionConfig) (*AdaptivePrecisionManager, error)
```
NewAdaptivePrecisionManager creates an adaptive precision manager

#### func (*AdaptivePrecisionManager) Cleanup

```go
func (apm *AdaptivePrecisionManager) Cleanup()
```
Cleanup releases resources

#### func (*AdaptivePrecisionManager) GetRecommendation

```go
func (apm *AdaptivePrecisionManager) GetRecommendation(matrixSize int) string
```
GetRecommendation provides a recommendation for a given matrix size

#### func (*AdaptivePrecisionManager) LearnFromBenchmark

```go
func (apm *AdaptivePrecisionManager) LearnFromBenchmark(benchmark *PrecisionBenchmark)
```
LearnFromBenchmark updates thresholds based on benchmark results

#### func (*AdaptivePrecisionManager) OptimalMatMul

```go
func (apm *AdaptivePrecisionManager) OptimalMatMul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
OptimalMatMul automatically chooses the best precision strategy

#### type AdaptivePrecisionSelector

```go
type AdaptivePrecisionSelector struct {
	// Thresholds for automatic precision selection
	SmallMatrixThreshold int     // Use mixed precision for matrices smaller than this
	AccuracyThreshold    float64 // Minimum acceptable accuracy
	PerformanceThreshold float64 // Minimum speedup required for mixed precision
}
```

AdaptivePrecisionSelector automatically chooses the best precision based on
operation characteristics

#### func  NewAdaptivePrecisionSelector

```go
func NewAdaptivePrecisionSelector() *AdaptivePrecisionSelector
```
NewAdaptivePrecisionSelector creates a new adaptive precision selector

#### func (*AdaptivePrecisionSelector) ShouldUseMixedPrecision

```go
func (aps *AdaptivePrecisionSelector) ShouldUseMixedPrecision(matrixSize int, operationType string) bool
```
ShouldUseMixedPrecision determines if mixed precision should be used for a given
operation

#### func (*AdaptivePrecisionSelector) UpdatePerformanceHistory

```go
func (aps *AdaptivePrecisionSelector) UpdatePerformanceHistory(matrixSize int, speedup, accuracy float64)
```
UpdatePerformanceHistory updates the performance history with new measurements

#### type AdvancedMemoryOptimizer

```go
type AdvancedMemoryOptimizer struct {
}
```

AdvancedMemoryOptimizer provides advanced memory optimization

#### func  NewAdvancedMemoryOptimizer

```go
func NewAdvancedMemoryOptimizer() *AdvancedMemoryOptimizer
```
NewAdvancedMemoryOptimizer creates a new advanced memory optimizer

#### type AdvancedOptimizer

```go
type AdvancedOptimizer struct {
}
```

AdvancedOptimizer provides advanced optimization techniques

#### func  NewAdvancedOptimizer

```go
func NewAdvancedOptimizer() *AdvancedOptimizer
```
NewAdvancedOptimizer creates a new advanced optimizer

#### type AllocationEvent

```go
type AllocationEvent struct {
}
```

AllocationEvent represents an allocation event

#### type AllocationEventType

```go
type AllocationEventType int
```

AllocationEventType represents types of allocation events

```go
const (
	AllocationEventConst AllocationEventType = iota
	DeallocationEvent
	AccessEvent
	MigrationEvent
)
```

#### type AllocationInfo

```go
type AllocationInfo struct {
}
```

AllocationInfo tracks tensor allocation information

#### type AllocationTracker

```go
type AllocationTracker struct {
}
```

AllocationTracker tracks memory allocations

#### type AnomalyCallback

```go
type AnomalyCallback func(anomaly GradientAnomaly)
```

AnomalyCallback is called when an anomaly is detected

#### type AsyncDataLoader

```go
type AsyncDataLoader struct {
}
```

AsyncDataLoader provides asynchronous data loading with prefetching

#### func  NewAsyncDataLoader

```go
func NewAsyncDataLoader(dataset Dataset, config DataLoaderConfig) (*AsyncDataLoader, error)
```
NewAsyncDataLoader creates a new asynchronous data loader

#### func (*AsyncDataLoader) BatchCount

```go
func (adl *AsyncDataLoader) BatchCount() int
```
BatchCount returns the number of batches (implements DataLoader interface)

#### func (*AsyncDataLoader) GetBatch

```go
func (adl *AsyncDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
```
GetBatch returns the next batch (implements DataLoader interface)

#### func (*AsyncDataLoader) GetDatasetSize

```go
func (adl *AsyncDataLoader) GetDatasetSize() int
```
GetDatasetSize returns the dataset size (implements DataLoader interface)

#### func (*AsyncDataLoader) GetStats

```go
func (adl *AsyncDataLoader) GetStats() DataLoaderStats
```
GetStats returns data loader statistics

#### func (*AsyncDataLoader) Reset

```go
func (adl *AsyncDataLoader) Reset() error
```
Reset resets the data loader state (implements DataLoader interface)

#### func (*AsyncDataLoader) SetBatchSize

```go
func (adl *AsyncDataLoader) SetBatchSize(batchSize int)
```
SetBatchSize sets the batch size (implements DataLoader interface)

#### func (*AsyncDataLoader) Shuffle

```go
func (adl *AsyncDataLoader) Shuffle() error
```
Shuffle shuffles the dataset indices (implements DataLoader interface)

#### func (*AsyncDataLoader) Start

```go
func (adl *AsyncDataLoader) Start() error
```
Start begins asynchronous data loading

#### func (*AsyncDataLoader) Stop

```go
func (adl *AsyncDataLoader) Stop()
```
Stop stops the asynchronous data loading

#### type AutoKernelSelector

```go
type AutoKernelSelector struct {
}
```

AutoKernelSelector selects optimal kernels based on input characteristics

#### func  NewAutoKernelSelector

```go
func NewAutoKernelSelector(mgm *MPSGraphManager) *AutoKernelSelector
```
NewAutoKernelSelector creates a kernel selector

#### func (*AutoKernelSelector) SelectKernel

```go
func (aks *AutoKernelSelector) SelectKernel(op Operation, inputShapes [][]int) string
```
SelectKernel chooses the optimal kernel for an operation

#### type AutoTuner

```go
type AutoTuner struct {
}
```

AutoTuner automatically optimizes performance parameters

#### func  NewAutoTuner

```go
func NewAutoTuner(profiler *PerformanceProfiler) *AutoTuner
```
NewAutoTuner creates a new auto-tuner

#### func (*AutoTuner) ProcessMetric

```go
func (at *AutoTuner) ProcessMetric(metric Metric)
```
ProcessMetric processes a performance metric for auto-tuning

#### func (*AutoTuner) RegisterParameter

```go
func (at *AutoTuner) RegisterParameter(param *TuningParameter)
```
RegisterParameter adds a tuning parameter

#### func (*AutoTuner) TuneParameters

```go
func (at *AutoTuner) TuneParameters(maxTrials int) PerformanceMetrics
```
TuneParameters optimizes all registered parameters

#### type AutocastMode

```go
type AutocastMode int
```

AutocastMode defines when to use float16 vs float32

```go
const (
	AutocastDisabled AutocastMode = iota
	AutocastForward               // Use float16 for forward pass only
	AutocastFull                  // Use float16 for forward and backward (with scaling)
)
```

#### type AutodiffConfig

```go
type AutodiffConfig struct {
	EnableProfiler     bool
	EnableDebugger     bool
	EnableOptimizer    bool
	MaxGraphDepth      int
	MemoryOptimization bool
	FusionOptimization bool
	ProfilingLevel     int
	DebugLevel         DebugLogLevel
}
```

AutodiffConfig provides global configuration for autodiff behavior

#### func  DefaultAutodiffConfig

```go
func DefaultAutodiffConfig() *AutodiffConfig
```
DefaultAutodiffConfig returns a default configuration

#### type AutodiffDebugger

```go
type AutodiffDebugger struct {
}
```

AutodiffDebugger provides debugging tools for automatic differentiation

#### func  NewAutodiffDebugger

```go
func NewAutodiffDebugger() *AutodiffDebugger
```
NewAutodiffDebugger creates a new autodiff debugger

#### func (*AutodiffDebugger) CheckBreakpoint

```go
func (d *AutodiffDebugger) CheckBreakpoint(tensor *GradientTensor) bool
```
CheckBreakpoint checks if a breakpoint should trigger for a tensor

#### type AutodiffMemoryManager

```go
type AutodiffMemoryManager struct {
}
```

AutodiffMemoryManager handles memory allocation and deallocation

#### type AutodiffMemoryStats

```go
type AutodiffMemoryStats struct {
	SavedTensorsMemory  int64
	GradientMemory      int64
	PeakGraphMemory     int64
	TotalAllocations    int64
	ActiveTensors       int64
	MemoryFragmentation float32
}
```

AutodiffMemoryStats tracks memory usage during autodiff

#### type AutodiffProfiler

```go
type AutodiffProfiler struct {
}
```

AutodiffProfiler provides detailed profiling and analysis of the computation
graph

#### func  GetAutodiffProfile

```go
func GetAutodiffProfile() *AutodiffProfiler
```
GetAutodiffProfile returns the current profiling results

#### func  NewAutodiffProfiler

```go
func NewAutodiffProfiler() *AutodiffProfiler
```
NewAutodiffProfiler creates a new autodiff profiler

#### func (*AutodiffProfiler) PrintProfile

```go
func (p *AutodiffProfiler) PrintProfile()
```
PrintProfile prints a detailed profiling report

#### type AutomaticCodeGenerator

```go
type AutomaticCodeGenerator struct {
}
```

AutomaticCodeGenerator generates optimized code for gradient computations

#### func  NewAutomaticCodeGenerator

```go
func NewAutomaticCodeGenerator() *AutomaticCodeGenerator
```
NewAutomaticCodeGenerator creates a new code generator

#### type BatchData

```go
type BatchData struct {
}
```

BatchData represents a loaded batch with metadata

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

#### type BatchOptimizer

```go
type BatchOptimizer struct {
}
```

BatchOptimizer optimizes memory usage for batch processing

#### func (*BatchOptimizer) GetBatchBuffer

```go
func (bo *BatchOptimizer) GetBatchBuffer() (*SharedBuffer, error)
```
GetBatchBuffer gets or creates an optimized buffer for batch processing

#### func (*BatchOptimizer) ReturnBatchBuffer

```go
func (bo *BatchOptimizer) ReturnBatchBuffer(buf *SharedBuffer)
```
ReturnBatchBuffer returns a buffer to the pool

#### type BatchRecord

```go
type BatchRecord struct {
	Size       int
	Latency    time.Duration
	Throughput float64
	Timestamp  time.Time
}
```

BatchRecord stores batch execution history

#### type BatchSizePredictor

```go
type BatchSizePredictor struct {
}
```

BatchSizePredictor predicts optimal batch sizes

#### func  NewBatchSizePredictor

```go
func NewBatchSizePredictor() *BatchSizePredictor
```
NewBatchSizePredictor creates a batch size predictor

#### func (*BatchSizePredictor) PredictOptimalSize

```go
func (bsp *BatchSizePredictor) PredictOptimalSize() int
```
PredictOptimalSize predicts the optimal batch size

#### func (*BatchSizePredictor) RecordBatch

```go
func (bsp *BatchSizePredictor) RecordBatch(size int, latency time.Duration, throughput float64)
```
RecordBatch adds a batch execution record

#### type BatchedTransfer

```go
type BatchedTransfer struct {
	Tensors   []*tensor.Tensor
	ToGPU     bool
	Operation string
}
```

BatchedTransfer represents a batch of tensor transfers

#### type BayesianOptimizer

```go
type BayesianOptimizer struct {
}
```

BayesianOptimizer implements Bayesian optimization

#### func  NewBayesianOptimizer

```go
func NewBayesianOptimizer() *BayesianOptimizer
```
NewBayesianOptimizer creates a Bayesian optimizer

#### func (*BayesianOptimizer) SuggestConfiguration

```go
func (bo *BayesianOptimizer) SuggestConfiguration(params map[string]*TuningParameter) map[string]interface{}
```
SuggestConfiguration suggests next configuration to try

#### func (*BayesianOptimizer) UpdateTrial

```go
func (bo *BayesianOptimizer) UpdateTrial(config map[string]interface{}, score float64)
```
UpdateTrial updates the optimizer with trial results

#### type BenchmarkResult

```go
type BenchmarkResult struct {
	Name                string
	BaselineTime        time.Duration
	OptimizedTime       time.Duration
	Speedup             float64
	MemoryBaseline      int64
	MemoryOptimized     int64
	MemorySavings       float64
	ThroughputBaseline  float64 // Operations per second
	ThroughputOptimized float64
	Notes               []string
}
```

BenchmarkResult stores the results of a benchmark

#### type BenchmarkSuite

```go
type BenchmarkSuite struct {
}
```

BenchmarkSuite runs comprehensive benchmarks for Phase 8C optimizations

#### func  NewBenchmarkSuite

```go
func NewBenchmarkSuite() *BenchmarkSuite
```
NewBenchmarkSuite creates a new benchmark suite

#### func  RunPhase8CBenchmarks

```go
func RunPhase8CBenchmarks() *BenchmarkSuite
```
RunPhase8CBenchmarks runs a complete benchmark suite for Phase 8C

#### func (*BenchmarkSuite) BenchmarkBufferReuse

```go
func (bs *BenchmarkSuite) BenchmarkBufferReuse()
```
BenchmarkBufferReuse benchmarks buffer reuse system performance

#### func (*BenchmarkSuite) BenchmarkConvolution

```go
func (bs *BenchmarkSuite) BenchmarkConvolution()
```
BenchmarkConvolution benchmarks convolution optimizations

#### func (*BenchmarkSuite) BenchmarkKernelCache

```go
func (bs *BenchmarkSuite) BenchmarkKernelCache()
```
BenchmarkKernelCache benchmarks kernel compilation caching

#### func (*BenchmarkSuite) BenchmarkMatrixMultiplication

```go
func (bs *BenchmarkSuite) BenchmarkMatrixMultiplication(sizes []int)
```
BenchmarkMatrixMultiplication benchmarks matrix multiplication optimizations

#### func (*BenchmarkSuite) BenchmarkMemoryTransfer

```go
func (bs *BenchmarkSuite) BenchmarkMemoryTransfer()
```
BenchmarkMemoryTransfer benchmarks CPU-GPU transfer optimization

#### func (*BenchmarkSuite) PrintResults

```go
func (bs *BenchmarkSuite) PrintResults()
```
PrintResults prints a detailed report of all benchmark results

#### func (*BenchmarkSuite) RunAllBenchmarks

```go
func (bs *BenchmarkSuite) RunAllBenchmarks()
```
RunAllBenchmarks runs the complete benchmark suite

#### type BufferPool

```go
type BufferPool struct {
}
```

BufferPool manages reusable tensor buffers

#### func  NewBufferPool

```go
func NewBufferPool(maxMemory int64) *BufferPool
```
NewBufferPool creates a new buffer pool

#### func (*BufferPool) GetBuffer

```go
func (bp *BufferPool) GetBuffer(shape []int) (*tensor.Tensor, error)
```
GetBuffer retrieves a buffer from the pool or creates a new one

#### func (*BufferPool) ReturnBuffer

```go
func (bp *BufferPool) ReturnBuffer(buffer *tensor.Tensor)
```
ReturnBuffer returns a buffer to the pool

#### type BufferReuseManager

```go
type BufferReuseManager struct {
}
```

BufferReuseManager manages reusable buffers for intermediate tensor operations

#### func  GetGlobalBufferReuseManager

```go
func GetGlobalBufferReuseManager() *BufferReuseManager
```
GetGlobalBufferReuseManager returns the global buffer reuse manager

#### func  NewBufferReuseManager

```go
func NewBufferReuseManager(memoryPool *GPUMemoryPool) *BufferReuseManager
```
NewBufferReuseManager creates a new buffer reuse manager

#### func (*BufferReuseManager) Close

```go
func (brm *BufferReuseManager) Close()
```
Close stops the buffer reuse manager and releases all resources

#### func (*BufferReuseManager) GetBuffer

```go
func (brm *BufferReuseManager) GetBuffer(shape []int, operation string) (*ReusableBuffer, error)
```
GetBuffer gets a reusable buffer for a tensor operation

#### func (*BufferReuseManager) GetStats

```go
func (brm *BufferReuseManager) GetStats() map[string]*BufferStats
```
GetStats returns buffer reuse statistics

#### func (*BufferReuseManager) GetTensorBuffer

```go
func (brm *BufferReuseManager) GetTensorBuffer(tensor *tensor.Tensor, operation string) (*ReusableBuffer, error)
```
GetTensorBuffer gets a buffer specifically for a tensor and tracks it

#### func (*BufferReuseManager) ReleaseTensorBuffer

```go
func (brm *BufferReuseManager) ReleaseTensorBuffer(tensor *tensor.Tensor)
```
ReleaseTensorBuffer releases a buffer associated with a tensor

#### func (*BufferReuseManager) ReturnBuffer

```go
func (brm *BufferReuseManager) ReturnBuffer(buffer *ReusableBuffer)
```
ReturnBuffer returns a buffer to the pool for reuse

#### type BufferStats

```go
type BufferStats struct {
	TotalAllocations int64         // Total allocations in this category
	TotalReuses      int64         // Total reuses
	TotalHits        int64         // Cache hits
	TotalMisses      int64         // Cache misses
	AverageLifetime  time.Duration // Average buffer lifetime
	PeakUsage        int           // Peak number of buffers in use
	CurrentUsage     int           // Current number of buffers in use
}
```

BufferStats tracks usage statistics for buffer categories

#### type CPUGPUSwapManager

```go
type CPUGPUSwapManager struct {
}
```

CPUGPUSwapManager handles swapping between CPU and GPU memory

#### type CacheKey

```go
type CacheKey struct {
	OperationType OpType
	InputShapes   string
	Order         int
	Parameters    string
}
```

CacheKey uniquely identifies a derivative computation

#### type CachedDerivative

```go
type CachedDerivative struct {
}
```

CachedDerivative stores a cached derivative

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

#### type CheckpointData

```go
type CheckpointData struct {
}
```

CheckpointData stores checkpointed computation information

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

#### type CheckpointingStrategy

```go
type CheckpointingStrategy int
```

CheckpointingStrategy defines different checkpointing strategies

```go
const (
	NoCheckpointing CheckpointingStrategy = iota
	UniformCheckpointing
	AdaptiveCheckpointing
	MemoryAwareCheckpointing
)
```

#### type ClippingEvent

```go
type ClippingEvent struct {
	Timestamp      time.Time
	OriginalNorm   float32
	ClippedNorm    float32
	ClipRatio      float32
	LayersAffected int
}
```

ClippingEvent records a gradient clipping event

#### type CoalescingStatistics

```go
type CoalescingStatistics struct {
	TotalOptimizations   int64
	SuccessfulCoalescing int64
	BankConflictsReduced int64
	AverageSpeedup       float64
	TotalMemoryAccesses  int64
	CoalescedAccesses    int64
}
```

CoalescingStatistics tracks memory coalescing performance

#### type CoalescingStrategy

```go
type CoalescingStrategy struct {
	OperationType     string
	RecommendedLayout TensorLayout
	RecommendedStride []int
	TileSize          []int
	BlockSize         []int
	PaddingRequired   []int
	ExpectedSpeedup   float64
	MemoryOverhead    float64
}
```

CoalescingStrategy defines how to optimize memory access for coalescing

#### func  OptimizeMemoryCoalescing

```go
func OptimizeMemoryCoalescing(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*CoalescingStrategy, error)
```
OptimizeMemoryCoalescing optimizes memory coalescing for an operation using the
global optimizer

#### type CodeTemplate

```go
type CodeTemplate struct {
	OperationType    OpType
	ForwardTemplate  string
	BackwardTemplate string
	Requirements     []string
	Optimizations    map[string]string
}
```

CodeTemplate defines how to generate code for an operation

#### type CommandQueuePool

```go
type CommandQueuePool struct {
}
```

CommandQueuePool manages a pool of reusable Metal command queues

#### func  GetGlobalCommandQueuePool

```go
func GetGlobalCommandQueuePool(device unsafe.Pointer) *CommandQueuePool
```
GetGlobalCommandQueuePool returns the singleton command queue pool

#### func  NewCommandQueuePool

```go
func NewCommandQueuePool(device unsafe.Pointer, poolSize int) *CommandQueuePool
```
NewCommandQueuePool creates a new pool of command queues

#### func (*CommandQueuePool) Available

```go
func (p *CommandQueuePool) Available() int
```
Available returns the number of available queues

#### func (*CommandQueuePool) GetQueue

```go
func (p *CommandQueuePool) GetQueue() unsafe.Pointer
```
GetQueue gets a command queue from the pool (blocks if none available)

#### func (*CommandQueuePool) GetQueueNonBlocking

```go
func (p *CommandQueuePool) GetQueueNonBlocking() unsafe.Pointer
```
GetQueueNonBlocking tries to get a queue without blocking

#### func (*CommandQueuePool) InUse

```go
func (p *CommandQueuePool) InUse() int
```
InUse returns the number of queues currently in use

#### func (*CommandQueuePool) ReturnQueue

```go
func (p *CommandQueuePool) ReturnQueue(queue unsafe.Pointer)
```
ReturnQueue returns a command queue to the pool

#### func (*CommandQueuePool) Size

```go
func (p *CommandQueuePool) Size() int
```
Size returns the total number of queues in the pool

#### type CommunicationCost

```go
type CommunicationCost struct {
	Latency   time.Duration
	Bandwidth int64
	Overhead  float32
}
```

CommunicationCost represents communication costs in distributed settings

#### type CompilationCache

```go
type CompilationCache struct {
}
```

CompilationCache caches compiled functions

#### type CompiledOperation

```go
type CompiledOperation struct {
	Graph       unsafe.Pointer
	Executable  unsafe.Pointer
	InputShapes [][]int
	OutputShape []int
	CacheKey    string
}
```

CompiledOperation represents a compiled MPS graph operation

#### type CompressedTensor

```go
type CompressedTensor struct {
}
```

CompressedTensor represents a compressed tensor

#### func  CompressGradient

```go
func CompressGradient(grad *tensor.Tensor, method string) (*CompressedTensor, error)
```
CompressGradient compresses a gradient tensor using available compression
methods

#### type CompressionBuffer

```go
type CompressionBuffer struct {
}
```

CompressionBuffer manages compression workspace

#### type CompressionEngine

```go
type CompressionEngine struct {
}
```

CompressionEngine handles tensor compression

#### type CompressionMethod

```go
type CompressionMethod interface {
	GetName() string
	Compress(tensor *tensor.Tensor) (*CompressedTensor, error)
	Decompress(compressed *CompressedTensor) (*tensor.Tensor, error)
	GetCompressionRatio() float32
	GetDecompressionSpeed() float32
}
```

CompressionMethod represents a compression method

#### type CompressionStatistics

```go
type CompressionStatistics struct {
}
```

CompressionStatistics tracks compression performance

#### type CompressionStats

```go
type CompressionStats struct {
	TotalCompressed   int64
	TotalDecompressed int64
	CompressionRatio  float32
	CompressionTime   time.Duration
	DecompressionTime time.Duration
	AccuracyLoss      float32
}
```

CompressionStats tracks compression statistics

#### type ComputationCostModel

```go
type ComputationCostModel struct {
}
```

ComputationCostModel models the cost of operations

#### func  NewComputationCostModel

```go
func NewComputationCostModel() *ComputationCostModel
```
NewComputationCostModel creates a new computation cost model

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

#### type ComputeNode

```go
type ComputeNode struct {
	ID        string
	Operation func(*tensor.Tensor) *tensor.Tensor
	Input     *tensor.Tensor
	Output    *tensor.Tensor
	StreamID  StreamID
	Status    int32 // 0: pending, 1: running, 2: complete
}
```

ComputeNode represents a node in the computation graph

#### type ConstantFolding

```go
type ConstantFolding struct{}
```

ConstantFolding pre-computes constant expressions

#### func (*ConstantFolding) Apply

```go
func (cf *ConstantFolding) Apply(graph *ComputationGraph) error
```

#### func (*ConstantFolding) GetBenefit

```go
func (cf *ConstantFolding) GetBenefit() OptimizationBenefit
```

#### func (*ConstantFolding) GetDescription

```go
func (cf *ConstantFolding) GetDescription() string
```

#### func (*ConstantFolding) GetName

```go
func (cf *ConstantFolding) GetName() string
```

#### func (*ConstantFolding) IsApplicable

```go
func (cf *ConstantFolding) IsApplicable(graph *ComputationGraph) bool
```

#### type Conv2DLayer

```go
type Conv2DLayer struct {
	// Parameters
	Weight *tensor.Tensor
	Bias   *tensor.Tensor

	// Gradients
	WeightGrad *tensor.Tensor
	BiasGrad   *tensor.Tensor

	// Configuration
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
}
```

Conv2DLayer represents a 2D convolution layer

#### func  NewConv2DLayer

```go
func NewConv2DLayer(inChannels, outChannels, kernelSize, stride, padding int) *Conv2DLayer
```
NewConv2DLayer creates a new Conv2D layer

#### func (*Conv2DLayer) Backward

```go
func (c *Conv2DLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor
```
Backward performs the backward pass

#### func (*Conv2DLayer) Forward

```go
func (c *Conv2DLayer) Forward(input *tensor.Tensor) *tensor.Tensor
```
Forward performs the forward pass

#### func (*Conv2DLayer) GetGradients

```go
func (c *Conv2DLayer) GetGradients() []*tensor.Tensor
```
GetGradients returns the layer gradients

#### func (*Conv2DLayer) GetParameters

```go
func (c *Conv2DLayer) GetParameters() []*tensor.Tensor
```
GetParameters returns the layer parameters

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

#### type DataType

```go
type DataType int
```

DataType represents tensor data types

```go
const (
	Float32 DataType = iota
	Float16
	Int32
	Int8
	Bool
)
```

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

#### type DeadCodeElimination

```go
type DeadCodeElimination struct{}
```

DeadCodeElimination removes unused operations from the graph

#### func (*DeadCodeElimination) Apply

```go
func (dce *DeadCodeElimination) Apply(graph *ComputationGraph) error
```

#### func (*DeadCodeElimination) GetBenefit

```go
func (dce *DeadCodeElimination) GetBenefit() OptimizationBenefit
```

#### func (*DeadCodeElimination) GetDescription

```go
func (dce *DeadCodeElimination) GetDescription() string
```

#### func (*DeadCodeElimination) GetName

```go
func (dce *DeadCodeElimination) GetName() string
```

#### func (*DeadCodeElimination) IsApplicable

```go
func (dce *DeadCodeElimination) IsApplicable(graph *ComputationGraph) bool
```

#### type DebugLogLevel

```go
type DebugLogLevel int
```

DebugLogLevel represents different levels of debug logging

```go
const (
	DebugOff DebugLogLevel = iota
	DebugError
	DebugWarning
	DebugInfo
	DebugVerbose
)
```

#### type DependencyGraph

```go
type DependencyGraph struct {
}
```

DependencyGraph tracks task dependencies

#### type DependencyNode

```go
type DependencyNode struct {
}
```

DependencyNode represents a node in the dependency graph

#### type DerivativeCache

```go
type DerivativeCache struct {
}
```

DerivativeCache caches computed derivatives for efficiency

#### type DetailedProfile

```go
type DetailedProfile struct {
	MatrixSize         int
	TotalOperationTime time.Duration

	// Detailed timing breakdown
	ConversionToFP16Time   time.Duration
	ComputeTime            time.Duration
	ConversionFromFP16Time time.Duration
	CGOOverheadTime        time.Duration
	MemoryTransferTime     time.Duration
	TensorAllocationTime   time.Duration

	// Comparison baseline
	Float32BaselineTime time.Duration

	// Performance metrics
	ConversionOverheadRatio float64 // Conversion time / Total time
	CGOOverheadRatio        float64 // CGO time / Total time
	MemoryOverheadRatio     float64 // Memory transfer time / Total time

	// Bottleneck identification
	PrimaryBottleneck    string
	BottleneckPercentage float64
}
```

DetailedProfile provides granular timing breakdown

#### type DetailedProfiler

```go
type DetailedProfiler struct {
}
```

DetailedProfiler provides comprehensive performance analysis

#### func  NewDetailedProfiler

```go
func NewDetailedProfiler() (*DetailedProfiler, error)
```
NewDetailedProfiler creates a new detailed profiler

#### func (*DetailedProfiler) Cleanup

```go
func (dp *DetailedProfiler) Cleanup()
```
Cleanup releases resources

#### func (*DetailedProfiler) GenerateOptimizationReport

```go
func (dp *DetailedProfiler) GenerateOptimizationReport(sizes []int, iterations int) error
```
GenerateOptimizationReport creates a comprehensive optimization report

#### func (*DetailedProfiler) IdentifyOptimizationOpportunities

```go
func (dp *DetailedProfiler) IdentifyOptimizationOpportunities(profile *DetailedProfile) []string
```
IdentifyOptimizationOpportunities provides specific recommendations

#### func (*DetailedProfiler) ProfileConversionOverhead

```go
func (dp *DetailedProfiler) ProfileConversionOverhead(sizes []int, iterations int) error
```
ProfileConversionOverhead specifically analyzes the conversion process

#### func (*DetailedProfiler) ProfileMatrixOperation

```go
func (dp *DetailedProfiler) ProfileMatrixOperation(A, B *tensor.Tensor, iterations int) (*DetailedProfile, error)
```
ProfileMatrixOperation provides detailed breakdown of matrix operation
performance

#### type DirectionalStats

```go
type DirectionalStats struct {
	CosineSimilarities []float32
	AngleChanges       []float32
	Consistency        float32
	Stability          float32
}
```

DirectionalStats tracks gradient direction statistics

#### type DistributedCoordinator

```go
type DistributedCoordinator struct {
}
```

DistributedCoordinator coordinates data loading across distributed nodes

#### type DistributedDataLoader

```go
type DistributedDataLoader struct {
}
```

DistributedDataLoader handles distributed data loading across multiple
GPUs/nodes

#### func  NewDistributedDataLoader

```go
func NewDistributedDataLoader(config DataLoaderConfig, nodeRank, worldSize int, localLoader DataLoader) (*DistributedDataLoader, error)
```
NewDistributedDataLoader creates a new distributed data loader

#### func (*DistributedDataLoader) GetBatch

```go
func (ddl *DistributedDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
```
GetBatch returns the next batch for this distributed node

#### func (*DistributedDataLoader) GetLocalBatchCount

```go
func (ddl *DistributedDataLoader) GetLocalBatchCount() int
```
GetLocalBatchCount returns the number of batches for this node

#### type DynamicBatchScheduler

```go
type DynamicBatchScheduler struct {
}
```

DynamicBatchScheduler manages dynamic batching and scheduling

#### func  NewDynamicBatchScheduler

```go
func NewDynamicBatchScheduler(memMgr *UnifiedMemoryManager, streamMgr *StreamManager) *DynamicBatchScheduler
```
NewDynamicBatchScheduler creates a dynamic batch scheduler

#### func (*DynamicBatchScheduler) SubmitRequest

```go
func (dbs *DynamicBatchScheduler) SubmitRequest(req *Request)
```
SubmitRequest adds a request to the scheduler

#### type DynamicComputationGraph

```go
type DynamicComputationGraph struct {
}
```

DynamicComputationGraph represents a dynamic computation graph

#### func  NewDynamicComputationGraph

```go
func NewDynamicComputationGraph() *DynamicComputationGraph
```
NewDynamicComputationGraph creates a dynamic computation graph

#### func (*DynamicComputationGraph) AddEdge

```go
func (cg *DynamicComputationGraph) AddEdge(from, to string)
```
AddEdge adds a dependency edge

#### func (*DynamicComputationGraph) AddNode

```go
func (cg *DynamicComputationGraph) AddNode(node *ComputeNode)
```
AddNode adds a computation node

#### func (*DynamicComputationGraph) Execute

```go
func (cg *DynamicComputationGraph) Execute(executor *ParallelExecutor)
```
Execute runs the computation graph

#### func (*DynamicComputationGraph) Schedule

```go
func (cg *DynamicComputationGraph) Schedule() []string
```
Schedule creates an execution schedule

#### type EfficientMixedPrecisionTrainer

```go
type EfficientMixedPrecisionTrainer struct {
}
```

EfficientMixedPrecisionTrainer provides optimized mixed precision operations
that minimize GPU↔CPU transfers and unnecessary conversions

#### func  NewEfficientMixedPrecisionTrainer

```go
func NewEfficientMixedPrecisionTrainer(config *MixedPrecisionConfig) (*EfficientMixedPrecisionTrainer, error)
```
NewEfficientMixedPrecisionTrainer creates an optimized mixed precision trainer

#### func (*EfficientMixedPrecisionTrainer) Cleanup

```go
func (mp *EfficientMixedPrecisionTrainer) Cleanup()
```
Cleanup releases resources

#### func (*EfficientMixedPrecisionTrainer) EfficientMatMul

```go
func (mp *EfficientMixedPrecisionTrainer) EfficientMatMul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
EfficientMatMul performs matrix multiplication with minimal overhead mixed
precision

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

#### type FixedMixedPrecisionTrainer

```go
type FixedMixedPrecisionTrainer struct {
}
```

FixedMixedPrecisionTrainer eliminates the conversion bottleneck entirely

#### func  NewFixedMixedPrecisionTrainer

```go
func NewFixedMixedPrecisionTrainer(config *MixedPrecisionConfig) (*FixedMixedPrecisionTrainer, error)
```
NewFixedMixedPrecisionTrainer creates a trainer that eliminates conversion
overhead

#### func (*FixedMixedPrecisionTrainer) BenchmarkPrecisionStrategies

```go
func (fmp *FixedMixedPrecisionTrainer) BenchmarkPrecisionStrategies(A, B *tensor.Tensor, iterations int) (*PrecisionBenchmark, error)
```
BenchmarkPrecisionStrategies compares different precision strategies

#### func (*FixedMixedPrecisionTrainer) Cleanup

```go
func (fmp *FixedMixedPrecisionTrainer) Cleanup()
```
Cleanup releases resources

#### func (*FixedMixedPrecisionTrainer) OptimalMatMul

```go
func (fmp *FixedMixedPrecisionTrainer) OptimalMatMul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
OptimalMatMul implements intelligent precision selection with ZERO conversion
overhead

#### type FloatMP16

```go
type FloatMP16 uint16
```

FloatMP16 represents a 16-bit floating point number

#### func  Float32ToFloatMP16

```go
func Float32ToFloatMP16(f float32) FloatMP16
```
Float32ToFloatMP16 converts float32 to float16

#### type FragmentationDetector

```go
type FragmentationDetector struct {
}
```

FragmentationDetector detects memory fragmentation

#### type FunctionSignature

```go
type FunctionSignature struct {
	InputTypes  []TensorType
	OutputTypes []TensorType
	Parameters  map[string]interface{}
}
```

FunctionSignature describes the function interface

#### type FusedActivationConfig

```go
type FusedActivationConfig struct {
	ActivationType ActivationType
	Alpha          float32 // For LeakyReLU, ELU
	InPlace        bool    // Whether to perform operation in-place
}
```

FusedActivationConfig configures fused activation operations

#### type FusedAttention

```go
type FusedAttention struct{}
```

Fused Multi-Head Attention

#### func (*FusedAttention) Backward

```go
func (f *FusedAttention) Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, configInterface interface{}) ([]*tensor.Tensor, error)
```

#### func (*FusedAttention) Forward

```go
func (f *FusedAttention) Forward(inputs []*tensor.Tensor, configInterface interface{}) (*tensor.Tensor, error)
```

#### func (*FusedAttention) GetMemorySavings

```go
func (f *FusedAttention) GetMemorySavings() int64
```

#### func (*FusedAttention) GetName

```go
func (f *FusedAttention) GetName() string
```

#### func (*FusedAttention) GetSpeedup

```go
func (f *FusedAttention) GetSpeedup() float32
```

#### type FusedAttentionConfig

```go
type FusedAttentionConfig struct {
	NumHeads    int
	DropoutRate float32
	Causal      bool    // Causal (masked) attention
	Scale       float32 // Attention scale factor
}
```

FusedAttentionConfig configures fused attention operations

#### type FusedConvBNReLU

```go
type FusedConvBNReLU struct{}
```

Fused Convolution + BatchNorm + ReLU

#### func (*FusedConvBNReLU) Backward

```go
func (f *FusedConvBNReLU) Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, configInterface interface{}) ([]*tensor.Tensor, error)
```

#### func (*FusedConvBNReLU) Forward

```go
func (f *FusedConvBNReLU) Forward(inputs []*tensor.Tensor, configInterface interface{}) (*tensor.Tensor, error)
```

#### func (*FusedConvBNReLU) GetMemorySavings

```go
func (f *FusedConvBNReLU) GetMemorySavings() int64
```

#### func (*FusedConvBNReLU) GetName

```go
func (f *FusedConvBNReLU) GetName() string
```

#### func (*FusedConvBNReLU) GetSpeedup

```go
func (f *FusedConvBNReLU) GetSpeedup() float32
```

#### type FusedConvolutionConfig

```go
type FusedConvolutionConfig struct {
	Conv2DParams Conv2DParams
	BatchNorm    *FusedNormalizationConfig
	Activation   *FusedActivationConfig
	Bias         bool
	DropoutRate  float32 // 0.0 = no dropout
}
```

FusedConvolutionConfig configures fused convolution operations

#### type FusedGELUDropout

```go
type FusedGELUDropout struct{}
```

Fused GELU + Dropout

#### func (*FusedGELUDropout) Backward

```go
func (f *FusedGELUDropout) Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, configInterface interface{}) ([]*tensor.Tensor, error)
```

#### func (*FusedGELUDropout) Forward

```go
func (f *FusedGELUDropout) Forward(inputs []*tensor.Tensor, configInterface interface{}) (*tensor.Tensor, error)
```

#### func (*FusedGELUDropout) GetMemorySavings

```go
func (f *FusedGELUDropout) GetMemorySavings() int64
```

#### func (*FusedGELUDropout) GetName

```go
func (f *FusedGELUDropout) GetName() string
```

#### func (*FusedGELUDropout) GetSpeedup

```go
func (f *FusedGELUDropout) GetSpeedup() float32
```

#### type FusedLayerNormLinear

```go
type FusedLayerNormLinear struct{}
```

Fused LayerNorm + Linear

#### func (*FusedLayerNormLinear) Backward

```go
func (f *FusedLayerNormLinear) Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, configInterface interface{}) ([]*tensor.Tensor, error)
```

#### func (*FusedLayerNormLinear) Forward

```go
func (f *FusedLayerNormLinear) Forward(inputs []*tensor.Tensor, configInterface interface{}) (*tensor.Tensor, error)
```

#### func (*FusedLayerNormLinear) GetMemorySavings

```go
func (f *FusedLayerNormLinear) GetMemorySavings() int64
```

#### func (*FusedLayerNormLinear) GetName

```go
func (f *FusedLayerNormLinear) GetName() string
```

#### func (*FusedLayerNormLinear) GetSpeedup

```go
func (f *FusedLayerNormLinear) GetSpeedup() float32
```

#### type FusedLinearActivation

```go
type FusedLinearActivation struct{}
```

Fused Linear + Activation

#### func (*FusedLinearActivation) Backward

```go
func (f *FusedLinearActivation) Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, configInterface interface{}) ([]*tensor.Tensor, error)
```

#### func (*FusedLinearActivation) Forward

```go
func (f *FusedLinearActivation) Forward(inputs []*tensor.Tensor, configInterface interface{}) (*tensor.Tensor, error)
```

#### func (*FusedLinearActivation) GetMemorySavings

```go
func (f *FusedLinearActivation) GetMemorySavings() int64
```

#### func (*FusedLinearActivation) GetName

```go
func (f *FusedLinearActivation) GetName() string
```

#### func (*FusedLinearActivation) GetSpeedup

```go
func (f *FusedLinearActivation) GetSpeedup() float32
```

#### type FusedLinearConfig

```go
type FusedLinearConfig struct {
	Bias       bool
	Activation *FusedActivationConfig
	Dropout    float32 // 0.0 = no dropout
	LayerNorm  *FusedNormalizationConfig
}
```

FusedLinearConfig configures fused linear layer operations

#### type FusedNormalizationConfig

```go
type FusedNormalizationConfig struct {
	NormType     BatchNormType
	Epsilon      float32
	Momentum     float32
	Training     bool
	FuseWithReLU bool // Fuse with ReLU activation
}
```

FusedNormalizationConfig configures fused normalization operations

#### type FusedOperation

```go
type FusedOperation interface {
	Forward(inputs []*tensor.Tensor, config interface{}) (*tensor.Tensor, error)
	Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, config interface{}) ([]*tensor.Tensor, error)
	GetName() string
	GetMemorySavings() int64
	GetSpeedup() float32
}
```

FusedOperation interface for all fused operations

#### func  GetFusedOperation

```go
func GetFusedOperation(name string) (FusedOperation, bool)
```
GetFusedOperation retrieves a fused operation by name

#### type FusedResidualBlock

```go
type FusedResidualBlock struct{}
```

Fused Residual Block (Conv + BN + ReLU + Conv + BN + Add + ReLU)

#### func (*FusedResidualBlock) Backward

```go
func (f *FusedResidualBlock) Backward(gradOutput *tensor.Tensor, savedTensors []*tensor.Tensor, configInterface interface{}) ([]*tensor.Tensor, error)
```

#### func (*FusedResidualBlock) Forward

```go
func (f *FusedResidualBlock) Forward(inputs []*tensor.Tensor, configInterface interface{}) (*tensor.Tensor, error)
```

#### func (*FusedResidualBlock) GetMemorySavings

```go
func (f *FusedResidualBlock) GetMemorySavings() int64
```

#### func (*FusedResidualBlock) GetName

```go
func (f *FusedResidualBlock) GetName() string
```

#### func (*FusedResidualBlock) GetSpeedup

```go
func (f *FusedResidualBlock) GetSpeedup() float32
```

#### type FusionOptimizer

```go
type FusionOptimizer struct {
}
```

FusionOptimizer identifies and fuses operations

#### func  NewFusionOptimizer

```go
func NewFusionOptimizer() *FusionOptimizer
```
NewFusionOptimizer creates a fusion optimizer

#### func (*FusionOptimizer) OptimizeOperations

```go
func (fo *FusionOptimizer) OptimizeOperations(ops []Operation) []Operation
```
OptimizeOperations applies fusion optimizations

#### func (*FusionOptimizer) RegisterPattern

```go
func (fo *FusionOptimizer) RegisterPattern(pattern FusionPattern)
```
RegisterPattern registers a fusion pattern

#### type FusionPattern

```go
type FusionPattern struct {
	Name        string
	Operations  []string
	CanFuse     func([]Operation) bool
	CreateFused func(*MPSGraphManager, []Operation) *CompiledOperation
}
```

FusionPattern represents a pattern of operations that can be fused

#### type FusionRegistry

```go
type FusionRegistry struct {
}
```

FusionRegistry tracks available fused operations

#### func  NewFusionRegistry

```go
func NewFusionRegistry() *FusionRegistry
```
NewFusionRegistry creates a new fusion registry

#### type FusionRule

```go
type FusionRule struct {
	Pattern     []OpType
	FusedOp     OpType
	Speedup     float32
	MemorySaved int64
}
```

FusionRule defines how operations can be fused together

#### type FusionScheduler

```go
type FusionScheduler struct {
}
```

FusionScheduler manages when to apply fusion optimizations

#### func  NewFusionScheduler

```go
func NewFusionScheduler() *FusionScheduler
```
NewFusionScheduler creates a new fusion scheduler

#### func (*FusionScheduler) DisableFusion

```go
func (fs *FusionScheduler) DisableFusion()
```
DisableFusion disables fusion optimizations

#### func (*FusionScheduler) EnableFusion

```go
func (fs *FusionScheduler) EnableFusion()
```
EnableFusion enables fusion optimizations

#### func (*FusionScheduler) GetFusionStats

```go
func (fs *FusionScheduler) GetFusionStats() map[string]*FusionStats
```
GetFusionStats returns statistics for all fusion operations

#### func (*FusionScheduler) RecordFusionResult

```go
func (fs *FusionScheduler) RecordFusionResult(operationName string, success bool, speedup float32, memorySaved int64, latency time.Duration)
```
RecordFusionResult records the result of a fusion operation

#### func (*FusionScheduler) SetMemoryThreshold

```go
func (fs *FusionScheduler) SetMemoryThreshold(threshold int64)
```
SetMemoryThreshold sets the memory threshold for fusion decisions

#### func (*FusionScheduler) SetPerformanceThreshold

```go
func (fs *FusionScheduler) SetPerformanceThreshold(threshold float32)
```
SetPerformanceThreshold sets the minimum performance gain required for fusion

#### func (*FusionScheduler) ShouldFuse

```go
func (fs *FusionScheduler) ShouldFuse(operationName string, memoryUsage int64, expectedSpeedup float32) bool
```
ShouldFuse determines if a fusion should be applied based on various criteria

#### type FusionStats

```go
type FusionStats struct {
	Count          int64
	TotalSpeedup   float32
	MemorySaved    int64
	SuccessRate    float32
	AverageLatency time.Duration
}
```

FusionStats tracks statistics for fusion operations

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

#### type GPUCPUTransferOptimizer

```go
type GPUCPUTransferOptimizer struct {
}
```

GPUCPUTransferOptimizer manages efficient data transfers between CPU and GPU

#### func  GetGlobalTransferOptimizer

```go
func GetGlobalTransferOptimizer() *GPUCPUTransferOptimizer
```
GetGlobalTransferOptimizer returns the global transfer optimizer instance

#### func  NewGPUCPUTransferOptimizer

```go
func NewGPUCPUTransferOptimizer() *GPUCPUTransferOptimizer
```
NewGPUCPUTransferOptimizer creates a new transfer optimizer

#### func (*GPUCPUTransferOptimizer) ClearCache

```go
func (opt *GPUCPUTransferOptimizer) ClearCache()
```
ClearCache clears the tensor transfer cache

#### func (*GPUCPUTransferOptimizer) EnableCaching

```go
func (opt *GPUCPUTransferOptimizer) EnableCaching(enable bool)
```
EnableCaching enables or disables transfer caching

#### func (*GPUCPUTransferOptimizer) GetCacheInfo

```go
func (opt *GPUCPUTransferOptimizer) GetCacheInfo() map[string]interface{}
```
GetCacheInfo returns information about cached tensors

#### func (*GPUCPUTransferOptimizer) GetTransferStats

```go
func (opt *GPUCPUTransferOptimizer) GetTransferStats() TransferStatistics
```
GetTransferStats returns current transfer statistics

#### func (*GPUCPUTransferOptimizer) InvalidateCPU

```go
func (opt *GPUCPUTransferOptimizer) InvalidateCPU(t *tensor.Tensor)
```
InvalidateCPU marks CPU data as invalid (e.g., after GPU computation)

#### func (*GPUCPUTransferOptimizer) InvalidateGPU

```go
func (opt *GPUCPUTransferOptimizer) InvalidateGPU(t *tensor.Tensor)
```
InvalidateGPU marks GPU data as invalid (e.g., after CPU modification)

#### func (*GPUCPUTransferOptimizer) MarkCPUValid

```go
func (opt *GPUCPUTransferOptimizer) MarkCPUValid(t *tensor.Tensor, operation string)
```
MarkCPUValid marks a tensor as having valid data on CPU

#### func (*GPUCPUTransferOptimizer) MarkGPUValid

```go
func (opt *GPUCPUTransferOptimizer) MarkGPUValid(t *tensor.Tensor, operation string)
```
MarkGPUValid marks a tensor as having valid data on GPU

#### func (*GPUCPUTransferOptimizer) OptimizeBatchTransfer

```go
func (opt *GPUCPUTransferOptimizer) OptimizeBatchTransfer(batch *BatchedTransfer) error
```
OptimizeBatchTransfer optimizes a batch of tensor transfers

#### func (*GPUCPUTransferOptimizer) OptimizeTransfer

```go
func (opt *GPUCPUTransferOptimizer) OptimizeTransfer(t *tensor.Tensor, toGPU bool, operation string) error
```
OptimizeTransfer optimizes a tensor transfer between CPU and GPU

#### func (*GPUCPUTransferOptimizer) SetBatchSize

```go
func (opt *GPUCPUTransferOptimizer) SetBatchSize(size int)
```
SetBatchSize sets the batch size for batched transfers

#### func (*GPUCPUTransferOptimizer) ShouldTransferToGPU

```go
func (opt *GPUCPUTransferOptimizer) ShouldTransferToGPU(t *tensor.Tensor) bool
```
ShouldTransferToGPU determines if a tensor should be transferred to GPU

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
func (p *GPUMemoryPool) GetStats() PoolMemoryStats
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

#### type GaussianProcess

```go
type GaussianProcess struct {
	X [][]float64
	Y []float64
}
```

GaussianProcess implements a simple Gaussian process

#### type GeneratedFunction

```go
type GeneratedFunction struct {
}
```

GeneratedFunction represents generated and compiled code

#### func  GenerateOptimizedCode

```go
func GenerateOptimizedCode(graph *ComputationGraph, functionName string) (*GeneratedFunction, error)
```
GenerateOptimizedCode generates optimized code for a computation graph

#### type GlobalGradientStats

```go
type GlobalGradientStats struct {
	GlobalNorm      float32
	TotalParameters int64
	ActiveLayers    int
	NormTrend       float32
	HealthScore     float32
}
```

GlobalGradientStats contains global gradient statistics

#### type GradientAccumulator

```go
type GradientAccumulator struct {
}
```

GradientAccumulator handles sophisticated gradient accumulation

#### type GradientAnalysisReport

```go
type GradientAnalysisReport struct {
	Timestamp    time.Time
	LayerReports map[string]*LayerGradientReport
	GlobalStats  *GlobalGradientStats
	Anomalies    []GradientAnomaly
}
```

GradientAnalysisReport contains comprehensive gradient analysis results

#### func  AnalyzeGradients

```go
func AnalyzeGradients(gradients []*GradientTensor, layerNames []string) (*GradientAnalysisReport, error)
```
AnalyzeGradients performs comprehensive gradient analysis

#### type GradientAnalyzer

```go
type GradientAnalyzer struct {
}
```

GradientAnalyzer provides detailed analysis of gradients

#### func  NewGradientAnalyzer

```go
func NewGradientAnalyzer(historySize int) *GradientAnalyzer
```
NewGradientAnalyzer creates a new gradient analyzer

#### type GradientAnomaly

```go
type GradientAnomaly struct {
	Type        GradientAnomalyType
	Timestamp   time.Time
	LayerName   string
	Severity    float32
	Description string
	Metadata    map[string]interface{}
}
```

GradientAnomaly represents a detected gradient anomaly

#### type GradientAnomalyTracker

```go
type GradientAnomalyTracker struct {
}
```

GradientAnomalyTracker detects gradient anomalies

#### type GradientAnomalyType

```go
type GradientAnomalyType int
```

GradientAnomalyType represents different types of gradient anomalies

```go
const (
	VanishingGradient GradientAnomalyType = iota
	ExplodingGradient
	GradientInstability
	GradientOscillation
	GradientPlateau
	GradientSpike
)
```

#### type GradientCheckpointing

```go
type GradientCheckpointing struct {
}
```

GradientCheckpointing implements gradient checkpointing for memory efficiency

#### func  NewGradientCheckpointing

```go
func NewGradientCheckpointing(mp *ModelParallelism, checkpointEvery int) *GradientCheckpointing
```
NewGradientCheckpointing creates a gradient checkpointing manager

#### func (*GradientCheckpointing) CheckpointBackward

```go
func (gc *GradientCheckpointing) CheckpointBackward(layers []Layer, gradOutput *tensor.Tensor)
```
CheckpointBackward runs backward pass with recomputation

#### func (*GradientCheckpointing) CheckpointForward

```go
func (gc *GradientCheckpointing) CheckpointForward(layers []Layer, input *tensor.Tensor) *tensor.Tensor
```
CheckpointForward runs forward pass with checkpointing

#### type GradientCompressionMethod

```go
type GradientCompressionMethod int
```

GradientCompressionMethod defines gradient compression methods

```go
const (
	NoCompression GradientCompressionMethod = iota
	TopKSparsification
	RandomSparsification
	Quantization
	ErrorFeedback
)
```

#### type GradientCompressor

```go
type GradientCompressor struct {
}
```

GradientCompressor handles gradient compression

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

#### type GradientLayerStats

```go
type GradientLayerStats struct {
	LayerName      string
	ParameterCount int64
	NormHistory    []float32
	MagnitudeStats *StatisticalSummary
	DirectionStats *DirectionalStats
	UpdateHistory  []GradientUpdate
	AnomalyCount   int64
	LastUpdated    time.Time
}
```

GradientLayerStats tracks statistics for gradients of a specific layer

#### type GradientManipulator

```go
type GradientManipulator struct {
}
```

GradientManipulator provides advanced gradient manipulation operations

#### func  NewGradientManipulator

```go
func NewGradientManipulator() *GradientManipulator
```
NewGradientManipulator creates a new gradient manipulator

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

#### type GradientNoiseInjector

```go
type GradientNoiseInjector struct {
}
```

GradientNoiseInjector adds controlled noise to gradients for regularization

#### type GradientSmoother

```go
type GradientSmoother struct {
}
```

GradientSmoother applies smoothing to gradients

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

#### func  CheckpointedOperation

```go
func CheckpointedOperation(forwardFn func([]*GradientTensor) (*GradientTensor, error), inputs []*GradientTensor) (*GradientTensor, error)
```
CheckpointedOperation creates a checkpointed operation

#### func  CloneGradientTensor

```go
func CloneGradientTensor(gt *GradientTensor) (*GradientTensor, error)
```
CloneGradientTensor creates a deep copy of a gradient tensor

#### func  ComputeHigherOrderDerivative

```go
func ComputeHigherOrderDerivative(function func(*GradientTensor) (*GradientTensor, error), input *GradientTensor, order int) (*GradientTensor, error)
```
ComputeHigherOrderDerivative computes higher-order derivatives

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

#### func  GradFusedAttention

```go
func GradFusedAttention(query, key, value *GradientTensor, config *FusedAttentionConfig) (*GradientTensor, error)
```
GradFusedAttention performs fused multi-head attention

#### func  GradFusedConvBNReLU

```go
func GradFusedConvBNReLU(input, kernel, gamma, beta, bias *GradientTensor, config *FusedConvolutionConfig) (*GradientTensor, error)
```
GradFusedConvBNReLU performs fused convolution + batch normalization + ReLU

#### func  GradFusedGELUDropout

```go
func GradFusedGELUDropout(input *GradientTensor, dropoutRate float32, training bool, seed uint32) (*GradientTensor, error)
```
GradFusedGELUDropout performs fused GELU + dropout

#### func  GradFusedLayerNormLinear

```go
func GradFusedLayerNormLinear(input, gamma, beta, weight, bias *GradientTensor, epsilon float32) (*GradientTensor, error)
```
GradFusedLayerNormLinear performs fused layer normalization + linear
transformation

#### func  GradFusedLinearActivation

```go
func GradFusedLinearActivation(input, weight, bias *GradientTensor, config *FusedLinearConfig) (*GradientTensor, error)
```
GradFusedLinearActivation performs fused linear transformation + activation

#### func  GradFusedResidualBlock

```go
func GradFusedResidualBlock(input, conv1Weight, bn1Gamma, bn1Beta, conv2Weight, bn2Gamma, bn2Beta *GradientTensor) (*GradientTensor, error)
```
GradFusedResidualBlock performs fused residual block
(Conv+BN+ReLU+Conv+BN+Add+ReLU)

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

#### func  TensorFusion

```go
func TensorFusion(tensors []*GradientTensor) ([]*GradientTensor, error)
```
TensorFusion fuses multiple small tensors into larger ones for efficiency

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

#### type GradientUpdate

```go
type GradientUpdate struct {
	Timestamp time.Time
	Norm      float32
	Direction []float32 // Unit vector
	Magnitude float32
	StepSize  float32
}
```

GradientUpdate represents a single gradient update

#### type GraphAnalysis

```go
type GraphAnalysis struct {
	NodeCount          int
	EdgeCount          int
	MaxDepth           int
	CriticalPath       []*GradientTensor
	CyclicDependencies bool
	ParallelizableOps  [][]OpType
	MemoryBottlenecks  []*GradientTensor
}
```

GraphAnalysis provides analysis of the computation graph structure

#### type GraphOptimizer

```go
type GraphOptimizer struct {
}
```

GraphOptimizer provides optimization tools for computation graphs

#### func  NewGraphOptimizer

```go
func NewGraphOptimizer() *GraphOptimizer
```
NewGraphOptimizer creates a new graph optimizer

#### func (*GraphOptimizer) OptimizeGraph

```go
func (go_ *GraphOptimizer) OptimizeGraph() error
```
OptimizeGraph applies various optimizations to the computation graph

#### type GraphParallelizer

```go
type GraphParallelizer struct {
}
```

GraphParallelizer identifies parallelizable operations

#### type HigherOrderAutodiff

```go
type HigherOrderAutodiff struct {
}
```

HigherOrderAutodiff provides higher-order automatic differentiation

#### func  NewHigherOrderAutodiff

```go
func NewHigherOrderAutodiff(maxOrder int) *HigherOrderAutodiff
```
NewHigherOrderAutodiff creates a new higher-order autodiff system

#### type IROperation

```go
type IROperation struct {
}
```

IROperation represents an operation in IR

#### type IRVariable

```go
type IRVariable struct {
}
```

IRVariable represents a variable in IR

#### type IntermediateRepresentation

```go
type IntermediateRepresentation struct {
}
```

IntermediateRepresentation represents code in IR form

#### type IntermediateTensorManager

```go
type IntermediateTensorManager struct {
}
```

IntermediateTensorManager manages intermediate tensors used in complex
operations

#### func  NewIntermediateTensorManager

```go
func NewIntermediateTensorManager(bufferManager *BufferReuseManager) *IntermediateTensorManager
```
NewIntermediateTensorManager creates a new intermediate tensor manager

#### func (*IntermediateTensorManager) CreateIntermediateTensor

```go
func (itm *IntermediateTensorManager) CreateIntermediateTensor(shape []int, operation string) (*tensor.Tensor, error)
```
CreateIntermediateTensor creates a temporary tensor for intermediate
computations

#### func (*IntermediateTensorManager) GetIntermediateTensorCount

```go
func (itm *IntermediateTensorManager) GetIntermediateTensorCount() int
```
GetIntermediateTensorCount returns the number of active intermediate tensors

#### func (*IntermediateTensorManager) ReleaseAllIntermediateTensors

```go
func (itm *IntermediateTensorManager) ReleaseAllIntermediateTensors()
```
ReleaseAllIntermediateTensors releases all intermediate tensors

#### type KernelCache

```go
type KernelCache struct {
}
```

KernelCache manages compiled Metal kernels with intelligent caching

#### func  GetGlobalKernelCache

```go
func GetGlobalKernelCache() *KernelCache
```
GetGlobalKernelCache returns the global kernel cache

#### func  NewKernelCache

```go
func NewKernelCache(device unsafe.Pointer) *KernelCache
```
NewKernelCache creates a new Metal kernel cache

#### func (*KernelCache) Close

```go
func (kc *KernelCache) Close()
```
Close shuts down the kernel cache

#### func (*KernelCache) GetCacheStats

```go
func (kc *KernelCache) GetCacheStats() (hitRate float64, entries int, sizeBytes int64, hitCount int64, missCount int64)
```
GetCacheStats returns cache performance statistics

#### func (*KernelCache) GetKernel

```go
func (kc *KernelCache) GetKernel(kernelSource string, options *KernelCompilationOptions) (unsafe.Pointer, error)
```
GetKernel retrieves a compiled kernel from cache or compiles it if needed

#### func (*KernelCache) InvalidateCache

```go
func (kc *KernelCache) InvalidateCache()
```
InvalidateCache clears all cached kernels

#### func (*KernelCache) SetCacheParams

```go
func (kc *KernelCache) SetCacheParams(maxSize int64, maxAge time.Duration)
```
SetCacheParams configures cache parameters

#### type KernelCacheEntry

```go
type KernelCacheEntry struct {
	CompiledKernel unsafe.Pointer         // Pointer to compiled Metal kernel
	CompileTime    time.Time              // When the kernel was compiled
	AccessCount    int64                  // Number of times accessed
	LastAccess     time.Time              // Last access time
	KernelSource   string                 // Source code of the kernel
	CompileOptions map[string]interface{} // Compilation options used
	FileSize       int64                  // Size of compiled kernel
}
```

KernelCacheEntry represents a cached compiled kernel

#### type KernelCompilationOptions

```go
type KernelCompilationOptions struct {
	OptimizationLevel int                    // 0=none, 1=basic, 2=aggressive
	FastMath          bool                   // Enable fast math optimizations
	Constants         map[string]interface{} // Compile-time constants
	MacroDefinitions  map[string]string      // Preprocessor macros
	DebugInfo         bool                   // Include debug information
}
```

KernelCompilationOptions represents options for kernel compilation

#### type KernelFunction

```go
type KernelFunction int
```

KernelFunction defines kernel function types

```go
const (
	RBFKernel KernelFunction = iota
	MaternKernel
)
```

#### type KernelStats

```go
type KernelStats struct {
	Name      string
	CallCount int64
	TotalTime time.Duration
	MinTime   time.Duration
	MaxTime   time.Duration
	AvgTime   time.Duration

	// Detailed metrics
	Instructions int64
	MemoryAccess int64
	CacheHits    int64
	CacheMisses  int64

	// Efficiency metrics
	Occupancy       float64
	Throughput      float64
	PowerEfficiency float64

	// Input characteristics
	InputSizes     [][]int
	ParameterCount int64
}
```

KernelStats tracks performance statistics for GPU kernels

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

#### type Layer

```go
type Layer interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(gradOutput *tensor.Tensor) *tensor.Tensor
	GetParameters() []*tensor.Tensor
	GetGradients() []*tensor.Tensor
}
```

Layer interface for model layers

#### type LayerGradientReport

```go
type LayerGradientReport struct {
	LayerName       string
	ParameterCount  int64
	Norm            float32
	Mean            float32
	Variance        float32
	Min             float32
	Max             float32
	Percentiles     map[int]float32
	Sparsity        float32
	DirectionChange float32
	UpdateStability float32
}
```

LayerGradientReport contains analysis for a single layer

#### type LearningRateScheduler

```go
type LearningRateScheduler struct {
}
```

LearningRateScheduler is a placeholder for demo purposes

#### func (*LearningRateScheduler) Step

```go
func (s *LearningRateScheduler) Step()
```
Step updates the learning rate scheduler

#### type LinearLayer

```go
type LinearLayer struct {
	// Parameters
	Weight *tensor.Tensor
	Bias   *tensor.Tensor

	// Gradients
	WeightGrad *tensor.Tensor
	BiasGrad   *tensor.Tensor

	// Configuration
	InputSize  int
	OutputSize int
	UseBias    bool
}
```

LinearLayer represents a fully connected layer

#### func  NewLinearLayer

```go
func NewLinearLayer(inputSize, outputSize int, useBias bool) *LinearLayer
```
NewLinearLayer creates a new linear layer

#### func (*LinearLayer) Backward

```go
func (l *LinearLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor
```
Backward performs the backward pass

#### func (*LinearLayer) Forward

```go
func (l *LinearLayer) Forward(input *tensor.Tensor) *tensor.Tensor
```
Forward performs the forward pass

#### func (*LinearLayer) GetGradients

```go
func (l *LinearLayer) GetGradients() []*tensor.Tensor
```
GetGradients returns the layer gradients

#### func (*LinearLayer) GetParameters

```go
func (l *LinearLayer) GetParameters() []*tensor.Tensor
```
GetParameters returns the layer parameters

#### type LoadBalancer

```go
type LoadBalancer struct {
}
```

LoadBalancer balances load across workers

#### type LoadBalancingStrategy

```go
type LoadBalancingStrategy int
```

LoadBalancingStrategy defines load balancing strategies

```go
const (
	RoundRobin LoadBalancingStrategy = iota
	LeastLoaded
	WorkStealing
	LocalityAware
)
```

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

#### type LossScaleStressTest

```go
type LossScaleStressTest struct {
}
```

LossScaleStressTest provides comprehensive testing of loss scaling mechanisms

#### func  NewLossScaleStressTest

```go
func NewLossScaleStressTest() *LossScaleStressTest
```
NewLossScaleStressTest creates a new stress test instance

#### func (*LossScaleStressTest) GenerateExtremePrecisionTestCases

```go
func (test *LossScaleStressTest) GenerateExtremePrecisionTestCases() ([]*tensor.Tensor, []string, error)
```
GenerateExtremePrecisionTestCases creates test cases for extreme precision
scenarios

#### func (*LossScaleStressTest) RecommendOptimalScale

```go
func (test *LossScaleStressTest) RecommendOptimalScale(results []StressTestResult) *MixedPrecisionConfig
```
RecommendOptimalScale analyzes test results and recommends optimal scaling
strategy

#### func (*LossScaleStressTest) TestOverflowScenarios

```go
func (test *LossScaleStressTest) TestOverflowScenarios() ([]StressTestResult, error)
```
TestOverflowScenarios tests various overflow scenarios to verify adaptive
scaling

#### func (*LossScaleStressTest) TestPrecisionLimits

```go
func (test *LossScaleStressTest) TestPrecisionLimits() (map[string]PrecisionTestResult, error)
```
TestPrecisionLimits tests the precision limits of float16 conversion

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

#### type MMapRegion

```go
type MMapRegion struct {
}
```

MMapRegion represents a memory-mapped file region

#### type MPSGraphManager

```go
type MPSGraphManager struct {
}
```

MPSGraphManager manages Metal Performance Shaders Graph operations

#### func  NewMPSGraphManager

```go
func NewMPSGraphManager(device unsafe.Pointer) *MPSGraphManager
```
NewMPSGraphManager creates a new MPS Graph manager

#### func (*MPSGraphManager) ActivationOp

```go
func (mgm *MPSGraphManager) ActivationOp(input string, activationType string) string
```
ActivationOp creates an activation operation

#### func (*MPSGraphManager) CompileGraph

```go
func (mgm *MPSGraphManager) CompileGraph(inputs []string, outputs []string) *CompiledOperation
```
CompileGraph compiles a subgraph for execution

#### func (*MPSGraphManager) Conv2DOp

```go
func (mgm *MPSGraphManager) Conv2DOp(input, weights string, stride, padding []int) string
```
Conv2DOp creates a 2D convolution operation

#### func (*MPSGraphManager) CreateTensor

```go
func (mgm *MPSGraphManager) CreateTensor(name string, shape []int, dataType string) unsafe.Pointer
```
CreateTensor creates an MPS graph tensor

#### func (*MPSGraphManager) Execute

```go
func (mgm *MPSGraphManager) Execute(compiled *CompiledOperation, inputs map[string]*tensor.Tensor) map[string]*tensor.Tensor
```
Execute runs a compiled graph operation

#### func (*MPSGraphManager) MatMulOp

```go
func (mgm *MPSGraphManager) MatMulOp(a, b string, transposeA, transposeB bool) string
```
MatMulOp creates a matrix multiplication operation

#### func (*MPSGraphManager) OptimizeGraph

```go
func (mgm *MPSGraphManager) OptimizeGraph(operations []Operation) []Operation
```
OptimizeGraph applies fusion and optimization passes

#### type MatMulOperation

```go
type MatMulOperation struct {
	A *tensor.Tensor
	B *tensor.Tensor
}
```

MatMulOperation represents a matrix multiplication operation

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

#### type MemoryAccessPattern

```go
type MemoryAccessPattern struct {
}
```

MemoryAccessPattern tracks how memory is accessed

#### type MemoryAllocation

```go
type MemoryAllocation struct {
}
```

MemoryAllocation represents a memory allocation

#### type MemoryBandwidthMonitor

```go
type MemoryBandwidthMonitor struct {
}
```

Memory bandwidth monitoring and optimization hints

#### func  GetGlobalBandwidthMonitor

```go
func GetGlobalBandwidthMonitor() *MemoryBandwidthMonitor
```
GetGlobalBandwidthMonitor returns the global bandwidth monitor

#### func  NewMemoryBandwidthMonitor

```go
func NewMemoryBandwidthMonitor() *MemoryBandwidthMonitor
```
NewMemoryBandwidthMonitor creates a new bandwidth monitor

#### func (*MemoryBandwidthMonitor) GetBandwidthStats

```go
func (mbm *MemoryBandwidthMonitor) GetBandwidthStats() (avgBandwidth float64, peakBandwidth float64, totalTransfers int64)
```
GetBandwidthStats returns current bandwidth statistics

#### func (*MemoryBandwidthMonitor) RecordTransfer

```go
func (mbm *MemoryBandwidthMonitor) RecordTransfer(bytes int64, duration time.Duration)
```
RecordTransfer records a memory transfer for bandwidth analysis

#### type MemoryBlock

```go
type MemoryBlock struct {
}
```

MemoryBlock represents a block of GPU memory

#### type MemoryCoalescingOptimizer

```go
type MemoryCoalescingOptimizer struct {
}
```

MemoryCoalescingOptimizer manages memory access pattern optimization

#### func  GetGlobalMemoryOptimizer

```go
func GetGlobalMemoryOptimizer() *MemoryCoalescingOptimizer
```
GetGlobalMemoryOptimizer returns the global memory coalescing optimizer

#### func  NewMemoryCoalescingOptimizer

```go
func NewMemoryCoalescingOptimizer() *MemoryCoalescingOptimizer
```
NewMemoryCoalescingOptimizer creates a new memory coalescing optimizer

#### func (*MemoryCoalescingOptimizer) AnalyzeAccessPattern

```go
func (mco *MemoryCoalescingOptimizer) AnalyzeAccessPattern(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*AccessPattern, error)
```
AnalyzeAccessPattern analyzes the memory access pattern for an operation

#### func (*MemoryCoalescingOptimizer) ClearCache

```go
func (mco *MemoryCoalescingOptimizer) ClearCache()
```
ClearCache clears the optimization cache

#### func (*MemoryCoalescingOptimizer) Enable

```go
func (mco *MemoryCoalescingOptimizer) Enable(enable bool)
```
Enable enables or disables the memory coalescing optimizer

#### func (*MemoryCoalescingOptimizer) GetStatistics

```go
func (mco *MemoryCoalescingOptimizer) GetStatistics() CoalescingStatistics
```
GetStatistics returns current coalescing statistics

#### func (*MemoryCoalescingOptimizer) IsEnabled

```go
func (mco *MemoryCoalescingOptimizer) IsEnabled() bool
```
IsEnabled returns whether the optimizer is enabled

#### func (*MemoryCoalescingOptimizer) OptimizeCoalescing

```go
func (mco *MemoryCoalescingOptimizer) OptimizeCoalescing(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*CoalescingStrategy, error)
```
OptimizeCoalescing optimizes memory access patterns for better coalescing

#### type MemoryConstraints

```go
type MemoryConstraints struct {
	MaxMemoryUsage     int64
	MaxTensorSize      int64
	FragmentationLimit float32
}
```

MemoryConstraints defines memory constraints

#### type MemoryCoordinator

```go
type MemoryCoordinator struct {
}
```

MemoryCoordinator coordinates memory access between CPU and GPU

#### func (*MemoryCoordinator) OptimizeAccess

```go
func (mc *MemoryCoordinator) OptimizeAccess(bufferName string, accessType int)
```
OptimizeAccess optimizes memory access based on patterns

#### type MemoryCost

```go
type MemoryCost struct {
	AllocationCost int64
	AccessCost     int64
	BandwidthUsage int64
}
```

MemoryCost represents memory usage costs

#### type MemoryEfficientAutodiffEngine

```go
type MemoryEfficientAutodiffEngine struct {
}
```

MemoryEfficientAutodiffEngine manages memory-efficient autodiff

#### func  NewMemoryEfficientAutodiffEngine

```go
func NewMemoryEfficientAutodiffEngine(config *MemoryEfficientConfig) *MemoryEfficientAutodiffEngine
```
NewMemoryEfficientAutodiffEngine creates a new memory-efficient autodiff engine

#### type MemoryEfficientConfig

```go
type MemoryEfficientConfig struct {
	MaxMemoryUsage        int64
	CheckpointingStrategy CheckpointingStrategy
	CheckpointingRatio    float32 // Fraction of layers to checkpoint
	GradientCompression   GradientCompressionMethod
	CompressionRatio      float32 // Compression ratio (0.0-1.0)
	EnableGradientScaling bool
	EnableMixedPrecision  bool
	EnableInPlaceOps      bool
	EnableTensorFusion    bool
	MemoryBudget          int64 // Target memory budget
	SwapThreshold         int64 // Threshold for CPU-GPU swapping
}
```

MemoryEfficientConfig configures memory-efficient autodiff

#### type MemoryEfficientGradContext

```go
type MemoryEfficientGradContext struct {
}
```

MemoryEfficientGradContext provides a context for memory-efficient gradient
computation

#### type MemoryEfficientTraining

```go
type MemoryEfficientTraining struct {
}
```

MemoryEfficientTraining coordinates memory-efficient training techniques

#### func  NewMemoryEfficientTraining

```go
func NewMemoryEfficientTraining(mp *ModelParallelism) *MemoryEfficientTraining
```
NewMemoryEfficientTraining creates a memory-efficient training coordinator

#### func (*MemoryEfficientTraining) TrainStep

```go
func (met *MemoryEfficientTraining) TrainStep(model []Layer, input, target *tensor.Tensor, loss func(*tensor.Tensor, *tensor.Tensor) float32) float32
```
TrainStep performs a memory-efficient training step

#### type MemoryLayout

```go
type MemoryLayout int
```

MemoryLayout represents tensor memory layout

```go
const (
	RowMajor MemoryLayout = iota
	ColumnMajor
	Packed
	Sparse
)
```

#### type MemoryOptimizationBenefit

```go
type MemoryOptimizationBenefit struct {
	MemorySaved          int64
	FragmentationReduced float32
	AllocationSpeedup    float32
}
```

MemoryOptimizationBenefit quantifies memory optimization benefits

#### type MemoryOptimizationSuite

```go
type MemoryOptimizationSuite struct {
}
```

MemoryOptimizationSuite combines all Phase 8C optimization components

#### func  GetGlobalMemoryOptimizationSuite

```go
func GetGlobalMemoryOptimizationSuite() *MemoryOptimizationSuite
```
GetGlobalMemoryOptimizationSuite returns the global memory optimization suite

#### func  NewMemoryOptimizationSuite

```go
func NewMemoryOptimizationSuite(device unsafe.Pointer, config *OptimizationConfig) *MemoryOptimizationSuite
```
NewMemoryOptimizationSuite creates a new memory optimization suite

#### func (*MemoryOptimizationSuite) Close

```go
func (mos *MemoryOptimizationSuite) Close()
```
Close releases all resources associated with the optimization suite

#### func (*MemoryOptimizationSuite) GetSuiteStats

```go
func (mos *MemoryOptimizationSuite) GetSuiteStats() map[string]interface{}
```
GetSuiteStats returns comprehensive statistics for the entire optimization suite

#### func (*MemoryOptimizationSuite) OptimizeTensorOperation

```go
func (mos *MemoryOptimizationSuite) OptimizeTensorOperation(
	operationType string,
	tensors []*tensor.Tensor,
	params map[string]interface{},
) (*OptimizedOperation, error)
```
OptimizeTensorOperation performs comprehensive optimization for a tensor
operation

#### type MemoryOptimizationTechnique

```go
type MemoryOptimizationTechnique interface {
	GetName() string
	Apply(allocations []*MemoryAllocation) error
	GetBenefit() MemoryOptimizationBenefit
}
```

MemoryOptimizationTechnique represents a memory optimization technique

#### type MemoryOptimizer

```go
type MemoryOptimizer struct {
}
```

MemoryOptimizer handles memory optimization for gradients

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

#### type MemoryStatistics

```go
type MemoryStatistics struct {
	ZeroCopyHits  int64
	Allocations   int64
	Deallocations int64
}
```

MemoryStatistics represents unified memory statistics

#### type MemoryStats

```go
type MemoryStats struct {
	TotalMemory int64
	UsedMemory  int64
	FreeMemory  int64

	// Allocation tracking
	AllocCount    int64
	DeallocCount  int64
	FragmentCount int64

	// Bandwidth metrics
	ReadBandwidth  float64
	WriteBandwidth float64
	PeakBandwidth  float64

	// Usage patterns
	AllocationSizes []int
	LifetimeHist    map[time.Duration]int
}
```

MemoryStats tracks GPU memory usage

#### type Metric

```go
type Metric struct {
	Timestamp  time.Time
	Type       string
	Name       string
	Value      float64
	Attributes map[string]interface{}
}
```

Metric represents a performance measurement

#### type MicroBatchAccumulator

```go
type MicroBatchAccumulator struct {
}
```

MicroBatchAccumulator handles gradient accumulation for micro-batching

#### func  NewMicroBatchAccumulator

```go
func NewMicroBatchAccumulator(targetSteps int) *MicroBatchAccumulator
```
NewMicroBatchAccumulator creates a gradient accumulator

#### func (*MicroBatchAccumulator) AccumulateGradients

```go
func (mba *MicroBatchAccumulator) AccumulateGradients(layerID string, grads *tensor.Tensor) bool
```
AccumulateGradients adds gradients from a micro-batch

#### func (*MicroBatchAccumulator) GetAccumulatedGradients

```go
func (mba *MicroBatchAccumulator) GetAccumulatedGradients() map[string]*tensor.Tensor
```
GetAccumulatedGradients returns the accumulated gradients

#### type MixedPrecisionConfig

```go
type MixedPrecisionConfig struct {
	Enabled              bool    // Enable mixed precision training
	LossScale            float32 // Initial loss scale factor
	LossScaleGrowthRate  float32 // Factor to increase loss scale when no overflow
	LossScaleBackoffRate float32 // Factor to decrease loss scale on overflow
	GrowthInterval       int     // Number of steps between loss scale growth attempts
	MaxLossScale         float32 // Maximum allowed loss scale
	MinLossScale         float32 // Minimum allowed loss scale
	SkipOverflowSteps    bool    // Skip optimizer step on gradient overflow
}
```

MixedPrecisionConfig configures mixed precision training behavior

#### func  DefaultMixedPrecisionConfig

```go
func DefaultMixedPrecisionConfig() *MixedPrecisionConfig
```
DefaultMixedPrecisionConfig returns default mixed precision configuration

#### type MixedPrecisionContext

```go
type MixedPrecisionContext struct {
}
```

MixedPrecisionContext provides mixed precision training context

#### type MixedPrecisionTrainer

```go
type MixedPrecisionTrainer struct {
}
```

MixedPrecisionTrainer manages mixed precision training state

#### func  NewMixedPrecisionTrainer

```go
func NewMixedPrecisionTrainer(config *MixedPrecisionConfig) (*MixedPrecisionTrainer, error)
```
NewMixedPrecisionTrainer creates a new mixed precision trainer

#### func (*MixedPrecisionTrainer) Cleanup

```go
func (mp *MixedPrecisionTrainer) Cleanup()
```
Cleanup releases GPU resources

#### func (*MixedPrecisionTrainer) ConvertTensorToFloat16

```go
func (mp *MixedPrecisionTrainer) ConvertTensorToFloat16(input *tensor.Tensor) (*tensor.Tensor, error)
```
ConvertTensorToFloat16 converts a float32 tensor to float16 representation

#### func (*MixedPrecisionTrainer) ForwardFloat16

```go
func (mp *MixedPrecisionTrainer) ForwardFloat16(input *tensor.Tensor, weights *tensor.Tensor, bias *tensor.Tensor) (*tensor.Tensor, error)
```
ForwardFloat16 performs forward pass with automatic mixed precision

#### func (*MixedPrecisionTrainer) GetCurrentLossScale

```go
func (mp *MixedPrecisionTrainer) GetCurrentLossScale() float32
```
GetCurrentLossScale returns the current loss scale value

#### func (*MixedPrecisionTrainer) GetOverflowStatus

```go
func (mp *MixedPrecisionTrainer) GetOverflowStatus() bool
```
GetOverflowStatus returns whether overflow was detected in the last gradient
computation

#### func (*MixedPrecisionTrainer) ScaleGradients

```go
func (mp *MixedPrecisionTrainer) ScaleGradients(gradients *tensor.Tensor) (*tensor.Tensor, error)
```
ScaleGradients applies loss scaling to gradients

#### func (*MixedPrecisionTrainer) ShouldSkipStep

```go
func (mp *MixedPrecisionTrainer) ShouldSkipStep() bool
```
ShouldSkipStep returns true if the optimizer step should be skipped due to
overflow

#### func (*MixedPrecisionTrainer) UnscaleGradients

```go
func (mp *MixedPrecisionTrainer) UnscaleGradients(scaledGradients *tensor.Tensor) (*tensor.Tensor, error)
```
UnscaleGradients removes loss scaling from gradients

#### func (*MixedPrecisionTrainer) UpdateLossScale

```go
func (mp *MixedPrecisionTrainer) UpdateLossScale()
```
UpdateLossScale updates the loss scale based on overflow detection

#### type MixedPrecisionTrainingConfig

```go
type MixedPrecisionTrainingConfig struct {
	TrainingConfig
	MixedPrecision *MixedPrecisionConfig
}
```

MixedPrecisionTrainingConfig extends TrainingConfig with mixed precision
settings

#### type MixedPrecisionTrainingLoop

```go
type MixedPrecisionTrainingLoop struct {
}
```

MixedPrecisionTrainingLoop implements automatic mixed precision training

#### func  NewMixedPrecisionTrainingLoop

```go
func NewMixedPrecisionTrainingLoop(config *MixedPrecisionTrainingConfig, opt optimizer.Optimizer) (*MixedPrecisionTrainingLoop, error)
```
NewMixedPrecisionTrainingLoop creates a new mixed precision training loop

#### func (*MixedPrecisionTrainingLoop) Cleanup

```go
func (loop *MixedPrecisionTrainingLoop) Cleanup()
```
Cleanup releases resources

#### func (*MixedPrecisionTrainingLoop) GetTrainingStats

```go
func (loop *MixedPrecisionTrainingLoop) GetTrainingStats() map[string]interface{}
```
GetTrainingStats returns comprehensive training statistics

#### func (*MixedPrecisionTrainingLoop) TrainEpoch

```go
func (loop *MixedPrecisionTrainingLoop) TrainEpoch(
	inputs []*tensor.Tensor,
	targets []*tensor.Tensor,
	weights []*tensor.Tensor,
	forwardFunc func(*tensor.Tensor, []*tensor.Tensor) (*tensor.Tensor, error),
	lossFunc func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error),
	backwardFunc func(*tensor.Tensor, []*tensor.Tensor) ([]*tensor.Tensor, error),
) error
```
TrainEpoch trains for one epoch with mixed precision

#### type ModelParallelism

```go
type ModelParallelism struct {
}
```

ModelParallelism implements model parallelism on a single device

#### func  NewModelParallelism

```go
func NewModelParallelism(streamMgr *StreamManager, memMgr *UnifiedMemoryManager) *ModelParallelism
```
NewModelParallelism creates a model parallelism manager

#### func (*ModelParallelism) PipelineForward

```go
func (mp *ModelParallelism) PipelineForward(stages []*PipelineStage, input *tensor.Tensor) *tensor.Tensor
```
PipelineForward executes forward pass with pipeline parallelism

#### func (*ModelParallelism) SplitModel

```go
func (mp *ModelParallelism) SplitModel(layers []Layer, strategy string) ([]*PipelineStage, error)
```
SplitModel splits a model into pipeline stages

#### type NodeInfo

```go
type NodeInfo struct {
	NodeID    int
	Address   string
	GPUCount  int
	Available bool
	LastSeen  time.Time
}
```

NodeInfo represents information about a distributed node

#### type NoiseEvent

```go
type NoiseEvent struct {
	Timestamp      time.Time
	NoiseLevel     float32
	NoiseType      NoiseType
	LayersAffected int
}
```

NoiseEvent records a noise injection event

#### type NoiseSchedule

```go
type NoiseSchedule struct {
	ScheduleType NoiseScheduleType
	InitialLevel float32
	FinalLevel   float32
	DecaySteps   int64
	DecayRate    float32
}
```

NoiseSchedule controls how noise level changes over time

#### type NoiseScheduleType

```go
type NoiseScheduleType int
```

NoiseScheduleType represents different noise scheduling strategies

```go
const (
	ConstantNoise NoiseScheduleType = iota
	LinearDecay
	ExponentialDecay
	CosineDecay
)
```

#### type NoiseType

```go
type NoiseType int
```

NoiseType represents different types of gradient noise

```go
const (
	GaussianNoise NoiseType = iota
	UniformNoise
	SaltPepperNoise
	DropoutNoise
)
```

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

#### type Operation

```go
type Operation struct {
	Type       string
	Inputs     []string
	Output     string
	Attributes map[string]interface{}
}
```

Operation represents a graph operation

#### type OperationCost

```go
type OperationCost struct {
	ComputeTime  time.Duration
	MemoryAccess int64
	FlopsCount   int64
	PowerUsage   float32
}
```

OperationCost represents the cost of an operation

#### type OperationScope

```go
type OperationScope struct {
}
```

OperationScope represents a scope for managing intermediate tensors in an
operation

#### func  NewOperationScope

```go
func NewOperationScope(operation string) *OperationScope
```
NewOperationScope creates a new operation scope for managing intermediate
tensors

#### func (*OperationScope) Close

```go
func (os *OperationScope) Close()
```
Close releases all intermediate tensors in this scope

#### func (*OperationScope) CreateTensor

```go
func (os *OperationScope) CreateTensor(shape []int) (*tensor.Tensor, error)
```
CreateTensor creates an intermediate tensor within this scope

#### func (*OperationScope) GetDuration

```go
func (os *OperationScope) GetDuration() time.Duration
```
GetDuration returns the duration this scope has been active

#### func (*OperationScope) GetTensorCount

```go
func (os *OperationScope) GetTensorCount() int
```
GetTensorCount returns the number of active intermediate tensors in this scope

#### type OperationStats

```go
type OperationStats struct {
	Count           int64
	TotalTime       time.Duration
	AverageTime     time.Duration
	PeakMemoryUsage int64
	TotalMemoryUsed int64
	GradientNorm    float32
	BackwardTime    time.Duration
	ForwardTime     time.Duration
}
```

OperationStats tracks statistics for each operation type

#### type OptimizationBenefit

```go
type OptimizationBenefit struct {
	MemorySaving   int64
	ComputeSaving  float32
	Speedup        float32
	AccuracyImpact float32
}
```

OptimizationBenefit quantifies the benefit of an optimization

#### type OptimizationConfig

```go
type OptimizationConfig struct {
	MaxMemoryPoolSize      int64 // Maximum GPU memory pool size
	MaxBufferCacheSize     int64 // Maximum buffer cache size
	MaxKernelCacheSize     int64 // Maximum kernel cache size
	MaxSharedMemory        int   // Maximum shared memory per threadgroup
	EnableTransferOpt      bool  // Enable CPU-GPU transfer optimization
	EnableMemoryCoalescing bool  // Enable memory coalescing optimization
	EnableLayoutOpt        bool  // Enable tensor layout optimization
	EnableBufferReuse      bool  // Enable buffer reuse optimization
	EnableKernelCache      bool  // Enable kernel compilation caching
	EnableSharedMemOpt     bool  // Enable shared memory optimization
	BandwidthMonitoring    bool  // Enable bandwidth monitoring
}
```

OptimizationConfig configures the memory optimization suite

#### func  DefaultOptimizationConfig

```go
func DefaultOptimizationConfig() *OptimizationConfig
```
DefaultOptimizationConfig returns a default configuration for memory
optimization

#### type OptimizationPass

```go
type OptimizationPass struct {
	Name        string
	Description string
	Apply       func(*IntermediateRepresentation) error
	Priority    int
}
```

OptimizationPass represents a code optimization pass

#### type OptimizationTechnique

```go
type OptimizationTechnique interface {
	GetName() string
	GetDescription() string
	Apply(graph *ComputationGraph) error
	GetBenefit() OptimizationBenefit
	IsApplicable(graph *ComputationGraph) bool
}
```

OptimizationTechnique represents an optimization technique

#### type OptimizedMixedPrecisionOps

```go
type OptimizedMixedPrecisionOps struct {
}
```

OptimizedMixedPrecisionOps provides optimized mixed precision operations

#### func  NewOptimizedMixedPrecisionOps

```go
func NewOptimizedMixedPrecisionOps(config *MixedPrecisionConfig) (*OptimizedMixedPrecisionOps, error)
```
NewOptimizedMixedPrecisionOps creates optimized mixed precision operations

#### func (*OptimizedMixedPrecisionOps) BatchedMatMulOptimized

```go
func (opt *OptimizedMixedPrecisionOps) BatchedMatMulOptimized(matrices [][]*tensor.Tensor) ([]*tensor.Tensor, error)
```
BatchedMatMulOptimized performs batched matrix operations with optimal precision
selection

#### func (*OptimizedMixedPrecisionOps) BenchmarkMatrixOperation

```go
func (opt *OptimizedMixedPrecisionOps) BenchmarkMatrixOperation(size int, iterations int) (*PerformanceBenchmark, error)
```
BenchmarkMatrixOperation provides detailed performance analysis

#### func (*OptimizedMixedPrecisionOps) Cleanup

```go
func (opt *OptimizedMixedPrecisionOps) Cleanup()
```
Cleanup releases resources

#### func (*OptimizedMixedPrecisionOps) MatMulOptimized

```go
func (opt *OptimizedMixedPrecisionOps) MatMulOptimized(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
MatMulOptimized performs optimized matrix multiplication with minimal overhead

#### type OptimizedOperation

```go
type OptimizedOperation struct {
	OperationType        string
	Tensors              []*OptimizedTensor
	SharedMemoryLayout   *SharedMemoryLayout
	BufferScope          *OperationScope
	StartTime            time.Time
	OptimizationTime     time.Duration
	ExecutionTime        time.Duration
	TotalMemoryUsed      int64
	BandwidthUtilization float64
}
```

OptimizedOperation represents a fully optimized tensor operation

#### func  OptimizeOperation

```go
func OptimizeOperation(operationType string, tensors []*tensor.Tensor, params map[string]interface{}) (*OptimizedOperation, error)
```
OptimizeOperation provides a high-level interface for optimizing tensor
operations

#### func (*OptimizedOperation) Cleanup

```go
func (oo *OptimizedOperation) Cleanup()
```
Cleanup releases resources associated with the optimized operation

#### func (*OptimizedOperation) Execute

```go
func (oo *OptimizedOperation) Execute(kernelSource string, additionalParams map[string]interface{}) error
```
Execute executes the optimized operation

#### func (*OptimizedOperation) GetOptimizationStats

```go
func (oo *OptimizedOperation) GetOptimizationStats() map[string]interface{}
```
GetOptimizationStats returns detailed statistics about the optimization

#### type OptimizedTensor

```go
type OptimizedTensor struct {
	Original   *tensor.Tensor
	Optimized  *tensor.Tensor
	LayoutInfo *TensorLayoutInfo
}
```

OptimizedTensor represents an optimized tensor with layout information

#### type OptimizedTensorBuffer

```go
type OptimizedTensorBuffer struct {
}
```

OptimizedTensorBuffer manages efficient buffer allocation for tensors

#### func  NewOptimizedTensorBuffer

```go
func NewOptimizedTensorBuffer(device unsafe.Pointer) *OptimizedTensorBuffer
```
NewOptimizedTensorBuffer creates a new optimized tensor buffer manager

#### func (*OptimizedTensorBuffer) GetTempBuffer

```go
func (otb *OptimizedTensorBuffer) GetTempBuffer(size int) (unsafe.Pointer, error)
```
GetTempBuffer allocates a temporary buffer for intermediate computations

#### func (*OptimizedTensorBuffer) ReleaseAll

```go
func (otb *OptimizedTensorBuffer) ReleaseAll()
```
ReleaseAll releases all temporary buffers

#### func (*OptimizedTensorBuffer) ReturnTempBuffer

```go
func (otb *OptimizedTensorBuffer) ReturnTempBuffer(buffer unsafe.Pointer)
```
ReturnTempBuffer returns a temporary buffer for reuse

#### type OptimizerTrial

```go
type OptimizerTrial struct {
	Parameters map[string]interface{}
	Score      float64
}
```

OptimizerTrial represents a trial in the optimizer

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

#### type ParallelExecutor

```go
type ParallelExecutor struct {
}
```

ParallelExecutor manages parallel kernel execution

#### func  NewParallelExecutor

```go
func NewParallelExecutor(streamMgr *StreamManager) *ParallelExecutor
```
NewParallelExecutor creates a new parallel execution manager

#### func (*ParallelExecutor) Shutdown

```go
func (pe *ParallelExecutor) Shutdown()
```
Shutdown gracefully shuts down the executor

#### func (*ParallelExecutor) Submit

```go
func (pe *ParallelExecutor) Submit(task StreamTask)
```
Submit adds a task to the execution queue

#### func (*ParallelExecutor) SubmitBatch

```go
func (pe *ParallelExecutor) SubmitBatch(tasks []StreamTask)
```
SubmitBatch submits multiple tasks for potential fusion

#### type ParallelizationEngine

```go
type ParallelizationEngine struct {
}
```

ParallelizationEngine manages parallel execution

#### func  NewParallelizationEngine

```go
func NewParallelizationEngine() *ParallelizationEngine
```
NewParallelizationEngine creates a new parallelization engine

#### type PerformanceBenchmark

```go
type PerformanceBenchmark struct {
	MatrixSize         int
	Float32Time        time.Duration
	MixedPrecisionTime time.Duration
	OptimizedTime      time.Duration
	ConversionTime     time.Duration
	ActualComputeTime  time.Duration
	MemoryUsage        int64
	Accuracy           float64
	Speedup            float64
	EffectiveSpeedup   float64
}
```

PerformanceBenchmark provides comprehensive performance analysis

#### type PerformanceMetrics

```go
type PerformanceMetrics struct {
	Latency     time.Duration
	Throughput  float64
	PowerUsage  float64
	MemoryUsage int64
	Accuracy    float64
}
```

PerformanceMetrics summarizes performance results

#### type PerformanceProfile

```go
type PerformanceProfile struct {
	MatrixSize           int
	Float32Time          int64   // nanoseconds
	MixedPrecisionTime   int64   // nanoseconds
	OptimalTime          int64   // nanoseconds
	MemoryBandwidthUsage float64 // GB/s
	ComputeIntensity     float64
	RecommendedPrecision string
	SpeedupAchieved      float64
	AccuracyLoss         float64
}
```

PerformanceProfile provides detailed performance characteristics

#### type PerformanceProfiler

```go
type PerformanceProfiler struct {

	// Auto-tuning integration
	AutoTuner *AutoTuner
}
```

PerformanceProfiler profiles GPU operations and system performance

#### func  NewPerformanceProfiler

```go
func NewPerformanceProfiler(device unsafe.Pointer) *PerformanceProfiler
```
NewPerformanceProfiler creates a new performance profiler

#### func (*PerformanceProfiler) ExportProfile

```go
func (pp *PerformanceProfiler) ExportProfile(filename string) error
```
ExportProfile exports profiling data to file

#### func (*PerformanceProfiler) GetAllStats

```go
func (pp *PerformanceProfiler) GetAllStats() map[string]*KernelStats
```
GetAllStats returns all performance statistics

#### func (*PerformanceProfiler) GetKernelStats

```go
func (pp *PerformanceProfiler) GetKernelStats(name string) *KernelStats
```
GetKernelStats returns statistics for a specific kernel

#### func (*PerformanceProfiler) ProfileKernel

```go
func (pp *PerformanceProfiler) ProfileKernel(name string, exec func())
```
ProfileKernel records performance data for a kernel execution

#### func (*PerformanceProfiler) StartProfiling

```go
func (pp *PerformanceProfiler) StartProfiling()
```
StartProfiling begins performance monitoring

#### func (*PerformanceProfiler) StopProfiling

```go
func (pp *PerformanceProfiler) StopProfiling()
```
StopProfiling ends performance monitoring

#### type PipelineStage

```go
type PipelineStage struct {
	ID         int
	Layers     []Layer
	StreamID   StreamID
	InputSize  []int
	OutputSize []int
}
```

PipelineStage represents a stage in the model pipeline

#### type PipelineTask

```go
type PipelineTask struct {
	StageID      int
	MicroBatchID int
	TaskType     string // "forward" or "backward"
}
```

PipelineTask represents a task in the pipeline schedule

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

#### type PoolMemoryStats

```go
type PoolMemoryStats struct {
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

PoolMemoryStats tracks memory pool statistics

#### type PowerStats

```go
type PowerStats struct {
	GPUPower    float64
	MemoryPower float64
	TotalPower  float64

	// Temperature
	GPUTemp    float64
	MemoryTemp float64

	// Throttling
	ThermalEvents int64
	PowerEvents   int64
}
```

PowerStats tracks power consumption

#### type PrecisionBenchmark

```go
type PrecisionBenchmark struct {
	MatrixSize                 int
	Iterations                 int
	Float32Time                time.Duration
	OriginalMixedPrecisionTime time.Duration
	FixedMixedPrecisionTime    time.Duration
	PureFloat32Time            time.Duration

	OriginalSpeedup  float64
	FixedSpeedup     float64
	ImprovementRatio float64
	Recommendation   string
}
```

PrecisionBenchmark contains results from precision strategy comparison

#### type PrecisionTestResult

```go
type PrecisionTestResult struct {
	TestCase         string
	OriginalValues   []float32
	ConvertedValues  []float32
	AbsoluteErrors   []float64
	RelativeErrors   []float64
	MaxAbsoluteError float64
	MaxRelativeError float64
	AvgAbsoluteError float64
	AvgRelativeError float64
}
```

PrecisionTestResult holds the results of precision testing

#### type PrecisionType

```go
type PrecisionType int
```

PrecisionType represents the data precision for mixed precision training

```go
const (
	PrecisionFloat32 PrecisionType = iota
	PrecisionFloatMP16
)
```

#### type PrefetchQueue

```go
type PrefetchQueue struct {
}
```

PrefetchQueue manages prefetching of tensors from CPU to GPU

#### type PrefetchTask

```go
type PrefetchTask struct {
}
```

PrefetchTask represents a prefetching task

#### type PrefetchedBatch

```go
type PrefetchedBatch struct {
}
```

PrefetchedBatch represents a prefetched batch

#### type PriorityItem

```go
type PriorityItem struct {
}
```

PriorityItem represents an item in the priority queue

#### type PriorityQueue

```go
type PriorityQueue struct {
}
```

PriorityQueue implements a priority queue for tasks

#### func  NewPriorityQueue

```go
func NewPriorityQueue() *PriorityQueue
```
NewPriorityQueue creates a new priority queue

#### func (*PriorityQueue) Pop

```go
func (pq *PriorityQueue) Pop() *Task
```
Pop removes and returns the highest priority item

#### func (*PriorityQueue) Push

```go
func (pq *PriorityQueue) Push(task *Task, priority int)
```
Push adds an item to the priority queue

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

#### type ReLULayer

```go
type ReLULayer struct {
}
```

ReLULayer represents a ReLU activation layer

#### func  NewReLULayer

```go
func NewReLULayer() *ReLULayer
```
NewReLULayer creates a new ReLU layer

#### func (*ReLULayer) Backward

```go
func (r *ReLULayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor
```
Backward performs the backward pass

#### func (*ReLULayer) Forward

```go
func (r *ReLULayer) Forward(input *tensor.Tensor) *tensor.Tensor
```
Forward performs the forward pass

#### func (*ReLULayer) GetGradients

```go
func (r *ReLULayer) GetGradients() []*tensor.Tensor
```
GetGradients returns empty slice (no gradients)

#### func (*ReLULayer) GetParameters

```go
func (r *ReLULayer) GetParameters() []*tensor.Tensor
```
GetParameters returns empty slice (no parameters)

#### type RecomputationManager

```go
type RecomputationManager struct {
}
```

RecomputationManager manages selective recomputation

#### func  NewRecomputationManager

```go
func NewRecomputationManager() *RecomputationManager
```
NewRecomputationManager creates a new recomputation manager

#### type RecomputationSchedule

```go
type RecomputationSchedule struct {
}
```

RecomputationSchedule defines when to recompute

#### type RecomputationStrategy

```go
type RecomputationStrategy int
```

RecomputationStrategy defines recomputation strategies

```go
const (
	NoRecomputation RecomputationStrategy = iota
	MemoryConstrainedRecomputation
	ComputeOptimalRecomputation
	HybridRecomputation
)
```

#### type Request

```go
type Request struct {
	ID         string
	Input      *tensor.Tensor
	Priority   int
	Timestamp  time.Time
	Deadline   time.Time
	ResultChan chan *tensor.Tensor

	// Request metadata
	ModelID  string
	BatchIdx int
}
```

Request represents an inference request

#### type RequestPriorityQueue

```go
type RequestPriorityQueue []*Request
```

RequestPriorityQueue implements a priority queue for requests

#### func (RequestPriorityQueue) Len

```go
func (pq RequestPriorityQueue) Len() int
```

#### func (RequestPriorityQueue) Less

```go
func (pq RequestPriorityQueue) Less(i, j int) bool
```

#### func (*RequestPriorityQueue) Pop

```go
func (pq *RequestPriorityQueue) Pop() interface{}
```

#### func (*RequestPriorityQueue) Push

```go
func (pq *RequestPriorityQueue) Push(x interface{})
```

#### func (RequestPriorityQueue) Swap

```go
func (pq RequestPriorityQueue) Swap(i, j int)
```

#### type ReusableBuffer

```go
type ReusableBuffer struct {
	GPUPtr         unsafe.Pointer // GPU buffer pointer
	Size           int            // Buffer size in bytes
	Shape          []int          // Current tensor shape using this buffer
	LastUsed       time.Time      // Last usage time for cleanup
	UsageCount     int            // Number of times reused
	Category       string         // Buffer size category (small, medium, large, etc.)
	IsActive       bool           // Whether buffer is currently in use
	OriginalTensor *tensor.Tensor // Original tensor if this buffer was allocated for one
}
```

ReusableBuffer represents a GPU buffer that can be reused for multiple
operations

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

#### type ScalingEvent

```go
type ScalingEvent struct {
	Timestamp      time.Time
	ScaleFactor    float32
	Reason         string
	LayersAffected int
}
```

ScalingEvent records a gradient scaling event

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

#### type Sequential

```go
type Sequential struct {
}
```

Sequential represents a sequential container of layers

#### func  NewSequential

```go
func NewSequential(layers ...Layer) *Sequential
```
NewSequential creates a new sequential model

#### func (*Sequential) Backward

```go
func (s *Sequential) Backward(gradOutput *tensor.Tensor) *tensor.Tensor
```
Backward performs backward pass through all layers

#### func (*Sequential) Forward

```go
func (s *Sequential) Forward(input *tensor.Tensor) *tensor.Tensor
```
Forward performs forward pass through all layers

#### func (*Sequential) GetGradients

```go
func (s *Sequential) GetGradients() []*tensor.Tensor
```
GetGradients returns all gradients from all layers

#### func (*Sequential) GetParameters

```go
func (s *Sequential) GetParameters() []*tensor.Tensor
```
GetParameters returns all parameters from all layers

#### type SharedBuffer

```go
type SharedBuffer struct {
}
```

SharedBuffer represents a zero-copy shared memory buffer

#### func (*SharedBuffer) CPUPtr

```go
func (sb *SharedBuffer) CPUPtr() unsafe.Pointer
```
CPUPtr returns the CPU pointer for the shared buffer

#### func (*SharedBuffer) GPUBuffer

```go
func (sb *SharedBuffer) GPUBuffer() unsafe.Pointer
```
GPUBuffer returns the GPU buffer pointer

#### func (*SharedBuffer) Size

```go
func (sb *SharedBuffer) Size() int
```
Size returns the buffer size

#### type SharedMemoryAccess

```go
type SharedMemoryAccess struct {
	Pattern       string // "sequential", "strided", "tiled", "random"
	Stride        int    // Access stride
	CoalesceWidth int    // Width for coalesced access
	ConflictFree  bool   // Whether access pattern is conflict-free
	PrefetchSize  int    // Amount to prefetch
}
```

SharedMemoryAccess describes memory access patterns

#### type SharedMemoryBank

```go
type SharedMemoryBank struct {
	BankID       int    // Bank identifier
	Size         int    // Size of this bank in bytes
	AccessStride int    // Optimal access stride for this bank
	DataType     string // Data type stored in this bank
}
```

SharedMemoryBank represents a memory bank configuration

#### type SharedMemoryLayout

```go
type SharedMemoryLayout struct {
	TotalSize     int                    // Total shared memory size needed
	Banks         []SharedMemoryBank     // Memory banks configuration
	AccessPattern SharedMemoryAccess     // Access pattern optimization
	Padding       []int                  // Padding to avoid bank conflicts
	TileSize      []int                  // Optimal tile sizes for operations
	ThreadMapping map[string]interface{} // Thread-to-memory mapping strategy
}
```

SharedMemoryLayout describes an optimized shared memory layout

#### func  OptimizeSharedMemoryForOperation

```go
func OptimizeSharedMemoryForOperation(operation string, tensors []*tensor.Tensor, params map[string]interface{}) (*SharedMemoryLayout, error)
```
OptimizeSharedMemoryForOperation provides a high-level interface for shared
memory optimization

#### type SharedMemoryOptimizer

```go
type SharedMemoryOptimizer struct {
}
```

SharedMemoryOptimizer optimizes shared memory usage in GPU kernels

#### func  GetGlobalSharedMemoryOptimizer

```go
func GetGlobalSharedMemoryOptimizer() *SharedMemoryOptimizer
```
GetGlobalSharedMemoryOptimizer returns the global shared memory optimizer

#### func  NewSharedMemoryOptimizer

```go
func NewSharedMemoryOptimizer() *SharedMemoryOptimizer
```
NewSharedMemoryOptimizer creates a new shared memory optimizer

#### func (*SharedMemoryOptimizer) ApplySharedMemoryOptimization

```go
func (smo *SharedMemoryOptimizer) ApplySharedMemoryOptimization(kernelSource string, layout *SharedMemoryLayout) (string, error)
```
ApplySharedMemoryOptimization applies shared memory optimization to a kernel

#### func (*SharedMemoryOptimizer) GenerateOptimizedKernel

```go
func (smo *SharedMemoryOptimizer) GenerateOptimizedKernel(operation string, params map[string]interface{}) (string, *SharedMemoryLayout, error)
```
GenerateOptimizedKernel generates an optimized kernel for a specific operation

#### func (*SharedMemoryOptimizer) OptimizeForConvolution

```go
func (smo *SharedMemoryOptimizer) OptimizeForConvolution(inputShape, kernelShape []int) (*SharedMemoryLayout, error)
```
OptimizeForConvolution optimizes shared memory for convolution operations

#### func (*SharedMemoryOptimizer) OptimizeForMatrixMultiplication

```go
func (smo *SharedMemoryOptimizer) OptimizeForMatrixMultiplication(M, N, K int) (*SharedMemoryLayout, error)
```
OptimizeForMatrixMultiplication optimizes shared memory for matrix
multiplication

#### func (*SharedMemoryOptimizer) OptimizeForReduction

```go
func (smo *SharedMemoryOptimizer) OptimizeForReduction(inputSize int, reductionType string) (*SharedMemoryLayout, error)
```
OptimizeForReduction optimizes shared memory for reduction operations

#### type SimpleMemoryCoalescingOptimizer

```go
type SimpleMemoryCoalescingOptimizer struct {
}
```

SimpleMemoryCoalescingOptimizer optimizes memory access patterns for GPU
efficiency Note: A more advanced version is available in
memory-coalescing-optimizer.go

#### func  GetGlobalSimpleMemoryOptimizer

```go
func GetGlobalSimpleMemoryOptimizer() *SimpleMemoryCoalescingOptimizer
```
GetGlobalSimpleMemoryOptimizer returns the global simple memory coalescing
optimizer

#### func  NewSimpleMemoryCoalescingOptimizer

```go
func NewSimpleMemoryCoalescingOptimizer() *SimpleMemoryCoalescingOptimizer
```
NewSimpleMemoryCoalescingOptimizer creates a new simple memory coalescing
optimizer

#### func (*SimpleMemoryCoalescingOptimizer) OptimizeTensorLayout

```go
func (mco *SimpleMemoryCoalescingOptimizer) OptimizeTensorLayout(tensor *tensor.Tensor, operation string) (*tensor.Tensor, error)
```
OptimizeTensorLayout reorganizes tensor data for optimal GPU access patterns

#### type SimpleTransferOptimizer

```go
type SimpleTransferOptimizer struct {
}
```

GPU-CPU Transfer Optimizer reduces unnecessary memory transfers Note: A more
advanced version is available in gpu-cpu-transfer-optimizer.go

#### func  GetGlobalSimpleTransferOptimizer

```go
func GetGlobalSimpleTransferOptimizer() *SimpleTransferOptimizer
```
GetGlobalSimpleTransferOptimizer returns the global simple transfer optimizer

#### func  NewSimpleTransferOptimizer

```go
func NewSimpleTransferOptimizer() *SimpleTransferOptimizer
```
NewSimpleTransferOptimizer creates a new simple transfer optimizer

#### func (*SimpleTransferOptimizer) CleanupTensor

```go
func (to *SimpleTransferOptimizer) CleanupTensor(tensor *tensor.Tensor)
```
CleanupTensor removes tracking for a tensor

#### func (*SimpleTransferOptimizer) MarkCPUInvalid

```go
func (to *SimpleTransferOptimizer) MarkCPUInvalid(tensor *tensor.Tensor)
```
MarkCPUInvalid marks a tensor's CPU data as invalid

#### func (*SimpleTransferOptimizer) MarkCPUValid

```go
func (to *SimpleTransferOptimizer) MarkCPUValid(tensor *tensor.Tensor)
```
MarkCPUValid marks a tensor as having valid CPU data

#### func (*SimpleTransferOptimizer) MarkGPUInvalid

```go
func (to *SimpleTransferOptimizer) MarkGPUInvalid(tensor *tensor.Tensor)
```
MarkGPUInvalid marks a tensor's GPU data as invalid

#### func (*SimpleTransferOptimizer) MarkGPUValid

```go
func (to *SimpleTransferOptimizer) MarkGPUValid(tensor *tensor.Tensor, operation string)
```
MarkGPUValid marks a tensor as having valid GPU data

#### func (*SimpleTransferOptimizer) ShouldTransferToCPU

```go
func (to *SimpleTransferOptimizer) ShouldTransferToCPU(tensor *tensor.Tensor) bool
```
ShouldTransferToCPU determines if a tensor needs to be transferred to CPU

#### func (*SimpleTransferOptimizer) ShouldTransferToGPU

```go
func (to *SimpleTransferOptimizer) ShouldTransferToGPU(tensor *tensor.Tensor) bool
```
ShouldTransferToGPU determines if a tensor needs to be transferred to GPU

#### type SmoothingBuffer

```go
type SmoothingBuffer struct {
}
```

SmoothingBuffer maintains smoothing history for a parameter

#### type SmoothingType

```go
type SmoothingType int
```

SmoothingType represents different gradient smoothing methods

```go
const (
	NoSmoothing SmoothingType = iota
	MovingAverage
	ExponentialMovingAverage
	MedianFilter
	GaussianFilter
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

#### type StatisticalSummary

```go
type StatisticalSummary struct {
	Mean        float32
	Variance    float32
	Min         float32
	Max         float32
	Percentiles map[int]float32 // 25th, 50th, 75th, 90th, 95th, 99th
	Skewness    float32
	Kurtosis    float32
}
```

StatisticalSummary provides statistical summary of gradient values

#### type StreamID

```go
type StreamID int
```

StreamID represents a specific command queue

```go
const (
	DefaultStream  StreamID = 0
	ComputeStream  StreamID = 1
	TransferStream StreamID = 2
	MaxStreams              = 8
)
```

#### type StreamInfo

```go
type StreamInfo struct {
	Source        string
	Format        string
	Compression   string
	EstimatedSize int64
	LastModified  time.Time
}
```

StreamInfo provides information about a data stream

#### type StreamManager

```go
type StreamManager struct {
}
```

StreamManager manages multiple Metal command queues for parallel execution

#### func  NewStreamManager

```go
func NewStreamManager(device unsafe.Pointer, numStreams int) (*StreamManager, error)
```
NewStreamManager creates a manager for multi-stream execution

#### func (*StreamManager) GetSpecificStream

```go
func (sm *StreamManager) GetSpecificStream(id StreamID) unsafe.Pointer
```
GetSpecificStream returns a specific stream by ID

#### func (*StreamManager) GetStream

```go
func (sm *StreamManager) GetStream() (StreamID, unsafe.Pointer)
```
GetStream returns the least loaded stream for load balancing

#### func (*StreamManager) SubmitToStream

```go
func (sm *StreamManager) SubmitToStream(id StreamID, work func(unsafe.Pointer))
```
SubmitToStream submits work to a specific stream

#### func (*StreamManager) SynchronizeAll

```go
func (sm *StreamManager) SynchronizeAll()
```
SynchronizeAll waits for all streams to complete

#### func (*StreamManager) SynchronizeStream

```go
func (sm *StreamManager) SynchronizeStream(id StreamID)
```
SynchronizeStream waits for a specific stream to complete

#### type StreamProvider

```go
type StreamProvider interface {
	NextBatch() (*BatchData, error)
	Reset() error
	EstimatedBatches() int
	GetStreamInfo() StreamInfo
}
```

StreamProvider interface for streaming data sources

#### type StreamScheduler

```go
type StreamScheduler struct {
}
```

StreamScheduler optimizes task scheduling across streams

#### func  NewStreamScheduler

```go
func NewStreamScheduler(executor *ParallelExecutor, streamMgr *StreamManager) *StreamScheduler
```
NewStreamScheduler creates an optimized task scheduler

#### func (*StreamScheduler) ScheduleDAG

```go
func (ss *StreamScheduler) ScheduleDAG(tasks []StreamTask)
```
ScheduleDAG schedules a directed acyclic graph of tasks

#### type StreamTask

```go
type StreamTask struct {
	ID       int
	StreamID StreamID
	Execute  func(unsafe.Pointer)
	Depends  []int // Task dependencies
	Priority int
}
```

StreamTask represents a GPU computation task

#### type StreamingDataLoader

```go
type StreamingDataLoader struct {
}
```

StreamingDataLoader provides streaming data loading for large datasets

#### func  NewStreamingDataLoader

```go
func NewStreamingDataLoader(config DataLoaderConfig, providers []StreamProvider) (*StreamingDataLoader, error)
```
NewStreamingDataLoader creates a new streaming data loader

#### func (*StreamingDataLoader) GetBatch

```go
func (sdl *StreamingDataLoader) GetBatch(batchIdx int) (*tensor.Tensor, *tensor.Tensor, error)
```
GetBatch returns the next batch from the stream

#### type StressTestResult

```go
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
```

StressTestResult records the results of a stress test scenario

#### type SwapBuffer

```go
type SwapBuffer struct {
}
```

SwapBuffer manages the swap buffer

#### type SwapInfo

```go
type SwapInfo struct {
}
```

SwapInfo tracks swapped tensor information

#### type SwapStats

```go
type SwapStats struct {
	TotalSwapOuts   int64
	TotalSwapIns    int64
	SwapOutBytes    int64
	SwapInBytes     int64
	PrefetchHits    int64
	PrefetchMisses  int64
	AverageSwapTime time.Duration
}
```

SwapStats tracks swapping statistics

#### type Task

```go
type Task struct {
}
```

Task represents a computation task

#### type TaskResult

```go
type TaskResult struct {
}
```

TaskResult represents the result of a task

#### type TaskScheduler

```go
type TaskScheduler struct {
}
```

TaskScheduler schedules tasks for execution

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

#### type TensorLayout

```go
type TensorLayout int
```

TensorLayout represents different memory layouts for tensors

```go
const (
	// Standard layouts
	LayoutRowMajor TensorLayout = iota // Standard row-major layout
	LayoutColMajor                     // Column-major layout

	// Optimized layouts for specific operations
	LayoutNHWC            // Batch, Height, Width, Channels (GPU-friendly for convolution)
	LayoutNCHW            // Batch, Channels, Height, Width (CPU-friendly)
	LayoutHWCN            // Height, Width, Channels, Batch (optimized for some GPU operations)
	LayoutTiled           // Tiled layout for large matrices
	LayoutBlockedRowMajor // Blocked row-major for cache efficiency
	LayoutBlockedColMajor // Blocked column-major for cache efficiency
	LayoutPadded          // Padded layout to avoid bank conflicts
)
```

#### func (TensorLayout) String

```go
func (layout TensorLayout) String() string
```
String returns the string representation of a layout

#### type TensorLayoutInfo

```go
type TensorLayoutInfo struct {
	OriginalShape  []int        // Original tensor shape
	OptimizedShape []int        // Optimized shape (may include padding)
	Layout         TensorLayout // Layout type
	Padding        []int        // Padding added to each dimension
	Stride         []int        // Stride for each dimension
	TileInfo       *TileInfo    // Tiling information if applicable
}
```

TensorLayoutInfo contains information about an optimized tensor layout

#### func  OptimizeTensorForOperation

```go
func OptimizeTensorForOperation(tensor *tensor.Tensor, operation string) (*tensor.Tensor, *TensorLayoutInfo, error)
```
OptimizeTensorForOperation optimizes a tensor layout for a specific operation

#### type TensorLayoutOptimizer

```go
type TensorLayoutOptimizer struct {
}
```

TensorLayoutOptimizer optimizes tensor layouts for specific GPU operations

#### func  GetGlobalLayoutOptimizer

```go
func GetGlobalLayoutOptimizer() *TensorLayoutOptimizer
```
GetGlobalLayoutOptimizer returns the global tensor layout optimizer

#### func  NewTensorLayoutOptimizer

```go
func NewTensorLayoutOptimizer() *TensorLayoutOptimizer
```
NewTensorLayoutOptimizer creates a new tensor layout optimizer

#### func (*TensorLayoutOptimizer) ApplyLayoutOptimization

```go
func (tlo *TensorLayoutOptimizer) ApplyLayoutOptimization(t *tensor.Tensor, operation string) (*tensor.Tensor, *TensorLayoutInfo, error)
```
ApplyLayoutOptimization applies the layout optimization to a tensor

#### func (*TensorLayoutOptimizer) OptimizeLayout

```go
func (tlo *TensorLayoutOptimizer) OptimizeLayout(shape []int, operation string, dataType string) *TensorLayoutInfo
```
OptimizeLayout determines the best layout for a tensor given the operation

#### type TensorType

```go
type TensorType struct {
	Shape    []int
	DataType DataType
	Layout   MemoryLayout
}
```

TensorType describes tensor characteristics

#### func (TensorType) String

```go
func (tt TensorType) String() string
```

#### type TensorUnifiedMemoryAdapter

```go
type TensorUnifiedMemoryAdapter struct {
}
```

TensorUnifiedMemoryAdapter adapts tensors to use unified memory system

#### func  GetGlobalTensorAdapter

```go
func GetGlobalTensorAdapter(device unsafe.Pointer) *TensorUnifiedMemoryAdapter
```
GetGlobalTensorAdapter returns the singleton tensor adapter

#### func  NewTensorUnifiedMemoryAdapter

```go
func NewTensorUnifiedMemoryAdapter(device unsafe.Pointer) *TensorUnifiedMemoryAdapter
```
NewTensorUnifiedMemoryAdapter creates a new adapter

#### func (*TensorUnifiedMemoryAdapter) EnsureUnifiedGPU

```go
func (tuma *TensorUnifiedMemoryAdapter) EnsureUnifiedGPU(t *tensor.Tensor) error
```
EnsureUnifiedGPU ensures tensor is on GPU using unified memory system

#### func (*TensorUnifiedMemoryAdapter) GetUnifiedCPUData

```go
func (tuma *TensorUnifiedMemoryAdapter) GetUnifiedCPUData(t *tensor.Tensor) unsafe.Pointer
```
GetUnifiedCPUData returns CPU data pointer for a tensor using unified memory

#### func (*TensorUnifiedMemoryAdapter) GetUnifiedGPUBuffer

```go
func (tuma *TensorUnifiedMemoryAdapter) GetUnifiedGPUBuffer(t *tensor.Tensor) unsafe.Pointer
```
GetUnifiedGPUBuffer returns the GPU buffer for a tensor using unified memory

#### func (*TensorUnifiedMemoryAdapter) GetUnifiedMemoryStatistics

```go
func (tuma *TensorUnifiedMemoryAdapter) GetUnifiedMemoryStatistics() MemoryStatistics
```
GetUnifiedMemoryStatistics returns unified memory statistics

#### func (*TensorUnifiedMemoryAdapter) ReleaseUnifiedBuffer

```go
func (tuma *TensorUnifiedMemoryAdapter) ReleaseUnifiedBuffer(t *tensor.Tensor)
```
ReleaseUnifiedBuffer releases the unified buffer for a tensor

#### func (*TensorUnifiedMemoryAdapter) SyncTensorToGPU

```go
func (tuma *TensorUnifiedMemoryAdapter) SyncTensorToGPU(t *tensor.Tensor) error
```
SyncTensorToGPU synchronizes tensor data to GPU using zero-copy if possible

#### type TensorView

```go
type TensorView struct {
}
```

TensorView represents a zero-copy view of a tensor

#### type ThreadPool

```go
type ThreadPool struct {
}
```

ThreadPool manages worker threads

#### func  NewThreadPool

```go
func NewThreadPool(numWorkers int) *ThreadPool
```
NewThreadPool creates a new thread pool

#### type TileInfo

```go
type TileInfo struct {
	TileSize   []int // Size of each tile
	NumTiles   []int // Number of tiles in each dimension
	TileStride []int // Stride between tiles
}
```

TileInfo contains tiling information for tiled layouts

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

#### type TransferCacheEntry

```go
type TransferCacheEntry struct {
	Tensor        *tensor.Tensor
	GPUValid      bool
	CPUValid      bool
	LastGPUAccess time.Time
	LastCPUAccess time.Time
	TransferCount int
	IsPinned      bool // Whether tensor uses pinned memory
	LastOperation string
}
```

TransferCacheEntry tracks tensor transfer state

#### type TransferStatistics

```go
type TransferStatistics struct {
	TotalTransfers        int64
	TotalBytesTransferred int64
	CPUToGPUTransfers     int64
	GPUToCPUTransfers     int64
	CacheHits             int64
	CacheMisses           int64
	AverageTransferTime   time.Duration
	PinnedMemoryUsed      int64
}
```

TransferStatistics tracks transfer performance metrics

#### type TuningParameter

```go
type TuningParameter struct {
	Name     string
	Type     string // "int", "float", "choice"
	MinValue float64
	MaxValue float64
	Choices  []interface{}
	Current  interface{}

	// Optimization hints
	Impact      float64 // 0-1 impact on performance
	Granularity float64 // Search granularity
}
```

TuningParameter represents an auto-tuning parameter

#### type TuningStrategy

```go
type TuningStrategy int
```

TuningStrategy defines the optimization approach

```go
const (
	RandomSearch TuningStrategy = iota
	GridSearch
	BayesianOptimization
	GeneticAlgorithm
	SimulatedAnnealing
)
```

#### type TuningTrial

```go
type TuningTrial struct {
	ID          int
	Parameters  map[string]interface{}
	Performance PerformanceMetrics
	Timestamp   time.Time
	Success     bool
}
```

TuningTrial records the result of a parameter configuration trial

#### type UnifiedAccessPattern

```go
type UnifiedAccessPattern struct {
	CPUReads   int
	CPUWrites  int
	GPUReads   int
	GPUWrites  int
	LastAccess int
}
```

UnifiedAccessPattern represents observed access patterns

#### type UnifiedMemoryManager

```go
type UnifiedMemoryManager struct {
}
```

UnifiedMemoryManager optimizes memory usage for Apple Silicon's unified
architecture

#### func  GetGlobalUnifiedMemoryManager

```go
func GetGlobalUnifiedMemoryManager(device unsafe.Pointer) *UnifiedMemoryManager
```
GetGlobalUnifiedMemoryManager returns the singleton unified memory manager

#### func  NewUnifiedMemoryManager

```go
func NewUnifiedMemoryManager(device unsafe.Pointer) *UnifiedMemoryManager
```
NewUnifiedMemoryManager creates a manager for unified memory optimization

#### func (*UnifiedMemoryManager) CreateSharedBuffer

```go
func (umm *UnifiedMemoryManager) CreateSharedBuffer(name string, size int) (*SharedBuffer, error)
```
CreateSharedBuffer creates a zero-copy buffer accessible by CPU and GPU

#### func (*UnifiedMemoryManager) CreateTensorView

```go
func (umm *UnifiedMemoryManager) CreateTensorView(parent *tensor.Tensor, offset, rows, cols int) (*TensorView, error)
```
CreateTensorView creates a zero-copy view of a tensor

#### func (*UnifiedMemoryManager) GetStatistics

```go
func (umm *UnifiedMemoryManager) GetStatistics() MemoryStatistics
```
GetStatistics returns memory management statistics

#### func (*UnifiedMemoryManager) MakeGPUResident

```go
func (umm *UnifiedMemoryManager) MakeGPUResident(bufferName string)
```
MakeGPUResident ensures buffer stays in GPU memory

#### func (*UnifiedMemoryManager) MapModelWeights

```go
func (umm *UnifiedMemoryManager) MapModelWeights(path string) (*MMapRegion, error)
```
MapModelWeights memory-maps model weights for efficient loading

#### func (*UnifiedMemoryManager) OptimizeBatchProcessing

```go
func (umm *UnifiedMemoryManager) OptimizeBatchProcessing(batchSize, featureSize int) *BatchOptimizer
```
OptimizeBatchProcessing optimizes memory for batch processing

#### func (*UnifiedMemoryManager) ReleaseSharedBuffer

```go
func (umm *UnifiedMemoryManager) ReleaseSharedBuffer(buf *SharedBuffer)
```
ReleaseSharedBuffer releases a shared buffer

#### type ValidationCallback

```go
type ValidationCallback func(epoch int, batch int, validationLoss float32) error
```

ValidationCallback is called during validation

#### type Worker

```go
type Worker struct {
}
```

Worker represents a worker thread

#### type WorkerLoad

```go
type WorkerLoad struct {
}
```

WorkerLoad tracks worker load

#### type WorkerStatistics

```go
type WorkerStatistics struct {
	TasksCompleted     int64
	TotalExecutionTime time.Duration
	AverageTaskTime    time.Duration
	ErrorCount         int64
	IdleTime           time.Duration
}
```

WorkerStatistics tracks worker performance

#### type ZeroCopyMixedPrecisionOps

```go
type ZeroCopyMixedPrecisionOps struct {
}
```

ZeroCopyMixedPrecisionOps provides zero-copy mixed precision operations

#### func  NewZeroCopyMixedPrecisionOps

```go
func NewZeroCopyMixedPrecisionOps(config *MixedPrecisionConfig) (*ZeroCopyMixedPrecisionOps, error)
```
NewZeroCopyMixedPrecisionOps creates zero-copy mixed precision operations

#### func (*ZeroCopyMixedPrecisionOps) BatchOptimalMatMul

```go
func (zc *ZeroCopyMixedPrecisionOps) BatchOptimalMatMul(operations []MatMulOperation) ([]*tensor.Tensor, error)
```
BatchOptimalMatMul processes multiple matrix operations with optimal precision
selection

#### func (*ZeroCopyMixedPrecisionOps) Cleanup

```go
func (zc *ZeroCopyMixedPrecisionOps) Cleanup()
```
Cleanup releases resources

#### func (*ZeroCopyMixedPrecisionOps) OptimalMatMul

```go
func (zc *ZeroCopyMixedPrecisionOps) OptimalMatMul(A, B *tensor.Tensor) (*tensor.Tensor, error)
```
OptimalMatMul chooses the best precision and implementation automatically

#### func (*ZeroCopyMixedPrecisionOps) ProfileOperation

```go
func (zc *ZeroCopyMixedPrecisionOps) ProfileOperation(A, B *tensor.Tensor, iterations int) (*PerformanceProfile, error)
```
ProfileOperation provides detailed performance analysis for a specific operation

#### type ZeroCopyTransfer

```go
type ZeroCopyTransfer struct {
}
```

ZeroCopyTransfer performs zero-copy data transfer between CPU and GPU

#### func (*ZeroCopyTransfer) TransferFromGPU

```go
func (zct *ZeroCopyTransfer) TransferFromGPU(gpuBuffer unsafe.Pointer, size int) ([]float32, error)
```
TransferFromGPU transfers data from GPU without copying (if possible)

#### func (*ZeroCopyTransfer) TransferToGPU

```go
func (zct *ZeroCopyTransfer) TransferToGPU(data []float32, name string) (unsafe.Pointer, error)
```
TransferToGPU transfers data to GPU without copying (if possible)
