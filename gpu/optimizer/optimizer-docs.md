# optimizer
--
    import "."


## Usage

#### func  ClipGradsByNorm

```go
func ClipGradsByNorm(grads []*tensor.Tensor, maxNorm float32) (float32, error)
```
ClipGradsByNorm clips gradients by global norm

#### func  ClipGradsByValue

```go
func ClipGradsByValue(grads []*tensor.Tensor, minValue, maxValue float32) error
```
ClipGradsByValue clips gradients by value

#### func  ComputeGradNorm

```go
func ComputeGradNorm(grads []*tensor.Tensor) (float32, error)
```
ComputeGradNorm computes the global gradient norm

#### type AdamConfig

```go
type AdamConfig struct {
	OptimizerConfig
	Beta1   float32
	Beta2   float32
	Epsilon float32
}
```

AdamConfig holds configuration specific to Adam optimizer

#### type AdamOptimizer

```go
type AdamOptimizer struct {
}
```

AdamOptimizer implements the Adam optimization algorithm

#### func  NewAdam

```go
func NewAdam(config AdamConfig) *AdamOptimizer
```
NewAdam creates a new Adam optimizer

#### func (*AdamOptimizer) GetLearningRate

```go
func (opt *AdamOptimizer) GetLearningRate() float32
```
GetLearningRate returns the current learning rate

#### func (*AdamOptimizer) GetStepCount

```go
func (opt *AdamOptimizer) GetStepCount() int64
```
GetStepCount returns the current step count

#### func (*AdamOptimizer) ReleaseGPU

```go
func (opt *AdamOptimizer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*AdamOptimizer) SetLearningRate

```go
func (opt *AdamOptimizer) SetLearningRate(lr float32)
```
SetLearningRate sets the learning rate

#### func (*AdamOptimizer) Step

```go
func (opt *AdamOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error
```
Step performs one optimization step

#### func (*AdamOptimizer) ZeroGrad

```go
func (opt *AdamOptimizer) ZeroGrad(grads []*tensor.Tensor) error
```
ZeroGrad zeros all gradients

#### type AdamWConfig

```go
type AdamWConfig struct {
	OptimizerConfig
	Beta1   float32
	Beta2   float32
	Epsilon float32
}
```

AdamWConfig holds configuration specific to AdamW optimizer

#### type AdamWOptimizer

```go
type AdamWOptimizer struct {
}
```

AdamWOptimizer implements the AdamW optimization algorithm (Adam with decoupled
weight decay)

#### func  NewAdamW

```go
func NewAdamW(config AdamWConfig) *AdamWOptimizer
```
NewAdamW creates a new AdamW optimizer

#### func (*AdamWOptimizer) GetLearningRate

```go
func (opt *AdamWOptimizer) GetLearningRate() float32
```
GetLearningRate returns the current learning rate

#### func (*AdamWOptimizer) GetStepCount

```go
func (opt *AdamWOptimizer) GetStepCount() int64
```
GetStepCount returns the current step count

#### func (*AdamWOptimizer) ReleaseGPU

```go
func (opt *AdamWOptimizer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*AdamWOptimizer) SetLearningRate

```go
func (opt *AdamWOptimizer) SetLearningRate(lr float32)
```
SetLearningRate sets the learning rate

#### func (*AdamWOptimizer) Step

```go
func (opt *AdamWOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error
```
Step performs one optimization step

#### func (*AdamWOptimizer) ZeroGrad

```go
func (opt *AdamWOptimizer) ZeroGrad(grads []*tensor.Tensor) error
```
ZeroGrad zeros all gradients

#### type ChainedScheduler

```go
type ChainedScheduler struct {
}
```

ChainedScheduler allows chaining multiple schedulers together

#### func  NewChainedScheduler

```go
func NewChainedScheduler(schedulers []LRScheduler, milestones []int64) *ChainedScheduler
```
NewChainedScheduler creates a new chained scheduler

#### func (*ChainedScheduler) GetLR

```go
func (s *ChainedScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*ChainedScheduler) SetOptimizer

```go
func (s *ChainedScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer for all schedulers

#### func (*ChainedScheduler) Step

```go
func (s *ChainedScheduler) Step(step int64) error
```
Step updates the learning rate using the appropriate scheduler

#### type CosineAnnealingScheduler

```go
type CosineAnnealingScheduler struct {
}
```

CosineAnnealingScheduler implements cosine annealing learning rate scheduling

#### func  NewCosineAnnealingScheduler

```go
func NewCosineAnnealingScheduler(initialLR, minLR float32, totalSteps int64) *CosineAnnealingScheduler
```
NewCosineAnnealingScheduler creates a new cosine annealing scheduler

#### func (*CosineAnnealingScheduler) GetLR

```go
func (s *CosineAnnealingScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*CosineAnnealingScheduler) SetOptimizer

```go
func (s *CosineAnnealingScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*CosineAnnealingScheduler) Step

```go
func (s *CosineAnnealingScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step

#### type CyclicLRScheduler

```go
type CyclicLRScheduler struct {
}
```

CyclicLRScheduler implements cyclical learning rates

#### func  NewCyclicLRScheduler

```go
func NewCyclicLRScheduler(baseLR, maxLR float32, stepSizeUp, stepSizeDown int64, mode string, gamma float32) *CyclicLRScheduler
```
NewCyclicLRScheduler creates a new cyclical LR scheduler

#### func (*CyclicLRScheduler) GetLR

```go
func (s *CyclicLRScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*CyclicLRScheduler) SetOptimizer

```go
func (s *CyclicLRScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*CyclicLRScheduler) Step

```go
func (s *CyclicLRScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step

#### type ExponentialDecayScheduler

```go
type ExponentialDecayScheduler struct {
}
```

ExponentialDecayScheduler implements exponential decay learning rate scheduling

#### func  NewExponentialDecayScheduler

```go
func NewExponentialDecayScheduler(initialLR, decayRate float32, decaySteps int64) *ExponentialDecayScheduler
```
NewExponentialDecayScheduler creates a new exponential decay scheduler

#### func (*ExponentialDecayScheduler) GetLR

```go
func (s *ExponentialDecayScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*ExponentialDecayScheduler) SetOptimizer

```go
func (s *ExponentialDecayScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*ExponentialDecayScheduler) Step

```go
func (s *ExponentialDecayScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step

#### type LRScheduler

```go
type LRScheduler interface {
	Step(step int64) error
	GetLR() float32
	SetOptimizer(opt Optimizer)
}
```

LRScheduler represents a learning rate scheduler interface

#### func  NewWarmupCosineScheduler

```go
func NewWarmupCosineScheduler(initialLR, maxLR, minLR float32, warmupSteps, totalSteps int64) LRScheduler
```
Helper function to create a common warmup + cosine annealing scheduler

#### func  NewWarmupExponentialScheduler

```go
func NewWarmupExponentialScheduler(initialLR, maxLR, decayRate float32, warmupSteps, decaySteps int64) LRScheduler
```
Helper function to create a common warmup + exponential decay scheduler

#### type OneCycleLRScheduler

```go
type OneCycleLRScheduler struct {
}
```

OneCycleLRScheduler implements the one-cycle learning rate policy

#### func  NewOneCycleLRScheduler

```go
func NewOneCycleLRScheduler(maxLR float32, totalSteps int64, pctStart float32, annealStrategy string) *OneCycleLRScheduler
```
NewOneCycleLRScheduler creates a new one-cycle LR scheduler

#### func (*OneCycleLRScheduler) GetLR

```go
func (s *OneCycleLRScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*OneCycleLRScheduler) SetOptimizer

```go
func (s *OneCycleLRScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*OneCycleLRScheduler) Step

```go
func (s *OneCycleLRScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step

#### type Optimizer

```go
type Optimizer interface {
	Step(params []*tensor.Tensor, grads []*tensor.Tensor) error
	ZeroGrad(grads []*tensor.Tensor) error
	GetLearningRate() float32
	SetLearningRate(lr float32)
	GetStepCount() int64
	ReleaseGPU()
}
```

Optimizer represents a generic optimizer interface

#### type OptimizerConfig

```go
type OptimizerConfig struct {
	LearningRate float32
	WeightDecay  float32
}
```

OptimizerConfig holds common configuration for all optimizers

#### type PolynomialDecayScheduler

```go
type PolynomialDecayScheduler struct {
}
```

PolynomialDecayScheduler implements polynomial decay learning rate scheduling

#### func  NewPolynomialDecayScheduler

```go
func NewPolynomialDecayScheduler(initialLR, finalLR float32, totalSteps int64, power float32) *PolynomialDecayScheduler
```
NewPolynomialDecayScheduler creates a new polynomial decay scheduler

#### func (*PolynomialDecayScheduler) GetLR

```go
func (s *PolynomialDecayScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*PolynomialDecayScheduler) SetOptimizer

```go
func (s *PolynomialDecayScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*PolynomialDecayScheduler) Step

```go
func (s *PolynomialDecayScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step

#### type RMSpropConfig

```go
type RMSpropConfig struct {
	OptimizerConfig
	Alpha    float32 // Smoothing constant
	Epsilon  float32
	Momentum float32
}
```

RMSpropConfig holds configuration specific to RMSprop optimizer

#### type RMSpropOptimizer

```go
type RMSpropOptimizer struct {
}
```

RMSpropOptimizer implements the RMSprop optimization algorithm

#### func  NewRMSprop

```go
func NewRMSprop(config RMSpropConfig) *RMSpropOptimizer
```
NewRMSprop creates a new RMSprop optimizer

#### func (*RMSpropOptimizer) GetLearningRate

```go
func (opt *RMSpropOptimizer) GetLearningRate() float32
```
GetLearningRate returns the current learning rate

#### func (*RMSpropOptimizer) GetStepCount

```go
func (opt *RMSpropOptimizer) GetStepCount() int64
```
GetStepCount returns the current step count

#### func (*RMSpropOptimizer) ReleaseGPU

```go
func (opt *RMSpropOptimizer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*RMSpropOptimizer) SetLearningRate

```go
func (opt *RMSpropOptimizer) SetLearningRate(lr float32)
```
SetLearningRate sets the learning rate

#### func (*RMSpropOptimizer) Step

```go
func (opt *RMSpropOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error
```
Step performs one optimization step

#### func (*RMSpropOptimizer) ZeroGrad

```go
func (opt *RMSpropOptimizer) ZeroGrad(grads []*tensor.Tensor) error
```
ZeroGrad zeros all gradients

#### type ReduceLROnPlateauScheduler

```go
type ReduceLROnPlateauScheduler struct {
}
```

ReduceLROnPlateauScheduler reduces learning rate when a metric has stopped
improving

#### func  NewReduceLROnPlateauScheduler

```go
func NewReduceLROnPlateauScheduler(initialLR, factor float32, patience int64, threshold, minLR float32, mode string) *ReduceLROnPlateauScheduler
```
NewReduceLROnPlateauScheduler creates a new ReduceLROnPlateau scheduler

#### func (*ReduceLROnPlateauScheduler) GetLR

```go
func (s *ReduceLROnPlateauScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*ReduceLROnPlateauScheduler) SetOptimizer

```go
func (s *ReduceLROnPlateauScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*ReduceLROnPlateauScheduler) Step

```go
func (s *ReduceLROnPlateauScheduler) Step(step int64) error
```
Step is required by the interface but does nothing for this scheduler

#### func (*ReduceLROnPlateauScheduler) StepWithMetric

```go
func (s *ReduceLROnPlateauScheduler) StepWithMetric(metric float32) error
```
Step updates the learning rate based on the metric value

#### type SGDConfig

```go
type SGDConfig struct {
	OptimizerConfig
	Momentum float32
}
```

SGDConfig holds configuration specific to SGD optimizer

#### type SGDOptimizer

```go
type SGDOptimizer struct {
}
```

SGDOptimizer implements Stochastic Gradient Descent with momentum

#### func  NewSGD

```go
func NewSGD(config SGDConfig) *SGDOptimizer
```
NewSGD creates a new SGD optimizer

#### func (*SGDOptimizer) GetLearningRate

```go
func (opt *SGDOptimizer) GetLearningRate() float32
```
GetLearningRate returns the current learning rate

#### func (*SGDOptimizer) GetStepCount

```go
func (opt *SGDOptimizer) GetStepCount() int64
```
GetStepCount returns the current step count

#### func (*SGDOptimizer) ReleaseGPU

```go
func (opt *SGDOptimizer) ReleaseGPU()
```
ReleaseGPU releases GPU resources

#### func (*SGDOptimizer) SetLearningRate

```go
func (opt *SGDOptimizer) SetLearningRate(lr float32)
```
SetLearningRate sets the learning rate

#### func (*SGDOptimizer) Step

```go
func (opt *SGDOptimizer) Step(params []*tensor.Tensor, grads []*tensor.Tensor) error
```
Step performs one optimization step

#### func (*SGDOptimizer) ZeroGrad

```go
func (opt *SGDOptimizer) ZeroGrad(grads []*tensor.Tensor) error
```
ZeroGrad zeros all gradients

#### type StepDecayScheduler

```go
type StepDecayScheduler struct {
}
```

StepDecayScheduler implements step decay learning rate scheduling

#### func  NewStepDecayScheduler

```go
func NewStepDecayScheduler(initialLR, gamma float32, stepSize int64) *StepDecayScheduler
```
NewStepDecayScheduler creates a new step decay scheduler

#### func (*StepDecayScheduler) GetLR

```go
func (s *StepDecayScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*StepDecayScheduler) SetOptimizer

```go
func (s *StepDecayScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*StepDecayScheduler) Step

```go
func (s *StepDecayScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step

#### type WarmupScheduler

```go
type WarmupScheduler struct {
}
```

WarmupScheduler implements learning rate warmup

#### func  NewWarmupScheduler

```go
func NewWarmupScheduler(targetLR float32, warmupSteps int64, warmupType int) *WarmupScheduler
```
NewWarmupScheduler creates a new warmup scheduler

#### func (*WarmupScheduler) GetLR

```go
func (s *WarmupScheduler) GetLR() float32
```
GetLR returns the current learning rate

#### func (*WarmupScheduler) SetBaseScheduler

```go
func (s *WarmupScheduler) SetBaseScheduler(scheduler LRScheduler)
```
SetBaseScheduler sets a base scheduler to use after warmup completes

#### func (*WarmupScheduler) SetOptimizer

```go
func (s *WarmupScheduler) SetOptimizer(opt Optimizer)
```
SetOptimizer sets the optimizer to update

#### func (*WarmupScheduler) Step

```go
func (s *WarmupScheduler) Step(step int64) error
```
Step updates the learning rate based on the current step
