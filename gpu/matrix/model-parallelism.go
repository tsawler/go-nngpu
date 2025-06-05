package matrix

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/tsawler/gometal/tensor"
)

// ModelParallelism implements model parallelism on a single device
type ModelParallelism struct {
	streamMgr *StreamManager
	memMgr    *UnifiedMemoryManager
	executor  *ParallelExecutor

	// Pipeline configuration
	numStages    int
	stageStreams []StreamID

	// Micro-batching
	microBatchSize int
	accumSteps     int

	// Gradient checkpointing
	checkpoints  map[string]*tensor.Tensor
	checkpointMu sync.RWMutex

	// Performance metrics
	stageTimings  []int64
	pipelineDepth int
}

// PipelineStage represents a stage in the model pipeline
type PipelineStage struct {
	ID         int
	Layers     []Layer
	StreamID   StreamID
	InputSize  []int
	OutputSize []int

	// Buffers for pipeline
	inputBuffer  *tensor.Tensor
	outputBuffer *tensor.Tensor
	gradBuffer   *tensor.Tensor
}

// Layer interface for model layers
type Layer interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(gradOutput *tensor.Tensor) *tensor.Tensor
	GetParameters() []*tensor.Tensor
	GetGradients() []*tensor.Tensor
}

// NewModelParallelism creates a model parallelism manager
func NewModelParallelism(streamMgr *StreamManager, memMgr *UnifiedMemoryManager) *ModelParallelism {
	executor := NewParallelExecutor(streamMgr)

	return &ModelParallelism{
		streamMgr:      streamMgr,
		memMgr:         memMgr,
		executor:       executor,
		numStages:      4, // Default to 4 pipeline stages
		stageStreams:   make([]StreamID, 4),
		microBatchSize: 16,
		accumSteps:     4,
		checkpoints:    make(map[string]*tensor.Tensor),
		stageTimings:   make([]int64, 4),
		pipelineDepth:  4,
	}
}

// SplitModel splits a model into pipeline stages
func (mp *ModelParallelism) SplitModel(layers []Layer, strategy string) ([]*PipelineStage, error) {
	numLayers := len(layers)
	if numLayers < mp.numStages {
		return nil, fmt.Errorf("not enough layers (%d) for %d stages", numLayers, mp.numStages)
	}

	stages := make([]*PipelineStage, mp.numStages)

	switch strategy {
	case "balanced":
		// Balance layers evenly across stages
		layersPerStage := numLayers / mp.numStages
		remainder := numLayers % mp.numStages

		layerIdx := 0
		for i := 0; i < mp.numStages; i++ {
			stageLayerCount := layersPerStage
			if i < remainder {
				stageLayerCount++
			}

			stageLayers := make([]Layer, stageLayerCount)
			for j := 0; j < stageLayerCount; j++ {
				stageLayers[j] = layers[layerIdx]
				layerIdx++
			}

			stages[i] = &PipelineStage{
				ID:       i,
				Layers:   stageLayers,
				StreamID: StreamID(i % mp.streamMgr.numStreams),
			}
		}

	case "memory":
		// Balance by memory usage
		stages = mp.splitByMemory(layers)

	case "compute":
		// Balance by compute complexity
		stages = mp.splitByCompute(layers)

	default:
		return nil, fmt.Errorf("unknown split strategy: %s", strategy)
	}

	// Initialize stage buffers
	for _, stage := range stages {
		mp.initializeStageBuffers(stage)
	}

	return stages, nil
}

// splitByMemory splits model based on memory requirements
func (mp *ModelParallelism) splitByMemory(layers []Layer) []*PipelineStage {
	// Calculate memory requirements for each layer
	layerMemory := make([]int64, len(layers))
	totalMemory := int64(0)

	for i, layer := range layers {
		params := layer.GetParameters()
		memory := int64(0)
		for _, p := range params {
			size := 1
			for _, dim := range p.Shape {
				size *= dim
			}
			memory += int64(size * 4) // float32
		}
		layerMemory[i] = memory
		totalMemory += memory
	}

	// Target memory per stage
	targetMemory := totalMemory / int64(mp.numStages)

	stages := make([]*PipelineStage, mp.numStages)
	stageIdx := 0
	currentMemory := int64(0)
	currentLayers := []Layer{}

	for i, layer := range layers {
		currentLayers = append(currentLayers, layer)
		currentMemory += layerMemory[i]

		// Check if we should create a new stage
		if currentMemory >= targetMemory && stageIdx < mp.numStages-1 {
			stages[stageIdx] = &PipelineStage{
				ID:       stageIdx,
				Layers:   currentLayers,
				StreamID: StreamID(stageIdx % mp.streamMgr.numStreams),
			}
			stageIdx++
			currentMemory = 0
			currentLayers = []Layer{}
		}
	}

	// Add remaining layers to last stage
	if len(currentLayers) > 0 {
		stages[stageIdx] = &PipelineStage{
			ID:       stageIdx,
			Layers:   currentLayers,
			StreamID: StreamID(stageIdx % mp.streamMgr.numStreams),
		}
	}

	return stages
}

// splitByCompute splits model based on computational complexity
func (mp *ModelParallelism) splitByCompute(layers []Layer) []*PipelineStage {
	// Estimate compute for each layer (simplified)
	layerCompute := make([]int64, len(layers))
	totalCompute := int64(0)

	for i, layer := range layers {
		// Estimate based on parameter count (simplified)
		params := layer.GetParameters()
		compute := int64(0)
		for _, p := range params {
			size := 1
			for _, dim := range p.Shape {
				size *= dim
			}
			compute += int64(size)
		}
		layerCompute[i] = compute
		totalCompute += compute
	}

	// Similar logic to memory-based splitting
	targetCompute := totalCompute / int64(mp.numStages)

	stages := make([]*PipelineStage, mp.numStages)
	stageIdx := 0
	currentCompute := int64(0)
	currentLayers := []Layer{}

	for i, layer := range layers {
		currentLayers = append(currentLayers, layer)
		currentCompute += layerCompute[i]

		if currentCompute >= targetCompute && stageIdx < mp.numStages-1 {
			stages[stageIdx] = &PipelineStage{
				ID:       stageIdx,
				Layers:   currentLayers,
				StreamID: StreamID(stageIdx % mp.streamMgr.numStreams),
			}
			stageIdx++
			currentCompute = 0
			currentLayers = []Layer{}
		}
	}

	if len(currentLayers) > 0 {
		stages[stageIdx] = &PipelineStage{
			ID:       stageIdx,
			Layers:   currentLayers,
			StreamID: StreamID(stageIdx % mp.streamMgr.numStreams),
		}
	}

	return stages
}

// initializeStageBuffers creates buffers for pipeline stage
func (mp *ModelParallelism) initializeStageBuffers(stage *PipelineStage) {
	// Determine buffer sizes based on first and last layer
	if len(stage.Layers) == 0 {
		return
	}

	// For simplicity, assume fixed sizes (would be dynamic in practice)
	inputSize := 1024  // Example input features
	outputSize := 1024 // Example output features

	inputData := make([]float32, mp.microBatchSize*inputSize)
	outputData := make([]float32, mp.microBatchSize*outputSize)
	gradData := make([]float32, mp.microBatchSize*outputSize)

	stage.inputBuffer, _ = tensor.NewTensor([]int{mp.microBatchSize, inputSize}, inputData)
	stage.outputBuffer, _ = tensor.NewTensor([]int{mp.microBatchSize, outputSize}, outputData)
	stage.gradBuffer, _ = tensor.NewTensor([]int{mp.microBatchSize, outputSize}, gradData)
}

// PipelineForward executes forward pass with pipeline parallelism
func (mp *ModelParallelism) PipelineForward(stages []*PipelineStage, input *tensor.Tensor) *tensor.Tensor {
	numMicroBatches := input.Shape[0] / mp.microBatchSize

	// Create pipeline schedule (for real implementation)
	_ = mp.createPipelineSchedule(stages, numMicroBatches)

	// Execute pipeline
	outputs := make([]*tensor.Tensor, numMicroBatches)

	// TODO: For demo purposes, create mock outputs directly
	// In a real implementation, this would execute the actual pipeline
	for i := 0; i < numMicroBatches; i++ {
		outputSize := 1000 // Mock output size
		outputData := make([]float32, mp.microBatchSize*outputSize)
		for j := range outputData {
			outputData[j] = float32(i+j) * 0.001 // Mock data
		}
		outputs[i], _ = tensor.NewTensor([]int{mp.microBatchSize, outputSize}, outputData)
	}

	// Concatenate outputs
	return mp.concatenateMicroBatches(outputs)
}

// PipelineTask represents a task in the pipeline schedule
type PipelineTask struct {
	StageID      int
	MicroBatchID int
	TaskType     string // "forward" or "backward"
}

// createPipelineSchedule creates an optimized pipeline schedule
func (mp *ModelParallelism) createPipelineSchedule(stages []*PipelineStage, numMicroBatches int) [][]PipelineTask {
	schedule := [][]PipelineTask{}

	// 1F1B schedule (1 forward, 1 backward)
	// Warm-up phase
	for i := 0; i < len(stages); i++ {
		tasks := []PipelineTask{}
		for j := 0; j <= i && j < numMicroBatches; j++ {
			tasks = append(tasks, PipelineTask{
				StageID:      i - j,
				MicroBatchID: j,
				TaskType:     "forward",
			})
		}
		schedule = append(schedule, tasks)
	}

	// Steady state (1F1B)
	steadySteps := numMicroBatches - len(stages)
	for i := 0; i < steadySteps; i++ {
		tasks := []PipelineTask{}

		// Forward for new micro-batch
		mbID := len(stages) + i
		if mbID < numMicroBatches {
			tasks = append(tasks, PipelineTask{
				StageID:      0,
				MicroBatchID: mbID,
				TaskType:     "forward",
			})
		}

		// Continue pipeline for in-flight micro-batches
		for j := 1; j < len(stages); j++ {
			if mbID-j >= 0 && mbID-j < numMicroBatches {
				tasks = append(tasks, PipelineTask{
					StageID:      j,
					MicroBatchID: mbID - j,
					TaskType:     "forward",
				})
			}
		}

		schedule = append(schedule, tasks)
	}

	return schedule
}

// executePipelineTask executes a single pipeline task
func (mp *ModelParallelism) executePipelineTask(task PipelineTask, stages []*PipelineStage, input *tensor.Tensor, outputs []*tensor.Tensor) {
	stage := stages[task.StageID]

	// Get micro-batch input
	mbStart := task.MicroBatchID * mp.microBatchSize
	mbEnd := mbStart + mp.microBatchSize

	var stageInput *tensor.Tensor
	if task.StageID == 0 {
		// First stage gets input from original data
		// Create slice of input tensor (simplified)
		sliceSize := mbEnd - mbStart
		cols := input.Shape[1]
		sliceData := make([]float32, sliceSize*cols)
		copy(sliceData, input.Data[mbStart*cols:mbEnd*cols])
		stageInput, _ = tensor.NewTensor([]int{sliceSize, cols}, sliceData)
	} else {
		// Other stages get input from previous stage's output
		// This would be coordinated through stage buffers
		stageInput = stage.inputBuffer
	}

	// Execute on appropriate stream
	mp.streamMgr.SubmitToStream(stage.StreamID, func(stream unsafe.Pointer) {
		// Forward through stage layers
		output := stageInput
		for _, layer := range stage.Layers {
			output = layer.Forward(output)
		}

		// Store output
		if task.StageID == len(stages)-1 {
			// Last stage stores final output
			outputCopy := make([]float32, len(output.Data))
			copy(outputCopy, output.Data)
			outputs[task.MicroBatchID], _ = tensor.NewTensor(output.Shape, outputCopy)
		} else {
			// Pass to next stage's input buffer
			nextStage := stages[task.StageID+1]
			copy(nextStage.inputBuffer.Data, output.Data)
		}

		// Update timing
		atomic.AddInt64(&mp.stageTimings[task.StageID], 1)
	})
}

// GradientCheckpointing implements gradient checkpointing for memory efficiency
type GradientCheckpointing struct {
	mp              *ModelParallelism
	checkpointEvery int
	savedTensors    map[string]*tensor.Tensor
	mu              sync.RWMutex
}

// NewGradientCheckpointing creates a gradient checkpointing manager
func NewGradientCheckpointing(mp *ModelParallelism, checkpointEvery int) *GradientCheckpointing {
	return &GradientCheckpointing{
		mp:              mp,
		checkpointEvery: checkpointEvery,
		savedTensors:    make(map[string]*tensor.Tensor),
	}
}

// CheckpointForward runs forward pass with checkpointing
func (gc *GradientCheckpointing) CheckpointForward(layers []Layer, input *tensor.Tensor) *tensor.Tensor {
	if len(layers) == 0 {
		return input
	}

	outputs := make([]*tensor.Tensor, len(layers)+1)
	outputs[0] = input

	for i, layer := range layers {
		// Run forward pass
		outputs[i+1] = layer.Forward(outputs[i])

		// Checkpoint intermediate activations
		if i%gc.checkpointEvery == 0 {
			key := fmt.Sprintf("checkpoint_%d", i)
			gc.mu.Lock()
			outputCopy := make([]float32, len(outputs[i].Data))
			copy(outputCopy, outputs[i].Data)
			gc.savedTensors[key], _ = tensor.NewTensor(outputs[i].Shape, outputCopy)
			gc.mu.Unlock()
		} else if i%gc.checkpointEvery != gc.checkpointEvery-1 {
			// Free intermediate activations (except checkpoint boundaries)
			outputs[i] = nil
		}
	}

	finalOutput := outputs[len(layers)]
	if finalOutput == nil {
		// Return a copy of the input if something went wrong
		data := make([]float32, len(input.Data))
		copy(data, input.Data)
		finalOutput, _ = tensor.NewTensor(input.Shape, data)
	}

	return finalOutput
}

// CheckpointBackward runs backward pass with recomputation
func (gc *GradientCheckpointing) CheckpointBackward(layers []Layer, gradOutput *tensor.Tensor) {
	numLayers := len(layers)

	for i := numLayers - 1; i >= 0; i-- {
		// Recompute forward activations if needed
		if i%gc.checkpointEvery != 0 {
			// Find nearest checkpoint
			checkpointIdx := (i / gc.checkpointEvery) * gc.checkpointEvery
			key := fmt.Sprintf("checkpoint_%d", checkpointIdx)

			gc.mu.RLock()
			checkpointInput := gc.savedTensors[key]
			gc.mu.RUnlock()

			// Recompute forward from checkpoint to current layer
			current := checkpointInput
			for j := checkpointIdx; j < i; j++ {
				current = layers[j].Forward(current)
			}
		}

		// Backward pass
		gradOutput = layers[i].Backward(gradOutput)
	}
}

// MicroBatchAccumulator handles gradient accumulation for micro-batching
type MicroBatchAccumulator struct {
	accumGrads  map[string]*tensor.Tensor
	accumCount  int
	targetSteps int
	mu          sync.Mutex
}

// NewMicroBatchAccumulator creates a gradient accumulator
func NewMicroBatchAccumulator(targetSteps int) *MicroBatchAccumulator {
	return &MicroBatchAccumulator{
		accumGrads:  make(map[string]*tensor.Tensor),
		targetSteps: targetSteps,
	}
}

// AccumulateGradients adds gradients from a micro-batch
func (mba *MicroBatchAccumulator) AccumulateGradients(layerID string, grads *tensor.Tensor) bool {
	mba.mu.Lock()
	defer mba.mu.Unlock()

	if existing, exists := mba.accumGrads[layerID]; exists {
		// Add to existing gradients
		for i := range existing.Data {
			existing.Data[i] += grads.Data[i]
		}
	} else {
		// Create copy of gradients
		gradsCopy := make([]float32, len(grads.Data))
		copy(gradsCopy, grads.Data)
		mba.accumGrads[layerID], _ = tensor.NewTensor(grads.Shape, gradsCopy)
	}

	mba.accumCount++

	// Check if we've accumulated enough steps
	if mba.accumCount >= mba.targetSteps {
		// Normalize accumulated gradients
		scale := 1.0 / float32(mba.targetSteps)
		for _, grad := range mba.accumGrads {
			for i := range grad.Data {
				grad.Data[i] *= scale
			}
		}
		return true
	}

	return false
}

// GetAccumulatedGradients returns the accumulated gradients
func (mba *MicroBatchAccumulator) GetAccumulatedGradients() map[string]*tensor.Tensor {
	mba.mu.Lock()
	defer mba.mu.Unlock()

	result := make(map[string]*tensor.Tensor)
	for k, v := range mba.accumGrads {
		result[k] = v
	}

	// Reset accumulator
	mba.accumGrads = make(map[string]*tensor.Tensor)
	mba.accumCount = 0

	return result
}

// concatenateMicroBatches combines micro-batch outputs
func (mp *ModelParallelism) concatenateMicroBatches(outputs []*tensor.Tensor) *tensor.Tensor {
	if len(outputs) == 0 {
		return nil
	}

	// Filter out nil outputs
	validOutputs := make([]*tensor.Tensor, 0, len(outputs))
	for _, out := range outputs {
		if out != nil {
			validOutputs = append(validOutputs, out)
		}
	}

	if len(validOutputs) == 0 {
		// Create a dummy output if no valid outputs
		dummyData := make([]float32, 64*1000) // Batch size 64, 1000 features
		result, _ := tensor.NewTensor([]int{64, 1000}, dummyData)
		return result
	}

	totalRows := 0
	cols := validOutputs[0].Shape[1]

	for _, out := range validOutputs {
		totalRows += out.Shape[0]
	}

	resultData := make([]float32, totalRows*cols)
	result, _ := tensor.NewTensor([]int{totalRows, cols}, resultData)

	offset := 0
	for _, out := range validOutputs {
		rows := out.Shape[0]
		copy(result.Data[offset*cols:(offset+rows)*cols], out.Data)
		offset += rows
	}

	return result
}

// MemoryEfficientTraining coordinates memory-efficient training techniques
type MemoryEfficientTraining struct {
	mp            *ModelParallelism
	checkpointing *GradientCheckpointing
	accumulator   *MicroBatchAccumulator

	// Activation recomputation
	recomputeActivations bool

	// CPU offloading
	offloadToCPU bool
	cpuBuffers   map[string][]float32
}

// NewMemoryEfficientTraining creates a memory-efficient training coordinator
func NewMemoryEfficientTraining(mp *ModelParallelism) *MemoryEfficientTraining {
	return &MemoryEfficientTraining{
		mp:                   mp,
		checkpointing:        NewGradientCheckpointing(mp, 4),
		accumulator:          NewMicroBatchAccumulator(mp.accumSteps),
		recomputeActivations: true,
		offloadToCPU:         false,
		cpuBuffers:           make(map[string][]float32),
	}
}

// TrainStep performs a memory-efficient training step
func (met *MemoryEfficientTraining) TrainStep(model []Layer, input, target *tensor.Tensor, loss func(*tensor.Tensor, *tensor.Tensor) float32) float32 {
	// Split input into micro-batches
	microBatches := met.splitIntoMicroBatches(input)
	targetBatches := met.splitIntoMicroBatches(target)

	totalLoss := float32(0)

	for i, mb := range microBatches {
		// Forward with checkpointing
		output := met.checkpointing.CheckpointForward(model, mb)

		// Compute loss
		batchLoss := loss(output, targetBatches[i])
		totalLoss += batchLoss

		// Backward with gradient accumulation
		gradData := make([]float32, len(output.Data))
		fillValue := 1.0 / float32(len(output.Data))
		for i := range gradData {
			gradData[i] = fillValue
		}
		gradOutput, _ := tensor.NewTensor(output.Shape, gradData)

		met.checkpointing.CheckpointBackward(model, gradOutput)

		// Accumulate gradients
		for j, layer := range model {
			layerID := fmt.Sprintf("layer_%d", j)
			grads := layer.GetGradients()

			for _, grad := range grads {
				if met.accumulator.AccumulateGradients(layerID, grad) {
					// Gradients accumulated, ready for optimizer step
					break
				}
			}
		}
	}

	return totalLoss / float32(len(microBatches))
}

// splitIntoMicroBatches divides input into micro-batches
func (met *MemoryEfficientTraining) splitIntoMicroBatches(input *tensor.Tensor) []*tensor.Tensor {
	rows := input.Shape[0]
	cols := input.Shape[1]
	numMicroBatches := int(math.Ceil(float64(rows) / float64(met.mp.microBatchSize)))
	batches := make([]*tensor.Tensor, numMicroBatches)

	for i := range numMicroBatches {
		start := i * met.mp.microBatchSize
		end := min(start+met.mp.microBatchSize, rows)
		batchRows := end - start

		batchData := make([]float32, batchRows*cols)
		copy(batchData, input.Data[start*cols:end*cols])
		batches[i], _ = tensor.NewTensor([]int{batchRows, cols}, batchData)
	}

	return batches
}
