package matrix

import (
	"github.com/tsawler/go-nngpu/tensor"
	"gonum.org/v1/gonum/mat"
)

// GPUDense wraps a tensor to implement gonum's mat.Matrix interface
type GPUDense struct {
	tensor *tensor.Tensor
}

// NewGPUDense creates a new GPU-backed matrix compatible with Gonum
func NewGPUDense(rows, cols int, data []float64) *GPUDense {
	// Convert float64 to float32 for GPU computation
	float32Data := make([]float32, len(data))
	for i, v := range data {
		float32Data[i] = float32(v)
	}

	t, err := tensor.NewTensor([]int{rows, cols}, float32Data)
	if err != nil {
		panic(err) // In production, handle this more gracefully
	}

	return &GPUDense{tensor: t}
}

// FromGonum converts a gonum matrix to GPUDense
func FromGonum(m mat.Matrix) *GPUDense {
	rows, cols := m.Dims()
	data := make([]float64, rows*cols)

	// Extract data from gonum matrix
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = m.At(i, j)
		}
	}

	return NewGPUDense(rows, cols, data)
}

// Implement mat.Matrix interface
func (g *GPUDense) Dims() (r, c int) {
	return g.tensor.Shape[0], g.tensor.Shape[1]
}

func (g *GPUDense) At(i, j int) float64 {
	// Ensure data is on CPU for reading
	if err := g.tensor.RetrieveCPU(); err != nil {
		panic(err)
	}
	rows, cols := g.Dims()
	if i < 0 || i >= rows || j < 0 || j >= cols {
		panic("matrix index out of range")
	}
	return float64(g.tensor.Data[i*cols+j])
}

func (g *GPUDense) T() mat.Matrix {
	// Implement transpose if needed
	panic("transpose not implemented yet")
}

// Implement mat.Mutable interface for setting values
func (g *GPUDense) Set(i, j int, v float64) {
	if err := g.tensor.RetrieveCPU(); err != nil {
		panic(err)
	}
	rows, cols := g.Dims()
	if i < 0 || i >= rows || j < 0 || j >= cols {
		panic("matrix index out of range")
	}
	g.tensor.Data[i*cols+j] = float32(v)
}

// GPU-accelerated matrix multiplication
func (g *GPUDense) Mul(a, b mat.Matrix) {
	// Convert inputs to GPU tensors if they aren't already
	var aTensor, bTensor *tensor.Tensor

	if gpuA, ok := a.(*GPUDense); ok {
		aTensor = gpuA.tensor
	} else {
		gpuA := FromGonum(a)
		aTensor = gpuA.tensor
	}

	if gpuB, ok := b.(*GPUDense); ok {
		bTensor = gpuB.tensor
	} else {
		gpuB := FromGonum(b)
		bTensor = gpuB.tensor
	}

	// Perform GPU multiplication
	result, err := MatMul(aTensor, bTensor)
	if err != nil {
		panic(err)
	}

	// Replace current tensor with result
	g.tensor.ReleaseGPU()
	g.tensor = result
}

// ToGonum converts back to a standard gonum Dense matrix
func (g *GPUDense) ToGonum() *mat.Dense {
	if err := g.tensor.RetrieveCPU(); err != nil {
		panic(err)
	}

	rows, cols := g.Dims()
	data := make([]float64, len(g.tensor.Data))
	for i, v := range g.tensor.Data {
		data[i] = float64(v)
	}

	return mat.NewDense(rows, cols, data)
}

// ReleaseGPU releases GPU resources
func (g *GPUDense) ReleaseGPU() {
	g.tensor.ReleaseGPU()
}

// GPUMatMul is a drop-in replacement for gonum's matrix multiplication
func GPUMatMul(a, b mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	gpuB := FromGonum(b)
	defer gpuB.ReleaseGPU()

	result, err := MatMul(gpuA.tensor, gpuB.tensor)
	if err != nil {
		panic(err)
	}
	defer result.ReleaseGPU()

	// Convert back to gonum format
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

// BatchGPUMatMul keeps matrices on GPU for multiple operations
func BatchGPUMatMul(operations []struct{ A, B mat.Matrix }) []*mat.Dense {
	results := make([]*mat.Dense, len(operations))

	for i, op := range operations {
		results[i] = GPUMatMul(op.A, op.B)
	}

	return results
}
