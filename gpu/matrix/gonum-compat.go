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
	// Implement transpose using our GPU implementation
	transposed, err := Transpose(g.tensor)
	if err != nil {
		panic(err)
	}
	return &GPUDense{tensor: transposed}
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

// GPU-accelerated element-wise addition
func (g *GPUDense) Add(a, b mat.Matrix) {
	var aTensor, bTensor *tensor.Tensor

	if gpuA, ok := a.(*GPUDense); ok {
		aTensor = gpuA.tensor
	} else {
		gpuA := FromGonum(a)
		aTensor = gpuA.tensor
		defer gpuA.ReleaseGPU()
	}

	if gpuB, ok := b.(*GPUDense); ok {
		bTensor = gpuB.tensor
	} else {
		gpuB := FromGonum(b)
		bTensor = gpuB.tensor
		defer gpuB.ReleaseGPU()
	}

	result, err := Add(aTensor, bTensor)
	if err != nil {
		panic(err)
	}

	g.tensor.ReleaseGPU()
	g.tensor = result
}

// GPU-accelerated element-wise subtraction
func (g *GPUDense) Sub(a, b mat.Matrix) {
	var aTensor, bTensor *tensor.Tensor

	if gpuA, ok := a.(*GPUDense); ok {
		aTensor = gpuA.tensor
	} else {
		gpuA := FromGonum(a)
		aTensor = gpuA.tensor
		defer gpuA.ReleaseGPU()
	}

	if gpuB, ok := b.(*GPUDense); ok {
		bTensor = gpuB.tensor
	} else {
		gpuB := FromGonum(b)
		bTensor = gpuB.tensor
		defer gpuB.ReleaseGPU()
	}

	result, err := Sub(aTensor, bTensor)
	if err != nil {
		panic(err)
	}

	g.tensor.ReleaseGPU()
	g.tensor = result
}

// GPU-accelerated element-wise multiplication (Hadamard product)
func (g *GPUDense) MulElem(a, b mat.Matrix) {
	var aTensor, bTensor *tensor.Tensor

	if gpuA, ok := a.(*GPUDense); ok {
		aTensor = gpuA.tensor
	} else {
		gpuA := FromGonum(a)
		aTensor = gpuA.tensor
		defer gpuA.ReleaseGPU()
	}

	if gpuB, ok := b.(*GPUDense); ok {
		bTensor = gpuB.tensor
	} else {
		gpuB := FromGonum(b)
		bTensor = gpuB.tensor
		defer gpuB.ReleaseGPU()
	}

	result, err := Mul(aTensor, bTensor)
	if err != nil {
		panic(err)
	}

	g.tensor.ReleaseGPU()
	g.tensor = result
}

// GPU-accelerated element-wise division
func (g *GPUDense) DivElem(a, b mat.Matrix) {
	var aTensor, bTensor *tensor.Tensor

	if gpuA, ok := a.(*GPUDense); ok {
		aTensor = gpuA.tensor
	} else {
		gpuA := FromGonum(a)
		aTensor = gpuA.tensor
		defer gpuA.ReleaseGPU()
	}

	if gpuB, ok := b.(*GPUDense); ok {
		bTensor = gpuB.tensor
	} else {
		gpuB := FromGonum(b)
		bTensor = gpuB.tensor
		defer gpuB.ReleaseGPU()
	}

	result, err := Div(aTensor, bTensor)
	if err != nil {
		panic(err)
	}

	g.tensor.ReleaseGPU()
	g.tensor = result
}

// GPU-accelerated scalar addition
func (g *GPUDense) AddScalar(scalar float64) {
	result, err := ScalarAdd(g.tensor, float32(scalar))
	if err != nil {
		panic(err)
	}

	g.tensor.ReleaseGPU()
	g.tensor = result
}

// GPU-accelerated scalar multiplication
func (g *GPUDense) Scale(scalar float64) {
	result, err := ScalarMul(g.tensor, float32(scalar))
	if err != nil {
		panic(err)
	}

	g.tensor.ReleaseGPU()
	g.tensor = result
}

// Phase 3: Advanced matrix operations

// Inverse computes the matrix inverse using GPU acceleration
func (g *GPUDense) Inverse() *GPUDense {
	result, err := Inverse(g.tensor)
	if err != nil {
		panic(err)
	}
	return &GPUDense{tensor: result}
}

// Det computes the matrix determinant using GPU acceleration
func (g *GPUDense) Det() float64 {
	det, err := Determinant(g.tensor)
	if err != nil {
		panic(err)
	}
	return float64(det)
}

// LU performs LU decomposition and returns L, U matrices and pivot indices
func (g *GPUDense) LU() (*GPUDense, *GPUDense, []int) {
	lu, err := LU(g.tensor)
	if err != nil {
		panic(err)
	}

	return &GPUDense{tensor: lu.L}, &GPUDense{tensor: lu.U}, lu.PivotIndices
}

// Phase 4: Advanced Decompositions

// QR performs QR decomposition and returns Q, R matrices
func (g *GPUDense) QR() (*GPUDense, *GPUDense) {
	qr, err := QR(g.tensor)
	if err != nil {
		panic(err)
	}

	return &GPUDense{tensor: qr.Q}, &GPUDense{tensor: qr.R}
}

// Cholesky performs Cholesky decomposition and returns the lower triangular matrix L
func (g *GPUDense) Cholesky() *GPUDense {
	result, err := Cholesky(g.tensor)
	if err != nil {
		panic(err)
	}
	return &GPUDense{tensor: result}
}

// Eigen performs eigenvalue decomposition for symmetric matrices
func (g *GPUDense) Eigen() (*GPUDense, *GPUDense) {
	eigen, err := Eigen(g.tensor)
	if err != nil {
		panic(err)
	}

	return &GPUDense{tensor: eigen.Eigenvalues}, &GPUDense{tensor: eigen.Eigenvectors}
}

// SVD performs Singular Value Decomposition
func (g *GPUDense) SVD() (*GPUDense, *GPUDense, *GPUDense) {
	svd, err := SVD(g.tensor)
	if err != nil {
		panic(err)
	}

	return &GPUDense{tensor: svd.U}, &GPUDense{tensor: svd.S}, &GPUDense{tensor: svd.VT}
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

// GPUAdd is a drop-in replacement for gonum's matrix addition
func GPUAdd(a, b mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	gpuB := FromGonum(b)
	defer gpuB.ReleaseGPU()

	result, err := Add(gpuA.tensor, gpuB.tensor)
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

// GPUSub is a drop-in replacement for gonum's matrix subtraction
func GPUSub(a, b mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	gpuB := FromGonum(b)
	defer gpuB.ReleaseGPU()

	result, err := Sub(gpuA.tensor, gpuB.tensor)
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

// GPUMulElem is a drop-in replacement for gonum's element-wise multiplication
func GPUMulElem(a, b mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	gpuB := FromGonum(b)
	defer gpuB.ReleaseGPU()

	result, err := Mul(gpuA.tensor, gpuB.tensor)
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

// GPUDivElem is a drop-in replacement for gonum's element-wise division
func GPUDivElem(a, b mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	gpuB := FromGonum(b)
	defer gpuB.ReleaseGPU()

	result, err := Div(gpuA.tensor, gpuB.tensor)
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

// Phase 3: Advanced GPU operations with Gonum compatibility

// GPUInverse is a drop-in replacement for gonum's matrix inverse
func GPUInverse(a mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	result, err := Inverse(gpuA.tensor)
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

// GPUDeterminant is a drop-in replacement for gonum's matrix determinant
func GPUDeterminant(a mat.Matrix) float64 {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	det, err := Determinant(gpuA.tensor)
	if err != nil {
		panic(err)
	}

	return float64(det)
}

// GPULUDecomposition represents LU decomposition results in Gonum format
type GPULUDecomposition struct {
	L            *mat.Dense
	U            *mat.Dense
	PivotIndices []int
}

// GPULU is a drop-in replacement for gonum's LU decomposition
func GPULU(a mat.Matrix) *GPULUDecomposition {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	lu, err := LU(gpuA.tensor)
	if err != nil {
		panic(err)
	}
	defer lu.ReleaseGPU()

	// Convert L matrix to gonum format
	if err := lu.L.RetrieveCPU(); err != nil {
		panic(err)
	}
	lRows, lCols := lu.L.Shape[0], lu.L.Shape[1]
	lData := make([]float64, len(lu.L.Data))
	for i, v := range lu.L.Data {
		lData[i] = float64(v)
	}
	lMatrix := mat.NewDense(lRows, lCols, lData)

	// Convert U matrix to gonum format
	if err := lu.U.RetrieveCPU(); err != nil {
		panic(err)
	}
	uRows, uCols := lu.U.Shape[0], lu.U.Shape[1]
	uData := make([]float64, len(lu.U.Data))
	for i, v := range lu.U.Data {
		uData[i] = float64(v)
	}
	uMatrix := mat.NewDense(uRows, uCols, uData)

	return &GPULUDecomposition{
		L:            lMatrix,
		U:            uMatrix,
		PivotIndices: lu.PivotIndices,
	}
}

// Phase 4: Advanced Decompositions with Gonum compatibility

// GPUQRDecomposition represents QR decomposition results in Gonum format
type GPUQRDecomposition struct {
	Q *mat.Dense
	R *mat.Dense
}

// GPUQR is a drop-in replacement for gonum's QR decomposition
func GPUQR(a mat.Matrix) *GPUQRDecomposition {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	qr, err := QR(gpuA.tensor)
	if err != nil {
		panic(err)
	}
	defer qr.ReleaseGPU()

	// Convert Q matrix to gonum format
	if err := qr.Q.RetrieveCPU(); err != nil {
		panic(err)
	}
	qRows, qCols := qr.Q.Shape[0], qr.Q.Shape[1]
	qData := make([]float64, len(qr.Q.Data))
	for i, v := range qr.Q.Data {
		qData[i] = float64(v)
	}
	qMatrix := mat.NewDense(qRows, qCols, qData)

	// Convert R matrix to gonum format
	if err := qr.R.RetrieveCPU(); err != nil {
		panic(err)
	}
	rRows, rCols := qr.R.Shape[0], qr.R.Shape[1]
	rData := make([]float64, len(qr.R.Data))
	for i, v := range qr.R.Data {
		rData[i] = float64(v)
	}
	rMatrix := mat.NewDense(rRows, rCols, rData)

	return &GPUQRDecomposition{
		Q: qMatrix,
		R: rMatrix,
	}
}

// GPUCholesky is a drop-in replacement for gonum's Cholesky decomposition
func GPUCholesky(a mat.Matrix) *mat.Dense {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	result, err := Cholesky(gpuA.tensor)
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

// GPUEigenDecomposition represents eigenvalue decomposition results in Gonum format
type GPUEigenDecomposition struct {
	Eigenvalues  *mat.VecDense
	Eigenvectors *mat.Dense
}

// GPUEigen is a drop-in replacement for gonum's eigenvalue decomposition
func GPUEigen(a mat.Matrix) *GPUEigenDecomposition {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	eigen, err := Eigen(gpuA.tensor)
	if err != nil {
		panic(err)
	}
	defer eigen.ReleaseGPU()

	// Convert eigenvalues to gonum VecDense format
	if err := eigen.Eigenvalues.RetrieveCPU(); err != nil {
		panic(err)
	}
	eigenvaluesData := make([]float64, len(eigen.Eigenvalues.Data))
	for i, v := range eigen.Eigenvalues.Data {
		eigenvaluesData[i] = float64(v)
	}
	eigenvaluesVec := mat.NewVecDense(len(eigenvaluesData), eigenvaluesData)

	// Convert eigenvectors to gonum Dense format
	if err := eigen.Eigenvectors.RetrieveCPU(); err != nil {
		panic(err)
	}
	evRows, evCols := eigen.Eigenvectors.Shape[0], eigen.Eigenvectors.Shape[1]
	evData := make([]float64, len(eigen.Eigenvectors.Data))
	for i, v := range eigen.Eigenvectors.Data {
		evData[i] = float64(v)
	}
	eigenvectorsMatrix := mat.NewDense(evRows, evCols, evData)

	return &GPUEigenDecomposition{
		Eigenvalues:  eigenvaluesVec,
		Eigenvectors: eigenvectorsMatrix,
	}
}

// GPUSVDDecomposition represents SVD results in Gonum format
type GPUSVDDecomposition struct {
	U  *mat.Dense
	S  *mat.VecDense
	VT *mat.Dense
}

// GPUSVD is a drop-in replacement for gonum's SVD
func GPUSVD(a mat.Matrix) *GPUSVDDecomposition {
	gpuA := FromGonum(a)
	defer gpuA.ReleaseGPU()

	svd, err := SVD(gpuA.tensor)
	if err != nil {
		panic(err)
	}
	defer svd.ReleaseGPU()

	// Convert U matrix to gonum format
	if err := svd.U.RetrieveCPU(); err != nil {
		panic(err)
	}
	uRows, uCols := svd.U.Shape[0], svd.U.Shape[1]
	uData := make([]float64, len(svd.U.Data))
	for i, v := range svd.U.Data {
		uData[i] = float64(v)
	}
	uMatrix := mat.NewDense(uRows, uCols, uData)

	// Convert S vector to gonum VecDense format
	if err := svd.S.RetrieveCPU(); err != nil {
		panic(err)
	}
	sData := make([]float64, len(svd.S.Data))
	for i, v := range svd.S.Data {
		sData[i] = float64(v)
	}
	sVector := mat.NewVecDense(len(sData), sData)

	// Convert VT matrix to gonum format
	if err := svd.VT.RetrieveCPU(); err != nil {
		panic(err)
	}
	vtRows, vtCols := svd.VT.Shape[0], svd.VT.Shape[1]
	vtData := make([]float64, len(svd.VT.Data))
	for i, v := range svd.VT.Data {
		vtData[i] = float64(v)
	}
	vtMatrix := mat.NewDense(vtRows, vtCols, vtData)

	return &GPUSVDDecomposition{
		U:  uMatrix,
		S:  sVector,
		VT: vtMatrix,
	}
}

// Batch operations for efficiency

// BatchGPUMatMul keeps matrices on GPU for multiple operations
func BatchGPUMatMul(operations []struct{ A, B mat.Matrix }) []*mat.Dense {
	results := make([]*mat.Dense, len(operations))

	for i, op := range operations {
		results[i] = GPUMatMul(op.A, op.B)
	}

	return results
}

// BatchGPUAdd performs multiple element-wise additions efficiently
func BatchGPUAdd(operations []struct{ A, B mat.Matrix }) []*mat.Dense {
	results := make([]*mat.Dense, len(operations))

	for i, op := range operations {
		results[i] = GPUAdd(op.A, op.B)
	}

	return results
}

// BatchGPUInverse performs multiple matrix inversions efficiently
func BatchGPUInverse(matrices []mat.Matrix) []*mat.Dense {
	results := make([]*mat.Dense, len(matrices))

	for i, m := range matrices {
		results[i] = GPUInverse(m)
	}

	return results
}

// BatchGPUQR performs multiple QR decompositions efficiently
func BatchGPUQR(matrices []mat.Matrix) []*GPUQRDecomposition {
	results := make([]*GPUQRDecomposition, len(matrices))

	for i, m := range matrices {
		results[i] = GPUQR(m)
	}

	return results
}

// BatchGPUCholesky performs multiple Cholesky decompositions efficiently
func BatchGPUCholesky(matrices []mat.Matrix) []*mat.Dense {
	results := make([]*mat.Dense, len(matrices))

	for i, m := range matrices {
		results[i] = GPUCholesky(m)
	}

	return results
}

// BatchGPUEigen performs multiple eigenvalue decompositions efficiently
func BatchGPUEigen(matrices []mat.Matrix) []*GPUEigenDecomposition {
	results := make([]*GPUEigenDecomposition, len(matrices))

	for i, m := range matrices {
		results[i] = GPUEigen(m)
	}

	return results
}

// BatchGPUSVD performs multiple SVD decompositions efficiently
func BatchGPUSVD(matrices []mat.Matrix) []*GPUSVDDecomposition {
	results := make([]*GPUSVDDecomposition, len(matrices))

	for i, m := range matrices {
		results[i] = GPUSVD(m)
	}

	return results
}

// GetTensor returns the underlying tensor.Tensor
func (g *GPUDense) GetTensor() *tensor.Tensor {
	return g.tensor
}

func NewGPUDenseFromTensor(t *tensor.Tensor) *GPUDense {
	return &GPUDense{tensor: t}
}
