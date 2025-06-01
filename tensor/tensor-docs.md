# tensor
--
    import "."


## Usage

#### type SparseFormat

```go
type SparseFormat int
```

SparseFormat represents different sparse matrix storage formats

```go
const (
	COO SparseFormat = iota // Coordinate format (row, col, value triplets)
	CSR                     // Compressed Sparse Row
	CSC                     // Compressed Sparse Column
)
```

#### func (SparseFormat) String

```go
func (sf SparseFormat) String() string
```
String returns a string representation of the sparse tensor format

#### type SparseTensor

```go
type SparseTensor struct {
	Shape  []int        // Dimensions of the tensor
	Format SparseFormat // Storage format
	NNZ    int          // Number of non-zero elements

	// COO format data
	RowIndices []int32   // Row indices for non-zero elements
	ColIndices []int32   // Column indices for non-zero elements
	Values     []float32 // Non-zero values

	// CSR/CSC format data (computed on demand)
	RowPtr  []int32   // CSR: row pointers, CSC: not used
	ColPtr  []int32   // CSC: column pointers, CSR: not used
	Indices []int32   // CSR: column indices, CSC: row indices
	Data    []float32 // Non-zero values in compressed format

	IsOwner bool // Exported so other packages can access
}
```

SparseTensor represents a sparse multi-dimensional array of float32

#### func  NewSparseTensorCOO

```go
func NewSparseTensorCOO(shape []int, rowIndices, colIndices []int32, values []float32) (*SparseTensor, error)
```
NewSparseTensorCOO creates a new sparse tensor in COO format

#### func  NewSparseTensorFromDense

```go
func NewSparseTensorFromDense(dense *Tensor, threshold float32) (*SparseTensor, error)
```
NewSparseTensorFromDense creates a sparse tensor from a dense tensor

#### func (*SparseTensor) Clone

```go
func (st *SparseTensor) Clone() (*SparseTensor, error)
```
Clone creates a deep copy of the sparse tensor

#### func (*SparseTensor) ConvertToCSC

```go
func (st *SparseTensor) ConvertToCSC() error
```
ConvertToCSC converts the sparse tensor to CSC format

#### func (*SparseTensor) ConvertToCSR

```go
func (st *SparseTensor) ConvertToCSR() error
```
ConvertToCSR converts the sparse tensor to CSR format

#### func (*SparseTensor) DevicePtr

```go
func (st *SparseTensor) DevicePtr() unsafe.Pointer
```
DevicePtr returns the unsafe.Pointer to the Metal device

#### func (*SparseTensor) EnsureGPU

```go
func (st *SparseTensor) EnsureGPU() error
```
EnsureGPU ensures the sparse tensor's data is on the GPU

#### func (*SparseTensor) GPUPtr

```go
func (st *SparseTensor) GPUPtr() unsafe.Pointer
```
GPUPtr returns the unsafe.Pointer to the GPU buffer

#### func (*SparseTensor) GetDensity

```go
func (st *SparseTensor) GetDensity() float64
```
GetDensity returns the density (sparsity ratio) of the matrix

#### func (*SparseTensor) GetFormat

```go
func (st *SparseTensor) GetFormat() SparseFormat
```
GetFormat returns the current storage format

#### func (*SparseTensor) GetNNZ

```go
func (st *SparseTensor) GetNNZ() int
```
GetNNZ returns the number of non-zero elements

#### func (*SparseTensor) IsCompatibleWith

```go
func (st *SparseTensor) IsCompatibleWith(other *SparseTensor) error
```
IsCompatibleWith checks if two sparse tensors are compatible for operations

#### func (*SparseTensor) ReleaseGPU

```go
func (st *SparseTensor) ReleaseGPU()
```
ReleaseGPU releases the GPU-side buffer for the sparse tensor

#### func (*SparseTensor) RetrieveCPU

```go
func (st *SparseTensor) RetrieveCPU() error
```
RetrieveCPU ensures the sparse tensor's data is on the CPU

#### func (*SparseTensor) ToDense

```go
func (st *SparseTensor) ToDense() (*Tensor, error)
```
ToDense converts the sparse tensor to a dense tensor

#### type Tensor

```go
type Tensor struct {
	Shape []int     // Dimensions of the tensor (e.g., [rows, cols] for a matrix)
	Data  []float32 // CPU-side data
}
```

Tensor represents a multi-dimensional array of float32, potentially backed by a
Metal GPU buffer.

#### func  NewTensor

```go
func NewTensor(shape []int, data []float32) (*Tensor, error)
```
NewTensor creates a new CPU-backed Tensor.

#### func (*Tensor) DevicePtr

```go
func (t *Tensor) DevicePtr() unsafe.Pointer
```
DevicePtr returns the unsafe.Pointer to the Metal device. This is an internal
helper for CGO calls within the go-nngpu module.

#### func (*Tensor) EnsureGPU

```go
func (t *Tensor) EnsureGPU() error
```
EnsureGPU ensures the tensor's data is on the GPU. If it's already on GPU, it
does nothing. If on CPU, it transfers it.

#### func (*Tensor) GPUPtr

```go
func (t *Tensor) GPUPtr() unsafe.Pointer
```
GPUPtr returns the unsafe.Pointer to the GPU buffer. This is an internal helper
for CGO calls within the go-nngpu module.

#### func (*Tensor) ReleaseGPU

```go
func (t *Tensor) ReleaseGPU()
```
ReleaseGPU releases the GPU-side buffer. Call this when the GPU data is no
longer needed. It checks if the tensor is the owner of the GPU buffer.

#### func (*Tensor) RetrieveCPU

```go
func (t *Tensor) RetrieveCPU() error
```
RetrieveCPU ensures the tensor's data is on the CPU. If it's already on CPU, it
does nothing. If on GPU, it transfers it.
