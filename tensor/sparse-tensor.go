package tensor

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework Accelerate
#cgo CFLAGS: -x objective-c -ObjC -fobjc-arc
#include <stdlib.h>
#include "../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"sort"
	"unsafe"
)

// SparseFormat represents different sparse matrix storage formats
type SparseFormat int

const (
	COO SparseFormat = iota // Coordinate format (row, col, value triplets)
	CSR                     // Compressed Sparse Row
	CSC                     // Compressed Sparse Column
)

// SparseTensor represents a sparse multi-dimensional array of float32
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

	// GPU data
	gpuPtr    unsafe.Pointer // Pointer to GPU buffer
	isOnGPU   bool
	IsOwner   bool // Exported so other packages can access
	devicePtr unsafe.Pointer

	// Metadata
	density float64 // Sparsity ratio (nnz / total_elements)
}

// NewSparseTensorCOO creates a new sparse tensor in COO format
func NewSparseTensorCOO(shape []int, rowIndices, colIndices []int32, values []float32) (*SparseTensor, error) {
	if len(rowIndices) != len(colIndices) || len(rowIndices) != len(values) {
		return nil, fmt.Errorf("length mismatch: rowIndices=%d, colIndices=%d, values=%d",
			len(rowIndices), len(colIndices), len(values))
	}

	if len(shape) != 2 {
		return nil, fmt.Errorf("sparse tensors currently support only 2D matrices, got %d dimensions", len(shape))
	}

	// Validate indices
	rows, cols := shape[0], shape[1]
	for i, row := range rowIndices {
		if row < 0 || int(row) >= rows {
			return nil, fmt.Errorf("row index %d out of bounds [0, %d)", row, rows)
		}
		if colIndices[i] < 0 || int(colIndices[i]) >= cols {
			return nil, fmt.Errorf("column index %d out of bounds [0, %d)", colIndices[i], cols)
		}
	}

	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}

	density := float64(len(values)) / float64(totalElements)

	st := &SparseTensor{
		Shape:      shape,
		Format:     COO,
		NNZ:        len(values),
		RowIndices: make([]int32, len(rowIndices)),
		ColIndices: make([]int32, len(colIndices)),
		Values:     make([]float32, len(values)),
		isOnGPU:    false,
		IsOwner:    true,
		density:    density,
	}

	copy(st.RowIndices, rowIndices)
	copy(st.ColIndices, colIndices)
	copy(st.Values, values)

	return st, nil
}

// NewSparseTensorFromDense creates a sparse tensor from a dense tensor
func NewSparseTensorFromDense(dense *Tensor, threshold float32) (*SparseTensor, error) {
	if len(dense.Shape) != 2 {
		return nil, fmt.Errorf("sparse tensors currently support only 2D matrices")
	}

	rows, cols := dense.Shape[0], dense.Shape[1]
	var rowIndices, colIndices []int32
	var values []float32

	// Extract non-zero elements
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := dense.Data[i*cols+j]
			if val > threshold || val < -threshold {
				rowIndices = append(rowIndices, int32(i))
				colIndices = append(colIndices, int32(j))
				values = append(values, val)
			}
		}
	}

	return NewSparseTensorCOO(dense.Shape, rowIndices, colIndices, values)
}

// ConvertToCSR converts the sparse tensor to CSR format
func (st *SparseTensor) ConvertToCSR() error {
	if st.Format == CSR {
		return nil // Already in CSR format
	}

	if st.Format != COO {
		return fmt.Errorf("can only convert from COO to CSR format")
	}

	rows := st.Shape[0]

	// Sort by row, then by column
	type triplet struct {
		row, col int32
		val      float32
	}

	triplets := make([]triplet, st.NNZ)
	for i := 0; i < st.NNZ; i++ {
		triplets[i] = triplet{st.RowIndices[i], st.ColIndices[i], st.Values[i]}
	}

	sort.Slice(triplets, func(i, j int) bool {
		if triplets[i].row != triplets[j].row {
			return triplets[i].row < triplets[j].row
		}
		return triplets[i].col < triplets[j].col
	})

	// Build CSR format
	st.RowPtr = make([]int32, rows+1)
	st.Indices = make([]int32, st.NNZ)
	st.Data = make([]float32, st.NNZ)

	rowPtr := 0
	for i := 0; i < st.NNZ; i++ {
		// Fill row pointer array
		for rowPtr <= int(triplets[i].row) {
			st.RowPtr[rowPtr] = int32(i)
			rowPtr++
		}

		st.Indices[i] = triplets[i].col
		st.Data[i] = triplets[i].val
	}

	// Fill remaining row pointers
	for rowPtr <= rows {
		st.RowPtr[rowPtr] = int32(st.NNZ)
		rowPtr++
	}

	st.Format = CSR
	return nil
}

// ConvertToCSC converts the sparse tensor to CSC format
func (st *SparseTensor) ConvertToCSC() error {
	if st.Format == CSC {
		return nil // Already in CSC format
	}

	if st.Format != COO {
		return fmt.Errorf("can only convert from COO to CSC format")
	}

	cols := st.Shape[1]

	// Sort by column, then by row
	type triplet struct {
		row, col int32
		val      float32
	}

	triplets := make([]triplet, st.NNZ)
	for i := 0; i < st.NNZ; i++ {
		triplets[i] = triplet{st.RowIndices[i], st.ColIndices[i], st.Values[i]}
	}

	sort.Slice(triplets, func(i, j int) bool {
		if triplets[i].col != triplets[j].col {
			return triplets[i].col < triplets[j].col
		}
		return triplets[i].row < triplets[j].row
	})

	// Build CSC format
	st.ColPtr = make([]int32, cols+1)
	st.Indices = make([]int32, st.NNZ)
	st.Data = make([]float32, st.NNZ)

	colPtr := 0
	for i := 0; i < st.NNZ; i++ {
		// Fill column pointer array
		for colPtr <= int(triplets[i].col) {
			st.ColPtr[colPtr] = int32(i)
			colPtr++
		}

		st.Indices[i] = triplets[i].row
		st.Data[i] = triplets[i].val
	}

	// Fill remaining column pointers
	for colPtr <= cols {
		st.ColPtr[colPtr] = int32(st.NNZ)
		colPtr++
	}

	st.Format = CSC
	return nil
}

// ToDense converts the sparse tensor to a dense tensor
func (st *SparseTensor) ToDense() (*Tensor, error) {
	rows, cols := st.Shape[0], st.Shape[1]
	denseData := make([]float32, rows*cols)

	switch st.Format {
	case COO:
		for i := 0; i < st.NNZ; i++ {
			row, col := st.RowIndices[i], st.ColIndices[i]
			denseData[row*int32(cols)+col] = st.Values[i]
		}
	case CSR:
		for row := 0; row < rows; row++ {
			start, end := st.RowPtr[row], st.RowPtr[row+1]
			for i := start; i < end; i++ {
				col := st.Indices[i]
				denseData[row*cols+int(col)] = st.Data[i]
			}
		}
	case CSC:
		for col := 0; col < cols; col++ {
			start, end := st.ColPtr[col], st.ColPtr[col+1]
			for i := start; i < end; i++ {
				row := st.Indices[i]
				denseData[int(row)*cols+col] = st.Data[i]
			}
		}
	default:
		return nil, fmt.Errorf("unsupported sparse format: %d", st.Format)
	}

	return NewTensor(st.Shape, denseData)
}

// EnsureGPU ensures the sparse tensor's data is on the GPU
func (st *SparseTensor) EnsureGPU() error {
	if st.isOnGPU {
		return nil
	}

	// Convert to appropriate format for GPU operations
	if st.Format == COO {
		if err := st.ConvertToCSR(); err != nil {
			return fmt.Errorf("failed to convert to CSR format: %w", err)
		}
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Create GPU buffer for sparse data structure
	var bufferSize int
	var dataPtr *C.float

	switch st.Format {
	case CSR:
		// For CSR, we need to upload rowPtr, indices, and data
		bufferSize = len(st.Data) * int(unsafe.Sizeof(float32(0)))
		if len(st.Data) > 0 {
			dataPtr = (*C.float)(unsafe.Pointer(&st.Data[0]))
		}
	case CSC:
		// For CSC, we need to upload colPtr, indices, and data
		bufferSize = len(st.Data) * int(unsafe.Sizeof(float32(0)))
		if len(st.Data) > 0 {
			dataPtr = (*C.float)(unsafe.Pointer(&st.Data[0]))
		}
	default:
		return fmt.Errorf("GPU operations require CSR or CSC format")
	}

	if bufferSize == 0 {
		// Empty sparse matrix
		st.isOnGPU = true
		return nil
	}

	var cGPUPtr C.GPUPtr
	var cDevice C.DevicePtr
	var cErr C.CError

	retCode := C.create_gpu_buffer(dataPtr, C.long(bufferSize), &cGPUPtr, &cDevice, &cErr)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("failed to create sparse GPU buffer (code %d): %s", retCode, errMsg)
	}

	st.gpuPtr = unsafe.Pointer(cGPUPtr)
	st.devicePtr = unsafe.Pointer(cDevice)
	st.isOnGPU = true

	return nil
}

// RetrieveCPU ensures the sparse tensor's data is on the CPU
func (st *SparseTensor) RetrieveCPU() error {
	if !st.isOnGPU {
		return nil
	}

	if st.gpuPtr == nil {
		return fmt.Errorf("sparse tensor is marked as on GPU but has no GPU pointer")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var dataPtr *C.float
	var bufferSize int

	switch st.Format {
	case CSR, CSC:
		if len(st.Data) > 0 {
			dataPtr = (*C.float)(unsafe.Pointer(&st.Data[0]))
			bufferSize = len(st.Data) * int(unsafe.Sizeof(float32(0)))
		}
	default:
		return fmt.Errorf("unsupported format for GPU retrieval: %d", st.Format)
	}

	if bufferSize == 0 {
		st.isOnGPU = false
		return nil
	}

	var cErr C.CError
	retCode := C.retrieve_gpu_buffer_data(C.GPUPtr(st.gpuPtr), dataPtr, C.long(bufferSize), &cErr)

	if retCode != 0 {
		var errMsg string
		if cErr.message != nil {
			errMsg = C.GoString(cErr.message)
			C.free_c_error_message(cErr.message)
		}
		return fmt.Errorf("failed to retrieve sparse GPU buffer data (code %d): %s", retCode, errMsg)
	}

	st.isOnGPU = false
	return nil
}

// ReleaseGPU releases the GPU-side buffer for the sparse tensor
func (st *SparseTensor) ReleaseGPU() {
	if st.isOnGPU && st.gpuPtr != nil && st.IsOwner {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		C.release_gpu_buffer(C.GPUPtr(st.gpuPtr))
		st.gpuPtr = nil
		st.isOnGPU = false
	}
}

// GPUPtr returns the unsafe.Pointer to the GPU buffer
func (st *SparseTensor) GPUPtr() unsafe.Pointer {
	return st.gpuPtr
}

// DevicePtr returns the unsafe.Pointer to the Metal device
func (st *SparseTensor) DevicePtr() unsafe.Pointer {
	return st.devicePtr
}

// GetDensity returns the density (sparsity ratio) of the matrix
func (st *SparseTensor) GetDensity() float64 {
	return st.density
}

// GetNNZ returns the number of non-zero elements
func (st *SparseTensor) GetNNZ() int {
	return st.NNZ
}

// GetFormat returns the current storage format
func (st *SparseTensor) GetFormat() SparseFormat {
	return st.Format
}

// String returns a string representation of the sparse tensor format
func (sf SparseFormat) String() string {
	switch sf {
	case COO:
		return "COO"
	case CSR:
		return "CSR"
	case CSC:
		return "CSC"
	default:
		return "Unknown"
	}
}

// IsCompatibleWith checks if two sparse tensors are compatible for operations
func (st *SparseTensor) IsCompatibleWith(other *SparseTensor) error {
	if len(st.Shape) != len(other.Shape) {
		return fmt.Errorf("dimension mismatch: %d vs %d", len(st.Shape), len(other.Shape))
	}

	for i, dim := range st.Shape {
		if dim != other.Shape[i] {
			return fmt.Errorf("shape mismatch at dimension %d: %d vs %d", i, dim, other.Shape[i])
		}
	}

	return nil
}

// Clone creates a deep copy of the sparse tensor
func (st *SparseTensor) Clone() (*SparseTensor, error) {
	clone := &SparseTensor{
		Shape:   make([]int, len(st.Shape)),
		Format:  st.Format,
		NNZ:     st.NNZ,
		density: st.density,
		IsOwner: true,
	}

	copy(clone.Shape, st.Shape)

	// Copy format-specific data
	switch st.Format {
	case COO:
		clone.RowIndices = make([]int32, len(st.RowIndices))
		clone.ColIndices = make([]int32, len(st.ColIndices))
		clone.Values = make([]float32, len(st.Values))
		copy(clone.RowIndices, st.RowIndices)
		copy(clone.ColIndices, st.ColIndices)
		copy(clone.Values, st.Values)
	case CSR:
		clone.RowPtr = make([]int32, len(st.RowPtr))
		clone.Indices = make([]int32, len(st.Indices))
		clone.Data = make([]float32, len(st.Data))
		copy(clone.RowPtr, st.RowPtr)
		copy(clone.Indices, st.Indices)
		copy(clone.Data, st.Data)
	case CSC:
		clone.ColPtr = make([]int32, len(st.ColPtr))
		clone.Indices = make([]int32, len(st.Indices))
		clone.Data = make([]float32, len(st.Data))
		copy(clone.ColPtr, st.ColPtr)
		copy(clone.Indices, st.Indices)
		copy(clone.Data, st.Data)
	}

	return clone, nil
}
