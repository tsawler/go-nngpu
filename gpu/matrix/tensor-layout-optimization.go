package matrix

import (
	"fmt"
	"sync"

	"github.com/tsawler/go-nngpu/tensor"
)

// Phase 8C: Tensor Layout Optimization
// This file implements optimized tensor data layouts for maximum GPU cache efficiency

// TensorLayout represents different memory layouts for tensors
type TensorLayout int

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

// String returns the string representation of a layout
func (layout TensorLayout) String() string {
	switch layout {
	case LayoutRowMajor:
		return "RowMajor"
	case LayoutColMajor:
		return "ColMajor"
	case LayoutNHWC:
		return "NHWC"
	case LayoutNCHW:
		return "NCHW"
	case LayoutHWCN:
		return "HWCN"
	case LayoutTiled:
		return "Tiled"
	case LayoutBlockedRowMajor:
		return "BlockedRowMajor"
	case LayoutBlockedColMajor:
		return "BlockedColMajor"
	case LayoutPadded:
		return "Padded"
	default:
		return "Unknown"
	}
}

// TensorLayoutOptimizer optimizes tensor layouts for specific GPU operations
type TensorLayoutOptimizer struct {
	cache     map[string]*TensorLayoutInfo // Cache of optimized layouts
	mutex     sync.RWMutex
	tileSize  int // Size of tiles for tiled layouts
	blockSize int // Size of blocks for blocked layouts
}

// TensorLayoutInfo contains information about an optimized tensor layout
type TensorLayoutInfo struct {
	OriginalShape []int        // Original tensor shape
	OptimizedShape []int       // Optimized shape (may include padding)
	Layout        TensorLayout // Layout type
	Padding       []int        // Padding added to each dimension
	Stride        []int        // Stride for each dimension
	TileInfo      *TileInfo    // Tiling information if applicable
}

// TileInfo contains tiling information for tiled layouts
type TileInfo struct {
	TileSize   []int // Size of each tile
	NumTiles   []int // Number of tiles in each dimension
	TileStride []int // Stride between tiles
}

// NewTensorLayoutOptimizer creates a new tensor layout optimizer
func NewTensorLayoutOptimizer() *TensorLayoutOptimizer {
	return &TensorLayoutOptimizer{
		cache:     make(map[string]*TensorLayoutInfo),
		tileSize:  64,  // 64x64 tiles for good cache locality
		blockSize: 16,  // 16x16 blocks for vectorization
	}
}

// OptimizeLayout determines the best layout for a tensor given the operation
func (tlo *TensorLayoutOptimizer) OptimizeLayout(shape []int, operation string, dataType string) *TensorLayoutInfo {
	// Create cache key
	key := fmt.Sprintf("%v_%s_%s", shape, operation, dataType)
	
	tlo.mutex.RLock()
	if info, exists := tlo.cache[key]; exists {
		tlo.mutex.RUnlock()
		return info
	}
	tlo.mutex.RUnlock()

	// Determine optimal layout based on operation and tensor properties
	var info *TensorLayoutInfo
	
	switch operation {
	case "conv2d", "convolution":
		info = tlo.optimizeForConvolution(shape)
	case "matmul", "gemm":
		info = tlo.optimizeForMatrixMultiplication(shape)
	case "elementwise", "broadcast":
		info = tlo.optimizeForElementwise(shape)
	case "reduction", "reduce":
		info = tlo.optimizeForReduction(shape)
	case "transpose":
		info = tlo.optimizeForTranspose(shape)
	case "attention":
		info = tlo.optimizeForAttention(shape)
	default:
		info = tlo.optimizeGeneral(shape)
	}
	
	// Cache the result
	tlo.mutex.Lock()
	tlo.cache[key] = info
	tlo.mutex.Unlock()
	
	return info
}

// optimizeForConvolution optimizes layout for convolution operations
func (tlo *TensorLayoutOptimizer) optimizeForConvolution(shape []int) *TensorLayoutInfo {
	if len(shape) != 4 {
		return tlo.optimizeGeneral(shape) // Fall back for non-4D tensors
	}
	
	batch, height, width, channels := shape[0], shape[1], shape[2], shape[3]
	
	// Calculate memory overhead before optimization
	originalSize := batch * height * width * channels
	
	// For GPU efficiency, pad channels to multiple of 4 or 8 (not 16)
	// Use smaller alignment to reduce memory waste
	var paddedChannels int
	if channels <= 4 {
		paddedChannels = ((channels + 3) / 4) * 4  // Align to 4 for small channel counts
	} else {
		paddedChannels = ((channels + 7) / 8) * 8  // Align to 8 for larger channel counts
	}
	channelPadding := paddedChannels - channels
	
	// Don't add spatial padding here - let the convolution operation handle it
	// This avoids unnecessary memory overhead for operations that don't need it
	paddedHeight := height
	paddedWidth := width
	
	// Calculate optimized size and check if overhead is reasonable
	optimizedSize := batch * paddedHeight * paddedWidth * paddedChannels
	overheadRatio := float64(optimizedSize) / float64(originalSize)
	
	// If overhead is too high (>50%), don't optimize
	if overheadRatio > 1.5 {
		return &TensorLayoutInfo{
			OriginalShape:  shape,
			OptimizedShape: shape,
			Layout:         LayoutNHWC,
			Padding:        []int{0, 0, 0, 0},
			Stride:         []int{height * width * channels, width * channels, channels, 1},
		}
	}
	
	optimizedShape := []int{batch, paddedHeight, paddedWidth, paddedChannels}
	padding := []int{0, 0, 0, channelPadding}
	stride := []int{paddedHeight * paddedWidth * paddedChannels, paddedWidth * paddedChannels, paddedChannels, 1}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         LayoutNHWC,
		Padding:        padding,
		Stride:         stride,
	}
}

// optimizeForMatrixMultiplication optimizes layout for matrix operations
func (tlo *TensorLayoutOptimizer) optimizeForMatrixMultiplication(shape []int) *TensorLayoutInfo {
	if len(shape) != 2 {
		return tlo.optimizeGeneral(shape)
	}
	
	rows, cols := shape[0], shape[1]
	
	// Choose between tiled and blocked layout based on matrix size
	if rows >= 512 && cols >= 512 {
		return tlo.optimizeWithTiling(shape, LayoutTiled)
	} else {
		return tlo.optimizeWithBlocking(shape, LayoutBlockedRowMajor)
	}
}

// optimizeForElementwise optimizes layout for elementwise operations
func (tlo *TensorLayoutOptimizer) optimizeForElementwise(shape []int) *TensorLayoutInfo {
	// For elementwise operations, ensure the last dimension is padded for vectorization
	totalSize := 1
	for _, dim := range shape {
		totalSize *= dim
	}
	
	// Pad to multiple of 16 for SIMD operations
	paddedSize := ((totalSize + 15) / 16) * 16
	elementPadding := paddedSize - totalSize
	
	if elementPadding == 0 {
		// No padding needed
		stride := make([]int, len(shape))
		stride[len(stride)-1] = 1
		for i := len(stride) - 2; i >= 0; i-- {
			stride[i] = stride[i+1] * shape[i+1]
		}
		
		return &TensorLayoutInfo{
			OriginalShape:  shape,
			OptimizedShape: shape,
			Layout:         LayoutRowMajor,
			Padding:        make([]int, len(shape)),
			Stride:         stride,
		}
	}
	
	// Add padding to the last dimension
	optimizedShape := make([]int, len(shape))
	copy(optimizedShape, shape)
	optimizedShape[len(optimizedShape)-1] += elementPadding
	
	padding := make([]int, len(shape))
	padding[len(padding)-1] = elementPadding
	
	stride := make([]int, len(shape))
	stride[len(stride)-1] = 1
	for i := len(stride) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * optimizedShape[i+1]
	}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         LayoutPadded,
		Padding:        padding,
		Stride:         stride,
	}
}

// optimizeForReduction optimizes layout for reduction operations
func (tlo *TensorLayoutOptimizer) optimizeForReduction(shape []int) *TensorLayoutInfo {
	// For reductions, organize data for efficient parallel access
	// Prefer layouts that allow coalesced memory access during reduction
	
	if len(shape) == 1 {
		return tlo.optimizeForElementwise(shape)
	}
	
	// For multi-dimensional reductions, ensure the reduction dimension is continuous
	// and well-aligned for parallel reduction
	lastDim := shape[len(shape)-1]
	paddedLastDim := ((lastDim + 31) / 32) * 32 // Pad to multiple of 32 for efficient reduction
	padding := paddedLastDim - lastDim
	
	if padding == 0 {
		return tlo.optimizeGeneral(shape)
	}
	
	optimizedShape := make([]int, len(shape))
	copy(optimizedShape, shape)
	optimizedShape[len(optimizedShape)-1] = paddedLastDim
	
	paddingArray := make([]int, len(shape))
	paddingArray[len(paddingArray)-1] = padding
	
	stride := make([]int, len(shape))
	stride[len(stride)-1] = 1
	for i := len(stride) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * optimizedShape[i+1]
	}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         LayoutPadded,
		Padding:        paddingArray,
		Stride:         stride,
	}
}

// optimizeForTranspose optimizes layout for transpose operations
func (tlo *TensorLayoutOptimizer) optimizeForTranspose(shape []int) *TensorLayoutInfo {
	if len(shape) != 2 {
		return tlo.optimizeGeneral(shape) // Only optimize 2D transposes for now
	}
	
	// For transpose operations, use blocked layout to improve cache locality
	return tlo.optimizeWithBlocking(shape, LayoutBlockedRowMajor)
}

// optimizeForAttention optimizes layout for attention operations
func (tlo *TensorLayoutOptimizer) optimizeForAttention(shape []int) *TensorLayoutInfo {
	if len(shape) != 3 {
		return tlo.optimizeGeneral(shape) // Expect [batch, seq_len, model_dim]
	}
	
	batch, seqLen, modelDim := shape[0], shape[1], shape[2]
	
	// Pad model dimension to multiple of head size (typically 64 or 128)
	headSize := 64
	paddedModelDim := ((modelDim + headSize - 1) / headSize) * headSize
	modelPadding := paddedModelDim - modelDim
	
	// Pad sequence length to multiple of 32 for efficient attention computation
	paddedSeqLen := ((seqLen + 31) / 32) * 32
	seqPadding := paddedSeqLen - seqLen
	
	optimizedShape := []int{batch, paddedSeqLen, paddedModelDim}
	padding := []int{0, seqPadding, modelPadding}
	stride := []int{paddedSeqLen * paddedModelDim, paddedModelDim, 1}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         LayoutPadded,
		Padding:        padding,
		Stride:         stride,
	}
}

// optimizeWithTiling creates a tiled layout for large tensors
func (tlo *TensorLayoutOptimizer) optimizeWithTiling(shape []int, layout TensorLayout) *TensorLayoutInfo {
	if len(shape) != 2 {
		return tlo.optimizeGeneral(shape)
	}
	
	rows, cols := shape[0], shape[1]
	tileRows := tlo.tileSize
	tileColumns := tlo.tileSize
	
	// Calculate number of tiles
	numTileRows := (rows + tileRows - 1) / tileRows
	numTileColumns := (cols + tileColumns - 1) / tileColumns
	
	// Calculate padding needed
	paddedRows := numTileRows * tileRows
	paddedColumns := numTileColumns * tileColumns
	rowPadding := paddedRows - rows
	colPadding := paddedColumns - cols
	
	optimizedShape := []int{paddedRows, paddedColumns}
	padding := []int{rowPadding, colPadding}
	
	// For tiled layout, stride is more complex
	stride := []int{paddedColumns, 1}
	
	tileInfo := &TileInfo{
		TileSize:   []int{tileRows, tileColumns},
		NumTiles:   []int{numTileRows, numTileColumns},
		TileStride: []int{tileRows * paddedColumns, tileColumns},
	}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         layout,
		Padding:        padding,
		Stride:         stride,
		TileInfo:       tileInfo,
	}
}

// optimizeWithBlocking creates a blocked layout for cache efficiency
func (tlo *TensorLayoutOptimizer) optimizeWithBlocking(shape []int, layout TensorLayout) *TensorLayoutInfo {
	if len(shape) != 2 {
		return tlo.optimizeGeneral(shape)
	}
	
	rows, cols := shape[0], shape[1]
	blockRows := tlo.blockSize
	blockColumns := tlo.blockSize
	
	// Calculate padding for block alignment
	paddedRows := ((rows + blockRows - 1) / blockRows) * blockRows
	paddedColumns := ((cols + blockColumns - 1) / blockColumns) * blockColumns
	rowPadding := paddedRows - rows
	colPadding := paddedColumns - cols
	
	optimizedShape := []int{paddedRows, paddedColumns}
	padding := []int{rowPadding, colPadding}
	stride := []int{paddedColumns, 1}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         layout,
		Padding:        padding,
		Stride:         stride,
	}
}

// optimizeGeneral provides general optimization for any tensor shape
func (tlo *TensorLayoutOptimizer) optimizeGeneral(shape []int) *TensorLayoutInfo {
	// General optimization: ensure the tensor size is aligned for SIMD operations
	totalSize := 1
	for _, dim := range shape {
		totalSize *= dim
	}
	
	// Align to 16 elements (64 bytes for float32)
	alignedSize := ((totalSize + 15) / 16) * 16
	padding := alignedSize - totalSize
	
	if padding == 0 {
		// No optimization needed
		stride := make([]int, len(shape))
		if len(stride) > 0 {
			stride[len(stride)-1] = 1
			for i := len(stride) - 2; i >= 0; i-- {
				stride[i] = stride[i+1] * shape[i+1]
			}
		}
		
		return &TensorLayoutInfo{
			OriginalShape:  shape,
			OptimizedShape: shape,
			Layout:         LayoutRowMajor,
			Padding:        make([]int, len(shape)),
			Stride:         stride,
		}
	}
	
	// Add padding to the last dimension
	optimizedShape := make([]int, len(shape))
	copy(optimizedShape, shape)
	if len(optimizedShape) > 0 {
		optimizedShape[len(optimizedShape)-1] += padding
	}
	
	paddingArray := make([]int, len(shape))
	if len(paddingArray) > 0 {
		paddingArray[len(paddingArray)-1] = padding
	}
	
	stride := make([]int, len(shape))
	if len(stride) > 0 {
		stride[len(stride)-1] = 1
		for i := len(stride) - 2; i >= 0; i-- {
			stride[i] = stride[i+1] * optimizedShape[i+1]
		}
	}
	
	return &TensorLayoutInfo{
		OriginalShape:  shape,
		OptimizedShape: optimizedShape,
		Layout:         LayoutPadded,
		Padding:        paddingArray,
		Stride:         stride,
	}
}

// ApplyLayoutOptimization applies the layout optimization to a tensor
func (tlo *TensorLayoutOptimizer) ApplyLayoutOptimization(t *tensor.Tensor, operation string) (*tensor.Tensor, *TensorLayoutInfo, error) {
	layoutInfo := tlo.OptimizeLayout(t.Shape, operation, "float32")
	
	// If no optimization is needed, return original tensor
	if layoutInfo.Layout == LayoutRowMajor && sumIntSlice(layoutInfo.Padding) == 0 {
		return t, layoutInfo, nil
	}
	
	// Create optimized tensor data
	optimizedData := make([]float32, calculateTotalSize(layoutInfo.OptimizedShape))
	
	// Copy and transform data according to the layout
	err := tlo.transformTensorData(t.Data, optimizedData, layoutInfo)
	if err != nil {
		return t, layoutInfo, err // Fall back to original tensor
	}
	
	// Create new tensor with optimized layout
	optimizedTensor, err := tensor.NewTensor(layoutInfo.OptimizedShape, optimizedData)
	if err != nil {
		return t, layoutInfo, err // Fall back to original tensor
	}
	
	return optimizedTensor, layoutInfo, nil
}

// transformTensorData transforms tensor data according to the layout information
func (tlo *TensorLayoutOptimizer) transformTensorData(src, dst []float32, layoutInfo *TensorLayoutInfo) error {
	switch layoutInfo.Layout {
	case LayoutRowMajor, LayoutPadded:
		return tlo.transformWithPadding(src, dst, layoutInfo)
	case LayoutTiled:
		return tlo.transformToTiled(src, dst, layoutInfo)
	case LayoutBlockedRowMajor:
		return tlo.transformToBlocked(src, dst, layoutInfo)
	case LayoutNHWC:
		return tlo.transformToNHWC(src, dst, layoutInfo)
	default:
		return tlo.transformWithPadding(src, dst, layoutInfo) // Default to padding
	}
}

// transformWithPadding applies padding transformation
func (tlo *TensorLayoutOptimizer) transformWithPadding(src, dst []float32, layoutInfo *TensorLayoutInfo) error {
	// For simple padding, we can use efficient copy operations
	if len(layoutInfo.OriginalShape) == 1 {
		copy(dst, src)
		return nil
	}
	
	// Multi-dimensional padding
	return tlo.copyWithPadding(src, dst, layoutInfo.OriginalShape, layoutInfo.OptimizedShape, layoutInfo.Padding)
}

// copyWithPadding copies data with padding in multiple dimensions
func (tlo *TensorLayoutOptimizer) copyWithPadding(src, dst []float32, originalShape, optimizedShape, padding []int) error {
	// This is a simplified implementation - in practice, you'd want optimized routines for each case
	if len(originalShape) == 2 {
		rows, cols := originalShape[0], originalShape[1]
		optCols := optimizedShape[1]
		
		for r := 0; r < rows; r++ {
			srcOffset := r * cols
			dstOffset := r * optCols
			copy(dst[dstOffset:dstOffset+cols], src[srcOffset:srcOffset+cols])
			// Padding columns are already zero-initialized
		}
	} else if len(originalShape) == 4 {
		// 4D tensor (common for convolution)
		batch, height, width, channels := originalShape[0], originalShape[1], originalShape[2], originalShape[3]
		_, optHeight, optWidth, optChannels := optimizedShape[0], optimizedShape[1], optimizedShape[2], optimizedShape[3]
		
		for b := 0; b < batch; b++ {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					srcOffset := ((b*height+h)*width+w)*channels
					dstOffset := ((b*optHeight+h)*optWidth+w)*optChannels
					copy(dst[dstOffset:dstOffset+channels], src[srcOffset:srcOffset+channels])
				}
			}
		}
	}
	
	return nil
}

// transformToTiled transforms data to tiled layout
func (tlo *TensorLayoutOptimizer) transformToTiled(src, dst []float32, layoutInfo *TensorLayoutInfo) error {
	if layoutInfo.TileInfo == nil || len(layoutInfo.OriginalShape) != 2 {
		return fmt.Errorf("invalid tiling information")
	}
	
	rows, cols := layoutInfo.OriginalShape[0], layoutInfo.OriginalShape[1]
	tileRows, tileCols := layoutInfo.TileInfo.TileSize[0], layoutInfo.TileInfo.TileSize[1]
	optCols := layoutInfo.OptimizedShape[1]
	
	// Copy data in tile order
	for tileR := 0; tileR < rows; tileR += tileRows {
		for tileC := 0; tileC < cols; tileC += tileCols {
			// Copy this tile
			maxR := min(tileR+tileRows, rows)
			maxC := min(tileC+tileCols, cols)
			
			for r := tileR; r < maxR; r++ {
				for c := tileC; c < maxC; c++ {
					srcIdx := r*cols + c
					dstIdx := r*optCols + c
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
	
	return nil
}

// transformToBlocked transforms data to blocked layout
func (tlo *TensorLayoutOptimizer) transformToBlocked(src, dst []float32, layoutInfo *TensorLayoutInfo) error {
	// Similar to tiled, but with different block organization
	return tlo.transformToTiled(src, dst, layoutInfo) // Simplified
}

// transformToNHWC transforms data to NHWC layout (if not already)
func (tlo *TensorLayoutOptimizer) transformToNHWC(src, dst []float32, layoutInfo *TensorLayoutInfo) error {
	// For now, assume data is already in NHWC format and just apply padding
	return tlo.transformWithPadding(src, dst, layoutInfo)
}

// Utility functions
func sumIntSlice(slice []int) int {
	sum := 0
	for _, v := range slice {
		sum += v
	}
	return sum
}

func calculateTotalSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

// min function removed to avoid conflict - using the one from memory-coalescing-optimizer.go

// Global layout optimizer
var globalLayoutOptimizer *TensorLayoutOptimizer
var layoutOptimizerOnce sync.Once

// GetGlobalLayoutOptimizer returns the global tensor layout optimizer
func GetGlobalLayoutOptimizer() *TensorLayoutOptimizer {
	layoutOptimizerOnce.Do(func() {
		globalLayoutOptimizer = NewTensorLayoutOptimizer()
	})
	return globalLayoutOptimizer
}

// OptimizeTensorForOperation optimizes a tensor layout for a specific operation
func OptimizeTensorForOperation(tensor *tensor.Tensor, operation string) (*tensor.Tensor, *TensorLayoutInfo, error) {
	optimizer := GetGlobalLayoutOptimizer()
	return optimizer.ApplyLayoutOptimization(tensor, operation)
}