package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"

	"github.com/tsawler/go-nngpu/tensor"
)

// PrecisionType represents the data precision for mixed precision training
type PrecisionType int

const (
	PrecisionFloat32 PrecisionType = iota
	PrecisionFloatMP16
)

// MixedPrecisionConfig configures mixed precision training behavior
type MixedPrecisionConfig struct {
	Enabled             bool          // Enable mixed precision training
	LossScale           float32       // Initial loss scale factor
	LossScaleGrowthRate float32       // Factor to increase loss scale when no overflow
	LossScaleBackoffRate float32      // Factor to decrease loss scale on overflow
	GrowthInterval      int           // Number of steps between loss scale growth attempts
	MaxLossScale        float32       // Maximum allowed loss scale
	MinLossScale        float32       // Minimum allowed loss scale
	SkipOverflowSteps   bool          // Skip optimizer step on gradient overflow
}

// DefaultMixedPrecisionConfig returns default mixed precision configuration
func DefaultMixedPrecisionConfig() *MixedPrecisionConfig {
	return &MixedPrecisionConfig{
		Enabled:             true,
		LossScale:           65536.0,   // 2^16
		LossScaleGrowthRate: 2.0,
		LossScaleBackoffRate: 0.5,
		GrowthInterval:      2000,
		MaxLossScale:        1048576.0, // 2^20
		MinLossScale:        1.0,
		SkipOverflowSteps:   true,
	}
}

// MixedPrecisionTrainer manages mixed precision training state
type MixedPrecisionTrainer struct {
	config              *MixedPrecisionConfig
	currentLossScale    float32
	stepsSinceLastGrowth int
	overflowDetected    bool
	scaleBuffer         unsafe.Pointer // GPU buffer for loss scale
	overflowBuffer      unsafe.Pointer // GPU buffer for overflow detection
	devicePtr           unsafe.Pointer
}

// NewMixedPrecisionTrainer creates a new mixed precision trainer
func NewMixedPrecisionTrainer(config *MixedPrecisionConfig) (*MixedPrecisionTrainer, error) {
	if config == nil {
		config = DefaultMixedPrecisionConfig()
	}

	mp := &MixedPrecisionTrainer{
		config:           config,
		currentLossScale: config.LossScale,
	}

	// Initialize GPU buffers for loss scaling
	if config.Enabled {
		if err := mp.initializeGPUBuffers(); err != nil {
			return nil, fmt.Errorf("failed to initialize GPU buffers: %w", err)
		}
	}

	return mp, nil
}

// initializeGPUBuffers creates GPU buffers for loss scaling operations
func (mp *MixedPrecisionTrainer) initializeGPUBuffers() error {
	// Create loss scale buffer using existing tensor infrastructure
	scaleData := []float32{mp.currentLossScale}
	scaleTensor, err := tensor.NewTensor([]int{1}, scaleData)
	if err != nil {
		return fmt.Errorf("failed to create scale tensor: %w", err)
	}
	
	if err := scaleTensor.EnsureGPU(); err != nil {
		return fmt.Errorf("failed to move scale tensor to GPU: %w", err)
	}
	
	// Store tensor instead of raw GPU pointer for compatibility
	// TODO: In a real implementation, we'd store the tensor reference
	
	return nil
}

// FloatMP16 represents a 16-bit floating point number
type FloatMP16 uint16

// Float32ToFloatMP16 converts float32 to float16
func Float32ToFloatMP16(f float32) FloatMP16 {
	bits := math.Float32bits(f)
	
	// Extract sign, exponent, and mantissa
	sign := (bits >> 31) & 0x1
	exp := (bits >> 23) & 0xFF
	mantissa := bits & 0x7FFFFF
	
	// Handle special cases
	if exp == 0 {
		// Zero or subnormal
		return FloatMP16(sign << 15)
	} else if exp == 0xFF {
		// Infinity or NaN
		if mantissa == 0 {
			// Infinity
			return FloatMP16((sign << 15) | 0x7C00)
		} else {
			// NaN
			return FloatMP16((sign << 15) | 0x7C00 | (mantissa >> 13))
		}
	}
	
	// Adjust exponent for float16 bias (15 vs 127)
	expAdjusted := int32(exp) - 127 + 15
	
	if expAdjusted >= 31 {
		// Overflow to infinity
		return FloatMP16((sign << 15) | 0x7C00)
	} else if expAdjusted <= 0 {
		// Underflow to zero or subnormal
		if expAdjusted < -10 {
			// Too small, round to zero
			return FloatMP16(sign << 15)
		}
		// Subnormal number
		shiftAmount := 1 - expAdjusted
		if shiftAmount > 0 {
			mantissa = (mantissa | 0x800000) >> uint32(shiftAmount)
		}
		return FloatMP16((sign << 15) | (mantissa >> 13))
	}
	
	// Normal number
	return FloatMP16((sign << 15) | (uint32(expAdjusted) << 10) | (mantissa >> 13))
}

// FloatMP16ToFloat32 converts float16 to float32
func FloatMP16ToFloat32(h FloatMP16) float32 {
	bits := uint32(h)
	
	// Extract sign, exponent, and mantissa
	sign := (bits >> 15) & 0x1
	exp := (bits >> 10) & 0x1F
	mantissa := bits & 0x3FF
	
	var result uint32
	
	if exp == 0 {
		if mantissa == 0 {
			// Zero
			result = sign << 31
		} else {
			// Subnormal number
			// Normalize the mantissa
			for (mantissa & 0x400) == 0 {
				mantissa <<= 1
				exp--
			}
			mantissa &= 0x3FF
			exp = exp - 15 + 127 + 1
			result = (sign << 31) | (uint32(exp) << 23) | (mantissa << 13)
		}
	} else if exp == 31 {
		// Infinity or NaN
		result = (sign << 31) | 0x7F800000 | (mantissa << 13)
	} else {
		// Normal number
		exp = exp - 15 + 127
		result = (sign << 31) | (uint32(exp) << 23) | (mantissa << 13)
	}
	
	return math.Float32frombits(result)
}

// ConvertTensorToFloat16 converts a float32 tensor to float16 representation
func (mp *MixedPrecisionTrainer) ConvertTensorToFloat16(input *tensor.Tensor) (*tensor.Tensor, error) {
	if !mp.config.Enabled {
		return input, nil
	}

	// Create float16 data
	float16Data := make([]float32, len(input.Data))
	for i, val := range input.Data {
		f16 := Float32ToFloatMP16(val)
		float16Data[i] = FloatMP16ToFloat32(f16) // Store as float32 but with float16 precision
	}

	result, err := tensor.NewTensor(input.Shape, float16Data)
	if err != nil {
		return nil, fmt.Errorf("failed to create float16 tensor: %w", err)
	}

	return result, nil
}

// ScaleGradients applies loss scaling to gradients
func (mp *MixedPrecisionTrainer) ScaleGradients(gradients *tensor.Tensor) (*tensor.Tensor, error) {
	if !mp.config.Enabled {
		return gradients, nil
	}

	// Scale gradients by current loss scale
	scaledData := make([]float32, len(gradients.Data))
	for i, grad := range gradients.Data {
		scaledData[i] = grad * mp.currentLossScale
	}

	result, err := tensor.NewTensor(gradients.Shape, scaledData)
	if err != nil {
		return nil, fmt.Errorf("failed to create scaled gradients: %w", err)
	}

	return result, nil
}

// UnscaleGradients removes loss scaling from gradients
func (mp *MixedPrecisionTrainer) UnscaleGradients(scaledGradients *tensor.Tensor) (*tensor.Tensor, error) {
	if !mp.config.Enabled {
		return scaledGradients, nil
	}

	// Check for gradient overflow
	mp.overflowDetected = false
	invScale := 1.0 / mp.currentLossScale

	unscaledData := make([]float32, len(scaledGradients.Data))
	for i, grad := range scaledGradients.Data {
		unscaled := grad * invScale
		
		// Check for overflow (inf or NaN)
		if math.IsInf(float64(unscaled), 0) || math.IsNaN(float64(unscaled)) {
			mp.overflowDetected = true
		}
		
		unscaledData[i] = unscaled
	}

	result, err := tensor.NewTensor(scaledGradients.Shape, unscaledData)
	if err != nil {
		return nil, fmt.Errorf("failed to create unscaled gradients: %w", err)
	}

	return result, nil
}

// UpdateLossScale updates the loss scale based on overflow detection
func (mp *MixedPrecisionTrainer) UpdateLossScale() {
	if !mp.config.Enabled {
		return
	}

	if mp.overflowDetected {
		// Reduce loss scale on overflow
		mp.currentLossScale *= mp.config.LossScaleBackoffRate
		if mp.currentLossScale < mp.config.MinLossScale {
			mp.currentLossScale = mp.config.MinLossScale
		}
		mp.stepsSinceLastGrowth = 0
	} else {
		// Increase loss scale if no overflow for specified interval
		mp.stepsSinceLastGrowth++
		if mp.stepsSinceLastGrowth >= mp.config.GrowthInterval {
			mp.currentLossScale *= mp.config.LossScaleGrowthRate
			if mp.currentLossScale > mp.config.MaxLossScale {
				mp.currentLossScale = mp.config.MaxLossScale
			}
			mp.stepsSinceLastGrowth = 0
		}
	}

	// TODO: Update loss scale (in a real implementation, would update GPU buffer)
}

// ShouldSkipStep returns true if the optimizer step should be skipped due to overflow
func (mp *MixedPrecisionTrainer) ShouldSkipStep() bool {
	return mp.config.Enabled && mp.config.SkipOverflowSteps && mp.overflowDetected
}

// GetCurrentLossScale returns the current loss scale value
func (mp *MixedPrecisionTrainer) GetCurrentLossScale() float32 {
	return mp.currentLossScale
}

// GetOverflowStatus returns whether overflow was detected in the last gradient computation
func (mp *MixedPrecisionTrainer) GetOverflowStatus() bool {
	return mp.overflowDetected
}

// Cleanup releases GPU resources
func (mp *MixedPrecisionTrainer) Cleanup() {
	// TODO: In a real implementation, would clean up GPU buffers
	// For now, just reset state
	mp.scaleBuffer = nil
	mp.overflowBuffer = nil
	mp.devicePtr = nil
}

// ForwardFloat16 performs forward pass with automatic mixed precision
func (mp *MixedPrecisionTrainer) ForwardFloat16(input *tensor.Tensor, weights *tensor.Tensor, bias *tensor.Tensor) (*tensor.Tensor, error) {
	if !mp.config.Enabled {
		return MatMul(input, weights)
	}

	// Convert inputs to float16 precision for computation
	inputF16, err := mp.ConvertTensorToFloat16(input)
	if err != nil {
		return nil, fmt.Errorf("failed to convert input to float16: %w", err)
	}

	weightsF16, err := mp.ConvertTensorToFloat16(weights)
	if err != nil {
		return nil, fmt.Errorf("failed to convert weights to float16: %w", err)
	}

	// Perform matrix multiplication in float16
	result, err := MatMul(inputF16, weightsF16)
	if err != nil {
		return nil, fmt.Errorf("failed to perform float16 matrix multiplication: %w", err)
	}

	// Add bias if provided
	if bias != nil {
		biasF16, err := mp.ConvertTensorToFloat16(bias)
		if err != nil {
			return nil, fmt.Errorf("failed to convert bias to float16: %w", err)
		}

		// Add bias (broadcasting)
		if err := result.RetrieveCPU(); err != nil {
			return nil, fmt.Errorf("failed to move result to CPU for bias addition: %w", err)
		}
		if err := biasF16.RetrieveCPU(); err != nil {
			return nil, fmt.Errorf("failed to move bias to CPU for addition: %w", err)
		}

		for i := range result.Data {
			result.Data[i] += biasF16.Data[i%len(biasF16.Data)]
		}
	}

	return result, nil
}