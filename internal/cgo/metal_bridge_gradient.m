// metal_bridge_gradient.m - Gradient computation and related operations
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Suppress deprecation warnings for CLAPACK
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <Accelerate/Accelerate.h>

#include "metal_bridge.h"
#include <stdlib.h>

// External references to global variables from main bridge
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External helper functions
extern void set_c_error_message(CError *err, NSString *format, ...);

// ==============================================================================
// Gradient Computation Operations
// ==============================================================================

// Gradient accumulation (existing += new)
int perform_gradient_accumulate(
    GPUPtr existingGradPtr,
    GPUPtr newGradPtr,
    long size,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> existing_buffer = (__bridge id<MTLBuffer>)existingGradPtr;
        id<MTLBuffer> new_buffer = (__bridge id<MTLBuffer>)newGradPtr;

        if (!existing_buffer || !new_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for gradient accumulation.");
            return -1;
        }

        float *existing_data = (float*)existing_buffer.contents;
        float *new_data = (float*)new_buffer.contents;
        
        // Simple CPU implementation: existing[i] += new[i]
        for (long i = 0; i < size; i++) {
            existing_data[i] += new_data[i];
        }
        
        return 0;
    }
}

// Sum of squares computation
int perform_tensor_sum_squares(
    GPUPtr inputPtr,
    long size,
    float *sumSquares,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;

        if (!input_buffer || !sumSquares) {
            set_c_error_message(err, @"Invalid buffer pointer or output pointer for sum squares.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        
        float sum = 0.0f;
        for (long i = 0; i < size; i++) {
            float val = input_data[i];
            sum += val * val;
        }
        
        *sumSquares = sum;
        return 0;
    }
}

// Sum along specific axis
int perform_sum_along_axis(
    GPUPtr inputPtr,
    int axis,
    long inputNDim,
    long *inputShape,
    GPUPtr outputPtr,
    long outputNDim,
    long *outputShape,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for sum along axis.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Calculate input and output sizes
        long input_size = 1;
        for (long i = 0; i < inputNDim; i++) {
            input_size *= inputShape[i];
        }
        
        long output_size = 1;
        for (long i = 0; i < outputNDim; i++) {
            output_size *= outputShape[i];
        }
        
        // Initialize output to zero
        memset(output_data, 0, output_size * sizeof(float));
        
        // For simplicity, implement only for 2D case (most common)
        if (inputNDim == 2) {
            long rows = inputShape[0];
            long cols = inputShape[1];
            
            if (axis == 0) {
                // Sum along rows (result has shape [cols])
                for (long j = 0; j < cols; j++) {
                    float sum = 0.0f;
                    for (long i = 0; i < rows; i++) {
                        sum += input_data[i * cols + j];
                    }
                    output_data[j] = sum;
                }
            } else if (axis == 1) {
                // Sum along columns (result has shape [rows])
                for (long i = 0; i < rows; i++) {
                    float sum = 0.0f;
                    for (long j = 0; j < cols; j++) {
                        sum += input_data[i * cols + j];
                    }
                    output_data[i] = sum;
                }
            }
        } else if (inputNDim == 1) {
            // Sum all elements for 1D tensor
            float sum = 0.0f;
            for (long i = 0; i < input_size; i++) {
                sum += input_data[i];
            }
            output_data[0] = sum;
        } else {
            set_c_error_message(err, @"Sum along axis only supports 1D and 2D tensors currently.");
            return -1;
        }
        
        return 0;
    }
}

// Broadcast gradient from reduced shape back to original shape
int perform_broadcast_gradient(
    GPUPtr gradPtr,
    long gradNDim,
    long *gradShape,
    GPUPtr outputPtr,
    long outputNDim,
    long *outputShape,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!grad_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for broadcast gradient.");
            return -1;
        }

        float *grad_data = (float*)grad_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Calculate sizes
        long grad_size = 1;
        for (long i = 0; i < gradNDim; i++) {
            grad_size *= gradShape[i];
        }
        
        long output_size = 1;
        for (long i = 0; i < outputNDim; i++) {
            output_size *= outputShape[i];
        }
        
        // Simple case: broadcast scalar to any shape
        if (grad_size == 1) {
            float grad_val = grad_data[0];
            for (long i = 0; i < output_size; i++) {
                output_data[i] = grad_val;
            }
            return 0;
        }
        
        // For 2D case: broadcast along appropriate dimensions
        if (gradNDim == 1 && outputNDim == 2) {
            long grad_len = gradShape[0];
            long out_rows = outputShape[0];
            long out_cols = outputShape[1];
            
            if (grad_len == out_cols) {
                // Broadcast along rows
                for (long i = 0; i < out_rows; i++) {
                    for (long j = 0; j < out_cols; j++) {
                        output_data[i * out_cols + j] = grad_data[j];
                    }
                }
            } else if (grad_len == out_rows) {
                // Broadcast along columns
                for (long i = 0; i < out_rows; i++) {
                    for (long j = 0; j < out_cols; j++) {
                        output_data[i * out_cols + j] = grad_data[i];
                    }
                }
            } else {
                set_c_error_message(err, @"Unsupported broadcast pattern for gradient.");
                return -1;
            }
        } else {
            // If shapes match, just copy
            if (grad_size == output_size) {
                memcpy(output_data, grad_data, output_size * sizeof(float));
            } else {
                set_c_error_message(err, @"Unsupported gradient broadcasting case.");
                return -1;
            }
        }
        
        return 0;
    }
}

// Element-wise gradient scaling
int perform_gradient_scale(
    GPUPtr gradPtr,
    long size,
    float scale,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;

        if (!grad_buffer) {
            set_c_error_message(err, @"Invalid buffer pointer for gradient scaling.");
            return -1;
        }

        float *grad_data = (float*)grad_buffer.contents;
        
        // Scale all elements: grad[i] *= scale
        for (long i = 0; i < size; i++) {
            grad_data[i] *= scale;
        }
        
        return 0;
    }
}

// ==============================================================================
// Tensor Clamping Operations
// ==============================================================================

// Element-wise maximum clamping
int perform_tensor_clamp_max(
    GPUPtr inputPtr,
    long size,
    float maxValue,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for tensor clamp max.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Clamp each element: output[i] = min(input[i], maxValue)
        for (long i = 0; i < size; i++) {
            output_data[i] = fminf(input_data[i], maxValue);
        }
        
        return 0;
    }
}

// Element-wise minimum clamping
int perform_tensor_clamp_min(
    GPUPtr inputPtr,
    long size,
    float minValue,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for tensor clamp min.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Clamp each element: output[i] = max(input[i], minValue)
        for (long i = 0; i < size; i++) {
            output_data[i] = fmaxf(input_data[i], minValue);
        }
        
        return 0;
    }
}

// Combined clamp operation
int perform_tensor_clamp(
    GPUPtr inputPtr,
    long size,
    float minValue,
    float maxValue,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for tensor clamp.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Clamp each element: output[i] = clamp(input[i], minValue, maxValue)
        for (long i = 0; i < size; i++) {
            output_data[i] = fmaxf(minValue, fminf(input_data[i], maxValue));
        }
        
        return 0;
    }
}

// ==============================================================================
// Tensor Reduction Operations
// ==============================================================================

// Compute L2 norm of a tensor
int perform_tensor_l2_norm(
    GPUPtr inputPtr,
    long size,
    float *norm,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;

        if (!input_buffer || !norm) {
            set_c_error_message(err, @"Invalid buffer pointer or output pointer for L2 norm.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        
        float sum_squares = 0.0f;
        for (long i = 0; i < size; i++) {
            float val = input_data[i];
            sum_squares += val * val;
        }
        
        *norm = sqrtf(sum_squares);
        return 0;
    }
}

// Reduce sum across all elements
int perform_tensor_sum_all(
    GPUPtr inputPtr,
    long size,
    float *sum,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;

        if (!input_buffer || !sum) {
            set_c_error_message(err, @"Invalid buffer pointer or output pointer for tensor sum all.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        
        float total = 0.0f;
        for (long i = 0; i < size; i++) {
            total += input_data[i];
        }
        
        *sum = total;
        return 0;
    }
}

// Reduce mean across all elements
int perform_tensor_mean_all(
    GPUPtr inputPtr,
    long size,
    float *mean,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;

        if (!input_buffer || !mean) {
            set_c_error_message(err, @"Invalid buffer pointer or output pointer for tensor mean all.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        
        float total = 0.0f;
        for (long i = 0; i < size; i++) {
            total += input_data[i];
        }
        
        *mean = total / (float)size;
        return 0;
    }
}

// Reduce max across all elements
int perform_tensor_max_all(
    GPUPtr inputPtr,
    long size,
    float *maxValue,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;

        if (!input_buffer || !maxValue) {
            set_c_error_message(err, @"Invalid buffer pointer or output pointer for tensor max all.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        
        if (size == 0) {
            set_c_error_message(err, @"Cannot compute max of empty tensor.");
            return -1;
        }
        
        float max_val = input_data[0];
        for (long i = 1; i < size; i++) {
            max_val = fmaxf(max_val, input_data[i]);
        }
        
        *maxValue = max_val;
        return 0;
    }
}

// Reduce min across all elements
int perform_tensor_min_all(
    GPUPtr inputPtr,
    long size,
    float *minValue,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;

        if (!input_buffer || !minValue) {
            set_c_error_message(err, @"Invalid buffer pointer or output pointer for tensor min all.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        
        if (size == 0) {
            set_c_error_message(err, @"Cannot compute min of empty tensor.");
            return -1;
        }
        
        float min_val = input_data[0];
        for (long i = 1; i < size; i++) {
            min_val = fminf(min_val, input_data[i]);
        }
        
        *minValue = min_val;
        return 0;
    }
}

// ==============================================================================
// Dropout Operations
// ==============================================================================

// Simple dropout implementation for training
int perform_dropout_forward(
    GPUPtr inputPtr,
    long size,
    float probability,
    unsigned int seed,
    GPUPtr outputPtr,
    GPUPtr maskPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;
        id<MTLBuffer> mask_buffer = (__bridge id<MTLBuffer>)maskPtr;

        if (!input_buffer || !output_buffer || !mask_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for dropout forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        float *mask_data = (float*)mask_buffer.contents;
        
        // Simple random number generation for dropout
        srand(seed);
        float scale = 1.0f / (1.0f - probability);
        
        for (long i = 0; i < size; i++) {
            float random_val = (float)rand() / RAND_MAX;
            if (random_val < probability) {
                // Drop this element
                mask_data[i] = 0.0f;
                output_data[i] = 0.0f;
            } else {
                // Keep this element and scale
                mask_data[i] = scale;
                output_data[i] = input_data[i] * scale;
            }
        }
        
        return 0;
    }
}

// Apply saved dropout mask during backward pass
int perform_dropout_backward(
    GPUPtr gradOutputPtr,
    GPUPtr maskPtr,
    long size,
    float probability,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> mask_buffer = (__bridge id<MTLBuffer>)maskPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !mask_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for dropout backward.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *mask_data = (float*)mask_buffer.contents;
        float *grad_input_data = (float*)grad_input_buffer.contents;
        
        // Apply the same mask to gradients
        for (long i = 0; i < size; i++) {
            grad_input_data[i] = grad_output_data[i] * mask_data[i];
        }
        
        return 0;
    }
}

// ==============================================================================
// Tensor Utility Operations
// ==============================================================================

// Copy tensor data
int perform_tensor_copy(
    GPUPtr srcPtr,
    GPUPtr dstPtr,
    long size,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)srcPtr;
        id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dstPtr;

        if (!src_buffer || !dst_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for tensor copy.");
            return -1;
        }

        float *src_data = (float*)src_buffer.contents;
        float *dst_data = (float*)dst_buffer.contents;
        
        memcpy(dst_data, src_data, size * sizeof(float));
        return 0;
    }
}

// ==============================================================================
// Gradient Clipping Operations
// ==============================================================================

// Gradient clipping by global norm
int perform_gradient_clip_by_norm(
    GPUPtr gradPtr,
    long size,
    float maxNorm,
    float *actualNorm,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;

        if (!grad_buffer || !actualNorm) {
            set_c_error_message(err, @"Invalid buffer pointer or actualNorm pointer for gradient clipping.");
            return -1;
        }

        float *grad_data = (float*)grad_buffer.contents;
        
        // Compute L2 norm
        float norm_squared = 0.0f;
        for (long i = 0; i < size; i++) {
            norm_squared += grad_data[i] * grad_data[i];
        }
        
        float norm = sqrtf(norm_squared);
        *actualNorm = norm;
        
        // Apply clipping if necessary
        if (norm > maxNorm) {
            float scale = maxNorm / norm;
            for (long i = 0; i < size; i++) {
                grad_data[i] *= scale;
            }
        }
        
        return 0;
    }
}

// Gradient clipping by value
int perform_gradient_clip_by_value(
    GPUPtr gradPtr,
    long size,
    float minValue,
    float maxValue,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;

        if (!grad_buffer) {
            set_c_error_message(err, @"Invalid buffer pointer for gradient value clipping.");
            return -1;
        }

        float *grad_data = (float*)grad_buffer.contents;
        
        for (long i = 0; i < size; i++) {
            grad_data[i] = fmaxf(minValue, fminf(grad_data[i], maxValue));
        }
        
        return 0;
    }
}

// Global gradient norm computation
int perform_global_gradient_norm(
    GPUPtr *gradPtrs,
    long *sizes,
    long numTensors,
    float *globalNorm,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        if (!gradPtrs || !sizes || !globalNorm) {
            set_c_error_message(err, @"Invalid pointers for global gradient norm computation.");
            return -1;
        }
        
        float global_norm_squared = 0.0f;
        
        for (long t = 0; t < numTensors; t++) {
            id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtrs[t];
            if (!grad_buffer) {
                set_c_error_message(err, @"Invalid gradient buffer at index %ld.", t);
                return -1;
            }
            
            float *grad_data = (float*)grad_buffer.contents;
            long size = sizes[t];
            
            for (long i = 0; i < size; i++) {
                global_norm_squared += grad_data[i] * grad_data[i];
            }
        }
        
        *globalNorm = sqrtf(global_norm_squared);
        return 0;
    }
}

// Apply gradient clipping to multiple tensors simultaneously
int perform_global_gradient_clip(
    GPUPtr *gradPtrs,
    long *sizes,
    long numTensors,
    float maxNorm,
    float *actualNorm,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // First compute global norm
        int result = perform_global_gradient_norm(gradPtrs, sizes, numTensors, actualNorm, mtlDevicePtr, err);
        if (result != 0) {
            return result;
        }
        
        // Apply clipping if necessary
        if (*actualNorm > maxNorm) {
            float scale = maxNorm / (*actualNorm);
            
            for (long t = 0; t < numTensors; t++) {
                id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtrs[t];
                float *grad_data = (float*)grad_buffer.contents;
                long size = sizes[t];
                
                for (long i = 0; i < size; i++) {
                    grad_data[i] *= scale;
                }
            }
        }
        
        return 0;
    }
}

// Set tensor elements to a specific value (for gradient zeroing)
int perform_tensor_fill(
    GPUPtr tensorPtr,       // Tensor to fill
    long size,              // Number of elements
    float value,            // Value to fill with
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)tensorPtr;
        
        if (!device) {
            set_c_error_message(err, @"Metal device is NULL.");
            return -1;
        }
        if (!buffer) {
            set_c_error_message(err, @"Metal buffer is NULL.");
            return -1;
        }
        
        // Validate buffer size
        if (buffer.length < size * sizeof(float)) {
            set_c_error_message(err, @"Buffer size is smaller than requested fill size.");
            return -1;
        }
        
        // Simple CPU implementation for reliability
        float *data = (float*)buffer.contents;
        for (long i = 0; i < size; i++) {
            data[i] = value;
        }
        
        return 0;
    }
}

#pragma clang diagnostic pop