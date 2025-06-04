// metal_bridge_loss.m - Loss Functions Implementation
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Suppress deprecation warnings for CLAPACK
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <Accelerate/Accelerate.h>
#pragma clang diagnostic pop

#include "metal_bridge.h"
#include <stdlib.h>

// External declarations for global variables
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External function declarations
extern void set_c_error_message(CError *err, NSString *format, ...);
extern void free_c_error_message(char *message);

// =============================================================================
// Loss Function Implementations
// =============================================================================

// Mean Squared Error (MSE) loss: L = (1/n) * sum((pred - target)²)
int perform_loss_mse_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for MSE forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        // Compute MSE: (1/n) * sum((pred - target)²)
        float sum_squared_error = 0.0f;
        for (long i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            sum_squared_error += diff * diff;
        }
        
        *loss = sum_squared_error / (float)size;
        return 0;
    }
}

// MSE gradient: d/dpred = (2/n) * (pred - target)
int perform_loss_mse_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for MSE backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // MSE gradient: (2/n) * (pred - target)
        float scale = 2.0f / (float)size;
        for (long i = 0; i < size; i++) {
            grad_input[i] = scale * (predictions[i] - targets[i]);
        }
        
        return 0;
    }
}

// Binary Cross-Entropy loss: L = -(1/n) * sum(target * log(pred) + (1-target) * log(1-pred))
int perform_loss_binary_crossentropy_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Binary CrossEntropy forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        const float epsilon = 1e-7f; // Small value to prevent log(0)
        float sum_loss = 0.0f;
        
        for (long i = 0; i < size; i++) {
            // Clamp predictions to prevent log(0) or log(1)
            float pred = fmaxf(epsilon, fminf(1.0f - epsilon, predictions[i]));
            float target = targets[i];
            
            sum_loss += -(target * logf(pred) + (1.0f - target) * logf(1.0f - pred));
        }
        
        *loss = sum_loss / (float)size;
        return 0;
    }
}

// Binary Cross-Entropy gradient: d/dpred = (1/n) * (pred - target) / (pred * (1 - pred))
int perform_loss_binary_crossentropy_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Binary CrossEntropy backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float epsilon = 1e-7f;
        float scale = 1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            // Clamp predictions to prevent division by zero
            float pred = fmaxf(epsilon, fminf(1.0f - epsilon, predictions[i]));
            float target = targets[i];
            
            // Simplified gradient: (pred - target) / (pred * (1 - pred))
            grad_input[i] = scale * (pred - target) / (pred * (1.0f - pred));
        }
        
        return 0;
    }
}

// Categorical Cross-Entropy loss: L = -(1/n) * sum_i sum_j target[i][j] * log(pred[i][j])
int perform_loss_categorical_crossentropy_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long batchSize, long numClasses,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Categorical CrossEntropy forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        const float epsilon = 1e-7f;
        float sum_loss = 0.0f;
        
        for (long b = 0; b < batchSize; b++) {
            for (long c = 0; c < numClasses; c++) {
                long idx = b * numClasses + c;
                float pred = fmaxf(epsilon, predictions[idx]); // Prevent log(0)
                float target = targets[idx];
                
                sum_loss += -target * logf(pred);
            }
        }
        
        *loss = sum_loss / (float)batchSize;
        return 0;
    }
}

// Categorical Cross-Entropy gradient: d/dpred[i][j] = -(1/n) * target[i][j] / pred[i][j]
int perform_loss_categorical_crossentropy_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long batchSize, long numClasses,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Categorical CrossEntropy backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float epsilon = 1e-7f;
        float scale = -1.0f / (float)batchSize;
        
        for (long b = 0; b < batchSize; b++) {
            for (long c = 0; c < numClasses; c++) {
                long idx = b * numClasses + c;
                float pred = fmaxf(epsilon, predictions[idx]);
                float target = targets[idx];
                
                grad_input[idx] = scale * target / pred;
            }
        }
        
        return 0;
    }
}

// Sparse Categorical Cross-Entropy loss: targets are class indices
int perform_loss_sparse_categorical_crossentropy_forward(
    GPUPtr predictionsPtr, int *targetIndices, long batchSize, long numClasses,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;

        if (!predictions_buffer || !targetIndices || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Sparse Categorical CrossEntropy forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        
        const float epsilon = 1e-7f;
        float sum_loss = 0.0f;
        
        for (long b = 0; b < batchSize; b++) {
            int target_class = targetIndices[b];
            
            if (target_class < 0 || target_class >= numClasses) {
                set_c_error_message(err, @"Target class index %d out of bounds [0, %ld) for batch %ld.", target_class, numClasses, b);
                return -2;
            }
            
            long idx = b * numClasses + target_class;
            float pred = fmaxf(epsilon, predictions[idx]);
            
            sum_loss += -logf(pred);
        }
        
        *loss = sum_loss / (float)batchSize;
        return 0;
    }
}

// Sparse Categorical Cross-Entropy gradient
int perform_loss_sparse_categorical_crossentropy_backward(
    GPUPtr predictionsPtr, int *targetIndices, long batchSize, long numClasses,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targetIndices || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Sparse Categorical CrossEntropy backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float epsilon = 1e-7f;
        float scale = -1.0f / (float)batchSize;
        
        // Initialize gradients to zero
        memset(grad_input, 0, batchSize * numClasses * sizeof(float));
        
        for (long b = 0; b < batchSize; b++) {
            int target_class = targetIndices[b];
            
            if (target_class < 0 || target_class >= numClasses) {
                set_c_error_message(err, @"Target class index %d out of bounds [0, %ld) for batch %ld.", target_class, numClasses, b);
                return -2;
            }
            
            long idx = b * numClasses + target_class;
            float pred = fmaxf(epsilon, predictions[idx]);
            
            grad_input[idx] = scale / pred;
        }
        
        return 0;
    }
}

// Huber loss: smooth combination of MSE and MAE
int perform_loss_huber_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float delta,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Huber forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        float sum_loss = 0.0f;
        
        for (long i = 0; i < size; i++) {
            float diff = fabsf(predictions[i] - targets[i]);
            
            if (diff <= delta) {
                // Quadratic region: 0.5 * diff²
                sum_loss += 0.5f * diff * diff;
            } else {
                // Linear region: delta * (diff - 0.5 * delta)
                sum_loss += delta * (diff - 0.5f * delta);
            }
        }
        
        *loss = sum_loss / (float)size;
        return 0;
    }
}

// Huber loss gradient
int perform_loss_huber_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float delta,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Huber backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        float scale = 1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            float abs_diff = fabsf(diff);
            
            if (abs_diff <= delta) {
                // Quadratic region: gradient = diff
                grad_input[i] = scale * diff;
            } else {
                // Linear region: gradient = delta * sign(diff)
                grad_input[i] = scale * delta * ((diff > 0.0f) ? 1.0f : -1.0f);
            }
        }
        
        return 0;
    }
}

// Mean Absolute Error (MAE) loss: L = (1/n) * sum(|pred - target|)
int perform_loss_mae_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for MAE forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        float sum_abs_error = 0.0f;
        for (long i = 0; i < size; i++) {
            sum_abs_error += fabsf(predictions[i] - targets[i]);
        }
        
        *loss = sum_abs_error / (float)size;
        return 0;
    }
}

// MAE gradient: d/dpred = (1/n) * sign(pred - target)
int perform_loss_mae_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for MAE backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        float scale = 1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            // Sign function: 1 if positive, -1 if negative, 0 if zero
            grad_input[i] = scale * ((diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f);
        }
        
        return 0;
    }
}

// Hinge loss: L = (1/n) * sum(max(0, 1 - target * pred))
int perform_loss_hinge_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Hinge forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        float sum_loss = 0.0f;
        
        for (long i = 0; i < size; i++) {
            float margin = 1.0f - targets[i] * predictions[i];
            sum_loss += fmaxf(0.0f, margin);
        }
        
        *loss = sum_loss / (float)size;
        return 0;
    }
}

// Hinge loss gradient: d/dpred = -(1/n) * target if margin > 0, 0 otherwise
int perform_loss_hinge_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Hinge backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        float scale = -1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            float margin = 1.0f - targets[i] * predictions[i];
            
            if (margin > 0.0f) {
                grad_input[i] = scale * targets[i];
            } else {
                grad_input[i] = 0.0f;
            }
        }
        
        return 0;
    }
}