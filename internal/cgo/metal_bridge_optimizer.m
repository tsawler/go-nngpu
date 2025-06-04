// metal_bridge_optimizer.m - Optimizer implementations and learning rate schedulers
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
// Optimizers (SGD, Adam, RMSprop with GPU state)
// ==============================================================================

// SGD (Stochastic Gradient Descent) optimizer
int perform_sgd_step(
    GPUPtr paramsPtr,
    GPUPtr gradPtr,
    long size,
    float learningRate,
    float momentum,
    GPUPtr momentumBufferPtr,
    float weightDecay,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> momentum_buffer = (__bridge id<MTLBuffer>)momentumBufferPtr;

        if (!params_buffer || !grad_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for SGD step.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *momentum_data = momentum_buffer ? (float*)momentum_buffer.contents : NULL;
        
        // SGD with momentum: v = momentum * v + grad, params = params - lr * v
        // SGD without momentum: params = params - lr * grad
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                grad_val += weightDecay * params_data[i];
            }
            
            if (momentum != 0.0f && momentum_data != NULL) {
                // Update momentum buffer
                momentum_data[i] = momentum * momentum_data[i] + grad_val;
                // Update parameters using momentum
                params_data[i] -= learningRate * momentum_data[i];
            } else {
                // Simple SGD without momentum
                params_data[i] -= learningRate * grad_val;
            }
        }
        
        return 0;
    }
}

// Adam optimizer step
int perform_adam_step(
    GPUPtr paramsPtr,
    GPUPtr gradPtr,
    long size,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    float weightDecay,
    GPUPtr m_bufferPtr,
    GPUPtr v_bufferPtr,
    long stepNumber,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> m_buffer = (__bridge id<MTLBuffer>)m_bufferPtr;
        id<MTLBuffer> v_buffer = (__bridge id<MTLBuffer>)v_bufferPtr;

        if (!params_buffer || !grad_buffer || !m_buffer || !v_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Adam step.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *m_data = (float*)m_buffer.contents;
        float *v_data = (float*)v_buffer.contents;
        
        // Bias correction factors
        float bias_correction1 = 1.0f - powf(beta1, (float)stepNumber);
        float bias_correction2 = 1.0f - powf(beta2, (float)stepNumber);
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                grad_val += weightDecay * params_data[i];
            }
            
            // Update biased first moment estimate
            m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * grad_val;
            
            // Update biased second raw moment estimate
            v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * grad_val * grad_val;
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_data[i] / bias_correction1;
            
            // Compute bias-corrected second raw moment estimate
            float v_hat = v_data[i] / bias_correction2;
            
            // Update parameters
            params_data[i] -= learningRate * m_hat / (sqrtf(v_hat) + epsilon);
        }
        
        return 0;
    }
}

// RMSprop optimizer step
int perform_rmsprop_step(
    GPUPtr paramsPtr,
    GPUPtr gradPtr,
    long size,
    float learningRate,
    float alpha,
    float epsilon,
    float momentum,
    float weightDecay,
    GPUPtr squaredGradBufferPtr,
    GPUPtr momentumBufferPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> squared_grad_buffer = (__bridge id<MTLBuffer>)squaredGradBufferPtr;
        id<MTLBuffer> momentum_buffer = (__bridge id<MTLBuffer>)momentumBufferPtr;

        if (!params_buffer || !grad_buffer || !squared_grad_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for RMSprop step.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *squared_grad_data = (float*)squared_grad_buffer.contents;
        float *momentum_data = momentum_buffer ? (float*)momentum_buffer.contents : NULL;
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                grad_val += weightDecay * params_data[i];
            }
            
            // Update running average of squared gradients
            squared_grad_data[i] = alpha * squared_grad_data[i] + (1.0f - alpha) * grad_val * grad_val;
            
            // Compute update step
            float update = grad_val / (sqrtf(squared_grad_data[i]) + epsilon);
            
            if (momentum != 0.0f && momentum_data != NULL) {
                // Apply momentum
                momentum_data[i] = momentum * momentum_data[i] + update;
                params_data[i] -= learningRate * momentum_data[i];
            } else {
                // Direct update without momentum
                params_data[i] -= learningRate * update;
            }
        }
        
        return 0;
    }
}

// AdaGrad optimizer step
int perform_adagrad_step(
    GPUPtr paramsPtr,
    GPUPtr gradPtr,
    long size,
    float learningRate,
    float epsilon,
    float weightDecay,
    GPUPtr accumulatedSquaredGradPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> accumulated_buffer = (__bridge id<MTLBuffer>)accumulatedSquaredGradPtr;

        if (!params_buffer || !grad_buffer || !accumulated_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for AdaGrad step.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *accumulated_data = (float*)accumulated_buffer.contents;
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                grad_val += weightDecay * params_data[i];
            }
            
            // Accumulate squared gradients
            accumulated_data[i] += grad_val * grad_val;
            
            // Update parameters
            params_data[i] -= learningRate * grad_val / (sqrtf(accumulated_data[i]) + epsilon);
        }
        
        return 0;
    }
}

// Adadelta optimizer step
int perform_adadelta_step(
    GPUPtr paramsPtr,
    GPUPtr gradPtr,
    long size,
    float rho,
    float epsilon,
    float weightDecay,
    GPUPtr accumulatedGradPtr,
    GPUPtr accumulatedDeltaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> accumulated_grad_buffer = (__bridge id<MTLBuffer>)accumulatedGradPtr;
        id<MTLBuffer> accumulated_delta_buffer = (__bridge id<MTLBuffer>)accumulatedDeltaPtr;

        if (!params_buffer || !grad_buffer || !accumulated_grad_buffer || !accumulated_delta_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Adadelta step.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *accumulated_grad_data = (float*)accumulated_grad_buffer.contents;
        float *accumulated_delta_data = (float*)accumulated_delta_buffer.contents;
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i];
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                grad_val += weightDecay * params_data[i];
            }
            
            // Update running average of squared gradients
            accumulated_grad_data[i] = rho * accumulated_grad_data[i] + (1.0f - rho) * grad_val * grad_val;
            
            // Compute parameter update
            float rms_delta = sqrtf(accumulated_delta_data[i] + epsilon);
            float rms_grad = sqrtf(accumulated_grad_data[i] + epsilon);
            float delta = -(rms_delta / rms_grad) * grad_val;
            
            // Update running average of squared parameter updates
            accumulated_delta_data[i] = rho * accumulated_delta_data[i] + (1.0f - rho) * delta * delta;
            
            // Update parameters
            params_data[i] += delta;
        }
        
        return 0;
    }
}

// AdamW optimizer step (Adam with decoupled weight decay)
int perform_adamw_step(
    GPUPtr paramsPtr,
    GPUPtr gradPtr,
    long size,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    float weightDecay,
    GPUPtr m_bufferPtr,
    GPUPtr v_bufferPtr,
    long stepNumber,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> m_buffer = (__bridge id<MTLBuffer>)m_bufferPtr;
        id<MTLBuffer> v_buffer = (__bridge id<MTLBuffer>)v_bufferPtr;

        if (!params_buffer || !grad_buffer || !m_buffer || !v_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for AdamW step.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *m_data = (float*)m_buffer.contents;
        float *v_data = (float*)v_buffer.contents;
        
        // Bias correction factors
        float bias_correction1 = 1.0f - powf(beta1, (float)stepNumber);
        float bias_correction2 = 1.0f - powf(beta2, (float)stepNumber);
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i];
            
            // Update biased first moment estimate
            m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * grad_val;
            
            // Update biased second raw moment estimate
            v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * grad_val * grad_val;
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_data[i] / bias_correction1;
            
            // Compute bias-corrected second raw moment estimate
            float v_hat = v_data[i] / bias_correction2;
            
            // Update parameters with Adam update
            params_data[i] -= learningRate * m_hat / (sqrtf(v_hat) + epsilon);
            
            // Apply decoupled weight decay (AdamW)
            if (weightDecay != 0.0f) {
                params_data[i] -= learningRate * weightDecay * params_data[i];
            }
        }
        
        return 0;
    }
}

// ==============================================================================
// Learning Rate Schedulers
// ==============================================================================

// Exponential decay scheduler
int perform_lr_exponential_decay(
    float *currentLR,
    float initialLR,
    float decayRate,
    long stepNumber,
    long decaySteps,
    CError *err
) {
    @autoreleasepool {
        if (!currentLR) {
            set_c_error_message(err, @"Invalid currentLR pointer for exponential decay.");
            return -1;
        }
        
        long decay_count = stepNumber / decaySteps;
        *currentLR = initialLR * powf(decayRate, (float)decay_count);
        
        return 0;
    }
}

// Step decay scheduler
int perform_lr_step_decay(
    float *currentLR,
    float initialLR,
    float gamma,
    long stepNumber,
    long stepSize,
    CError *err
) {
    @autoreleasepool {
        if (!currentLR) {
            set_c_error_message(err, @"Invalid currentLR pointer for step decay.");
            return -1;
        }
        
        long decay_count = stepNumber / stepSize;
        *currentLR = initialLR * powf(gamma, (float)decay_count);
        
        return 0;
    }
}

// Cosine annealing scheduler
int perform_lr_cosine_annealing(
    float *currentLR,
    float initialLR,
    float minLR,
    long currentStep,
    long maxSteps,
    CError *err
) {
    @autoreleasepool {
        if (!currentLR) {
            set_c_error_message(err, @"Invalid currentLR pointer for cosine annealing.");
            return -1;
        }
        
        if (maxSteps <= 0) {
            set_c_error_message(err, @"Max steps must be positive for cosine annealing.");
            return -1;
        }
        
        float progress = (float)currentStep / (float)maxSteps;
        if (progress > 1.0f) progress = 1.0f;
        
        *currentLR = minLR + (initialLR - minLR) * 0.5f * (1.0f + cosf(M_PI * progress));
        
        return 0;
    }
}

// Linear decay scheduler
int perform_lr_linear_decay(
    float *currentLR,
    float initialLR,
    float finalLR,
    long currentStep,
    long totalSteps,
    CError *err
) {
    @autoreleasepool {
        if (!currentLR) {
            set_c_error_message(err, @"Invalid currentLR pointer for linear decay.");
            return -1;
        }
        
        if (totalSteps <= 0) {
            set_c_error_message(err, @"Total steps must be positive for linear decay.");
            return -1;
        }
        
        float progress = (float)currentStep / (float)totalSteps;
        if (progress > 1.0f) progress = 1.0f;
        
        *currentLR = initialLR + (finalLR - initialLR) * progress;
        
        return 0;
    }
}

// Polynomial decay scheduler
int perform_lr_polynomial_decay(
    float *currentLR,       // Current learning rate (input/output)
    float initialLR,        // Initial learning rate
    float finalLR,          // Final learning rate
    long stepNumber,        // Current step number
    long totalSteps,        // Total number of steps
    float power,            // Power of polynomial (typically 1.0 for linear)
    CError *err
) {
    @autoreleasepool {
        if (!currentLR) {
            set_c_error_message(err, @"Invalid currentLR pointer for polynomial decay.");
            return -1;
        }
        
        if (totalSteps <= 0) {
            set_c_error_message(err, @"Total steps must be positive for polynomial decay.");
            return -1;
        }
        
        float progress = (float)stepNumber / (float)totalSteps;
        if (progress > 1.0f) progress = 1.0f;
        
        *currentLR = finalLR + (initialLR - finalLR) * powf(1.0f - progress, power);
        
        return 0;
    }
}

// ==============================================================================
// Optimizer State Management
// ==============================================================================

// Initialize optimizer state buffers
int initialize_optimizer_state(
    GPUPtr statePtr,
    long size,
    float initialValue,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> state_buffer = (__bridge id<MTLBuffer>)statePtr;

        if (!state_buffer) {
            set_c_error_message(err, @"Invalid buffer pointer for optimizer state initialization.");
            return -1;
        }

        float *state_data = (float*)state_buffer.contents;
        
        // Initialize all elements to the specified value (usually 0.0)
        for (long i = 0; i < size; i++) {
            state_data[i] = initialValue;
        }
        
        return 0;
    }
}

// Zero optimizer state buffers
int zero_optimizer_state(
    GPUPtr statePtr,
    long size,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> state_buffer = (__bridge id<MTLBuffer>)statePtr;

        if (!state_buffer) {
            set_c_error_message(err, @"Invalid buffer pointer for optimizer state zeroing.");
            return -1;
        }

        float *state_data = (float*)state_buffer.contents;
        
        // Zero all elements
        memset(state_data, 0, size * sizeof(float));
        
        return 0;
    }
}

// Copy optimizer state
int copy_optimizer_state(
    GPUPtr srcStatePtr,
    GPUPtr dstStatePtr,
    long size,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)srcStatePtr;
        id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dstStatePtr;

        if (!src_buffer || !dst_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for optimizer state copy.");
            return -1;
        }

        float *src_data = (float*)src_buffer.contents;
        float *dst_data = (float*)dst_buffer.contents;
        
        memcpy(dst_data, src_data, size * sizeof(float));
        
        return 0;
    }
}

// ==============================================================================
// Advanced Optimizer Features
// ==============================================================================

// Fused Adam step with gradient accumulation
int perform_fused_adam_step_with_accumulation(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // New gradients
    GPUPtr accumulatedGradPtr, // Accumulated gradients buffer
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float beta1, float beta2, float epsilon, float weightDecay,
    GPUPtr m_bufferPtr, GPUPtr v_bufferPtr,
    long stepNumber,
    float accumulationSteps, // Number of gradient accumulation steps
    int isAccumulationStep, // 1 if this is an accumulation step, 0 if optimizer step
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_buffer = (__bridge id<MTLBuffer>)paramsPtr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtr;
        id<MTLBuffer> accum_grad_buffer = (__bridge id<MTLBuffer>)accumulatedGradPtr;
        id<MTLBuffer> m_buffer = (__bridge id<MTLBuffer>)m_bufferPtr;
        id<MTLBuffer> v_buffer = (__bridge id<MTLBuffer>)v_bufferPtr;

        if (!params_buffer || !grad_buffer || !accum_grad_buffer || !m_buffer || !v_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for fused Adam step with accumulation.");
            return -1;
        }

        float *params_data = (float*)params_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *accum_grad_data = (float*)accum_grad_buffer.contents;
        float *m_data = (float*)m_buffer.contents;
        float *v_data = (float*)v_buffer.contents;
        
        // Bias correction factors
        float bias_correction1 = 1.0f - powf(beta1, (float)stepNumber);
        float bias_correction2 = 1.0f - powf(beta2, (float)stepNumber);
        
        for (long i = 0; i < size; i++) {
            // Accumulate gradients first
            accum_grad_data[i] += grad_data[i];
            float effective_grad = accum_grad_data[i] / accumulationSteps;
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                effective_grad += weightDecay * params_data[i];
            }
            
            // Update biased first moment estimate
            m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * effective_grad;
            
            // Update biased second raw moment estimate
            v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * effective_grad * effective_grad;
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_data[i] / bias_correction1;
            
            // Compute bias-corrected second raw moment estimate
            float v_hat = v_data[i] / bias_correction2;
            
            // Update parameters
            params_data[i] -= learningRate * m_hat / (sqrtf(v_hat) + epsilon);
            
            // Reset accumulated gradients for next accumulation cycle
            accum_grad_data[i] = 0.0f;
        }
        
        return 0;
    }
}

// Mixed precision SGD step
int perform_mixed_precision_sgd_step(
    GPUPtr params_fp32_Ptr,    // FP32 master parameters
    GPUPtr params_fp16_Ptr,    // FP16 working parameters
    GPUPtr grad_fp16_Ptr,      // FP16 gradients
    long size,                 // Number of parameters
    float learningRate,        // Learning rate
    float momentum,            // Momentum coefficient
    GPUPtr momentumBufferPtr,  // Momentum buffer (FP32)
    float weightDecay,         // Weight decay coefficient
    float gradScale,           // Gradient scale factor for mixed precision
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> params_fp32_buffer = (__bridge id<MTLBuffer>)params_fp32_Ptr;
        id<MTLBuffer> params_fp16_buffer = (__bridge id<MTLBuffer>)params_fp16_Ptr;
        id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)grad_fp16_Ptr;
        id<MTLBuffer> momentum_buffer = (__bridge id<MTLBuffer>)momentumBufferPtr;

        if (!params_fp32_buffer || !params_fp16_buffer || !grad_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for mixed precision SGD step.");
            return -1;
        }

        float *params_fp32_data = (float*)params_fp32_buffer.contents;
        float *params_fp16_data = (float*)params_fp16_buffer.contents;
        float *grad_data = (float*)grad_buffer.contents;
        float *momentum_data = momentum_buffer ? (float*)momentum_buffer.contents : NULL;
        
        // Unscale gradients
        float inv_grad_scale = 1.0f / gradScale;
        
        for (long i = 0; i < size; i++) {
            float grad_val = grad_data[i] * inv_grad_scale;
            
            // Apply weight decay if specified
            if (weightDecay != 0.0f) {
                grad_val += weightDecay * params_fp32_data[i];
            }
            
            if (momentum != 0.0f && momentum_data != NULL) {
                // Update momentum buffer
                momentum_data[i] = momentum * momentum_data[i] + grad_val;
                // Update parameters using momentum
                params_fp32_data[i] -= learningRate * momentum_data[i];
            } else {
                // Simple SGD without momentum
                params_fp32_data[i] -= learningRate * grad_val;
            }
        }
        
        return 0;
    }
}

// Parameter gradient synchronization across multiple buffers
int synchronize_parameter_gradients(
    GPUPtr *gradPtrs,
    long *sizes,
    long numTensors,
    float scale,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        if (!gradPtrs || !sizes) {
            set_c_error_message(err, @"Invalid pointers for gradient synchronization.");
            return -1;
        }
        
        // Scale all gradients
        for (long t = 0; t < numTensors; t++) {
            id<MTLBuffer> grad_buffer = (__bridge id<MTLBuffer>)gradPtrs[t];
            if (!grad_buffer) {
                set_c_error_message(err, @"Invalid gradient buffer at index %ld.", t);
                return -1;
            }
            
            float *grad_data = (float*)grad_buffer.contents;
            long size = sizes[t];
            
            for (long i = 0; i < size; i++) {
                grad_data[i] *= scale;
            }
        }
        
        return 0;
    }
}

// Learning rate warmup
int perform_lr_warmup(
    float *currentLR,       // Current learning rate (input/output)
    float targetLR,         // Target learning rate after warmup
    long stepNumber,        // Current step number
    long warmupSteps,       // Number of warmup steps
    int warmupType,         // 0: linear, 1: exponential
    CError *err
) {
    @autoreleasepool {
        if (!currentLR) {
            set_c_error_message(err, @"Invalid currentLR pointer for warmup.");
            return -1;
        }
        
        if (stepNumber >= warmupSteps) {
            *currentLR = targetLR;
            return 0;
        }
        
        float progress = (float)stepNumber / (float)warmupSteps;
        
        if (warmupType == 0) {
            // Linear warmup
            *currentLR = targetLR * progress;
        } else {
            // Exponential warmup
            *currentLR = targetLR * powf(progress, 2.0f);
        }
        
        return 0;
    }
}