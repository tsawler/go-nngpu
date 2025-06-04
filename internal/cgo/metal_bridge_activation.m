// metal_bridge_activation.m - Activation Functions Module
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
#include <math.h>

// External declarations for global Metal device and command queue
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External helper function for setting error messages
extern void set_c_error_message(CError *err, NSString *format, ...);

// --- Phase 6A: Activation Functions ---

// ReLU activation function: f(x) = max(0, x)
int perform_activation_relu_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for ReLU forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply ReLU: output[i] = max(0, input[i])
        for (long i = 0; i < size; i++) {
            output_data[i] = fmaxf(0.0f, input_data[i]);
        }
        
        return 0;
    }
}

// ReLU derivative: f'(x) = 1 if x > 0, 0 otherwise
int perform_activation_relu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for ReLU backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // ReLU derivative: grad_input[i] = grad_output[i] if activation_output[i] > 0, 0 otherwise
        for (long i = 0; i < size; i++) {
            grad_input[i] = (activation_output[i] > 0.0f) ? grad_output[i] : 0.0f;
        }
        
        return 0;
    }
}

// Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
int perform_activation_sigmoid_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Sigmoid forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply Sigmoid: output[i] = 1 / (1 + exp(-input[i]))
        for (long i = 0; i < size; i++) {
            // Clamp input to prevent overflow
            float x = fmaxf(-88.0f, fminf(88.0f, input_data[i]));
            output_data[i] = 1.0f / (1.0f + expf(-x));
        }
        
        return 0;
    }
}

// Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
int perform_activation_sigmoid_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Sigmoid backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // Sigmoid derivative: grad_input[i] = grad_output[i] * activation_output[i] * (1 - activation_output[i])
        for (long i = 0; i < size; i++) {
            float sigmoid_out = activation_output[i];
            grad_input[i] = grad_output[i] * sigmoid_out * (1.0f - sigmoid_out);
        }
        
        return 0;
    }
}

// Tanh activation function: f(x) = tanh(x)
int perform_activation_tanh_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Tanh forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply Tanh: output[i] = tanh(input[i])
        for (long i = 0; i < size; i++) {
            // Clamp input to prevent overflow
            float x = fmaxf(-88.0f, fminf(88.0f, input_data[i]));
            output_data[i] = tanhf(x);
        }
        
        return 0;
    }
}

// Tanh derivative: f'(x) = 1 - tanh²(x)
int perform_activation_tanh_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Tanh backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // Tanh derivative: grad_input[i] = grad_output[i] * (1 - tanh²(activation_output[i]))
        for (long i = 0; i < size; i++) {
            float tanh_out = activation_output[i];
            grad_input[i] = grad_output[i] * (1.0f - tanh_out * tanh_out);
        }
        
        return 0;
    }
}

// Softmax activation function (1D): f(x_i) = exp(x_i) / sum(exp(x_j))
int perform_activation_softmax_1d_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Softmax 1D forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Find max for numerical stability
        float max_val = input_data[0];
        for (long i = 1; i < size; i++) {
            max_val = fmaxf(max_val, input_data[i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (long i = 0; i < size; i++) {
            output_data[i] = expf(input_data[i] - max_val);
            sum += output_data[i];
        }
        
        // Normalize
        for (long i = 0; i < size; i++) {
            output_data[i] /= sum;
        }
        
        return 0;
    }
}

// Softmax derivative (1D): Jacobian matrix computation
int perform_activation_softmax_1d_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Softmax 1D backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // Compute dot product of grad_output and activation_output
        float dot_product = 0.0f;
        for (long i = 0; i < size; i++) {
            dot_product += grad_output[i] * activation_output[i];
        }
        
        // Softmax gradient: grad_input[i] = activation_output[i] * (grad_output[i] - dot_product)
        for (long i = 0; i < size; i++) {
            grad_input[i] = activation_output[i] * (grad_output[i] - dot_product);
        }
        
        return 0;
    }
}

// Softmax activation function (2D - batch processing)
int perform_activation_softmax_2d_forward(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Softmax 2D forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply softmax to each row (sample) in the batch
        for (long b = 0; b < batchSize; b++) {
            long offset = b * features;
            
            // Find max for numerical stability
            float max_val = input_data[offset];
            for (long i = 1; i < features; i++) {
                max_val = fmaxf(max_val, input_data[offset + i]);
            }
            
            // Compute exponentials and sum
            float sum = 0.0f;
            for (long i = 0; i < features; i++) {
                output_data[offset + i] = expf(input_data[offset + i] - max_val);
                sum += output_data[offset + i];
            }
            
            // Normalize
            for (long i = 0; i < features; i++) {
                output_data[offset + i] /= sum;
            }
        }
        
        return 0;
    }
}

// Softmax derivative (2D - batch processing)
int perform_activation_softmax_2d_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long batchSize, long features,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Softmax 2D backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // Apply softmax gradient to each row (sample) in the batch
        for (long b = 0; b < batchSize; b++) {
            long offset = b * features;
            
            // Compute dot product for this sample
            float dot_product = 0.0f;
            for (long i = 0; i < features; i++) {
                dot_product += grad_output[offset + i] * activation_output[offset + i];
            }
            
            // Apply softmax gradient
            for (long i = 0; i < features; i++) {
                grad_input[offset + i] = activation_output[offset + i] * 
                                       (grad_output[offset + i] - dot_product);
            }
        }
        
        return 0;
    }
}

// Leaky ReLU activation function: f(x) = max(alpha * x, x)
int perform_activation_leaky_relu_forward(
    GPUPtr inputPtr, long size,
    float alpha,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Leaky ReLU forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply Leaky ReLU: output[i] = max(alpha * input[i], input[i])
        for (long i = 0; i < size; i++) {
            float x = input_data[i];
            output_data[i] = (x > 0.0f) ? x : alpha * x;
        }
        
        return 0;
    }
}

// Leaky ReLU derivative: f'(x) = 1 if x > 0, alpha otherwise
int perform_activation_leaky_relu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    float alpha,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Leaky ReLU backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // Leaky ReLU derivative: we need to use the original input, but we can infer from output
        for (long i = 0; i < size; i++) {
            // If activation_output > 0, original input was positive
            // If activation_output <= 0, original input was negative (since output = alpha * input)
            float derivative = (activation_output[i] > 0.0f) ? 1.0f : alpha;
            grad_input[i] = grad_output[i] * derivative;
        }
        
        return 0;
    }
}

// ELU activation function: f(x) = x if x > 0, alpha * (exp(x) - 1) otherwise
int perform_activation_elu_forward(
    GPUPtr inputPtr, long size,
    float alpha,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for ELU forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply ELU: output[i] = x if x > 0, alpha * (exp(x) - 1) otherwise
        for (long i = 0; i < size; i++) {
            float x = input_data[i];
            if (x > 0.0f) {
                output_data[i] = x;
            } else {
                // Clamp to prevent overflow
                x = fmaxf(-88.0f, x);
                output_data[i] = alpha * (expf(x) - 1.0f);
            }
        }
        
        return 0;
    }
}

// ELU derivative: f'(x) = 1 if x > 0, alpha * exp(x) otherwise
int perform_activation_elu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    float alpha,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for ELU backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // ELU derivative
        for (long i = 0; i < size; i++) {
            float y = activation_output[i];
            float derivative;
            
            if (y > 0.0f) {
                // Original input was positive
                derivative = 1.0f;
            } else {
                // Original input was negative, y = alpha * (exp(x) - 1)
                // So exp(x) = (y / alpha) + 1, and derivative = alpha * exp(x) = alpha * ((y / alpha) + 1) = y + alpha
                derivative = y + alpha;
            }
            
            grad_input[i] = grad_output[i] * derivative;
        }
        
        return 0;
    }
}

// Swish activation function: f(x) = x * sigmoid(x)
int perform_activation_swish_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Swish forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply Swish: output[i] = x * sigmoid(x) = x / (1 + exp(-x))
        for (long i = 0; i < size; i++) {
            float x = input_data[i];
            // Clamp to prevent overflow
            x = fmaxf(-88.0f, fminf(88.0f, x));
            float sigmoid = 1.0f / (1.0f + expf(-x));
            output_data[i] = x * sigmoid;
        }
        
        return 0;
    }
}

// Swish derivative: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
int perform_activation_swish_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Swish backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // For Swish gradient, we need the original input
        // We can derive it from the activation output using numerical methods
        // But for efficiency, we'll store intermediate sigmoid values during forward pass
        // For now, we'll recompute sigmoid from the ratio y/x where y is activation output
        
        for (long i = 0; i < size; i++) {
            float y = activation_output[i];
            
            // Handle edge case where y is very close to 0
            if (fabsf(y) < 1e-8f) {
                grad_input[i] = grad_output[i] * 0.5f; // Derivative at x=0
                continue;
            }
            
            // For swish, we need to solve y = x * sigmoid(x) for x
            // This is complex, so we'll use an approximation
            // For practical purposes, we can use the fact that sigmoid(x) ≈ y/x for the range we care about
            
            // Estimate x from y using iterative approach (simplified)
            float x = y; // Initial guess
            for (int iter = 0; iter < 3; iter++) {
                float sigmoid_x = 1.0f / (1.0f + expf(-x));
                float f = x * sigmoid_x - y;
                float df = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
                if (fabsf(df) > 1e-8f) {
                    x = x - f / df;
                }
            }
            
            // Now compute the derivative
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            float derivative = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
            grad_input[i] = grad_output[i] * derivative;
        }
        
        return 0;
    }
}

// GELU activation function: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
int perform_activation_gelu_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for GELU forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
        const float coeff = 0.044715f;
        
        // Apply GELU
        for (long i = 0; i < size; i++) {
            float x = input_data[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            // Clamp to prevent overflow in tanh
            inner = fmaxf(-88.0f, fminf(88.0f, inner));
            output_data[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
        
        return 0;
    }
}

// GELU derivative
int perform_activation_gelu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> activation_output_buffer = (__bridge id<MTLBuffer>)activationOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !activation_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for GELU backward.");
            return -1;
        }

        float *grad_output = (float*)grad_output_buffer.contents;
        float *activation_output = (float*)activation_output_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        
        // For GELU gradient, we need the original input
        // We'll use numerical approximation to recover x from activation output
        for (long i = 0; i < size; i++) {
            float y = activation_output[i];
            
            // Use Newton-Raphson to find x such that GELU(x) = y
            float x = y; // Initial guess
            for (int iter = 0; iter < 3; iter++) {
                float x_cubed = x * x * x;
                float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
                inner = fmaxf(-88.0f, fminf(88.0f, inner));
                float tanh_inner = tanhf(inner);
                
                float f = 0.5f * x * (1.0f + tanh_inner) - y;
                
                // Derivative for Newton-Raphson
                float dtanh_dx = 1.0f - tanh_inner * tanh_inner;
                float dinner_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
                float df = 0.5f * (1.0f + tanh_inner) + 0.5f * x * dtanh_dx * dinner_dx;
                
                if (fabsf(df) > 1e-8f) {
                    x = x - f / df;
                }
            }
            
            // Now compute the actual derivative
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            inner = fmaxf(-88.0f, fminf(88.0f, inner));
            float tanh_inner = tanhf(inner);
            float sech2_inner = 1.0f - tanh_inner * tanh_inner;
            float dinner_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
            
            float derivative = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2_inner * dinner_dx;
            grad_input[i] = grad_output[i] * derivative;
        }
        
        return 0;
    }
}