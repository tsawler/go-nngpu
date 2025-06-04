// metal_bridge_normalization.m - Normalization-related functions extracted from metal_bridge.m
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

// External declarations for global variables and helper functions
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External helper function declarations
extern void set_c_error_message(CError *err, NSString *format, ...);

// --- Batch Normalization Operations ---

// Batch mean computation (across batch dimension)
int perform_batch_mean(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr meanPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;

        if (!input_buffer || !mean_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for batch mean computation.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        
        // Initialize means to zero
        memset(mean_data, 0, features * sizeof(float));
        
        // Compute mean for each feature across the batch
        for (long f = 0; f < features; f++) {
            float sum = 0.0f;
            for (long b = 0; b < batchSize; b++) {
                long idx = b * features + f;
                sum += input_data[idx];
            }
            mean_data[f] = sum / (float)batchSize;
        }
        
        return 0;
    }
}

// Batch variance computation (across batch dimension)
int perform_batch_variance(
    GPUPtr inputPtr, GPUPtr meanPtr, long batchSize, long features,
    GPUPtr variancePtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;
        id<MTLBuffer> variance_buffer = (__bridge id<MTLBuffer>)variancePtr;

        if (!input_buffer || !mean_buffer || !variance_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for batch variance computation.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        float *variance_data = (float*)variance_buffer.contents;
        
        // Initialize variances to zero
        memset(variance_data, 0, features * sizeof(float));
        
        // Compute variance for each feature across the batch
        for (long f = 0; f < features; f++) {
            float sum_squared_diff = 0.0f;
            float mean_val = mean_data[f];
            
            for (long b = 0; b < batchSize; b++) {
                long idx = b * features + f;
                float diff = input_data[idx] - mean_val;
                sum_squared_diff += diff * diff;
            }
            variance_data[f] = sum_squared_diff / (float)batchSize;
        }
        
        return 0;
    }
}

// Batch normalization forward pass
int perform_batch_norm_forward(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr meanPtr, GPUPtr variancePtr,
    GPUPtr gammaPtr, GPUPtr betaPtr,
    float epsilon,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;
        id<MTLBuffer> variance_buffer = (__bridge id<MTLBuffer>)variancePtr;
        id<MTLBuffer> gamma_buffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> beta_buffer = (__bridge id<MTLBuffer>)betaPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !mean_buffer || !variance_buffer || !gamma_buffer || !beta_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for batch normalization forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        float *variance_data = (float*)variance_buffer.contents;
        float *gamma_data = (float*)gamma_buffer.contents;
        float *beta_data = (float*)beta_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Apply batch normalization: output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
        for (long b = 0; b < batchSize; b++) {
            for (long f = 0; f < features; f++) {
                long idx = b * features + f;
                
                float normalized = (input_data[idx] - mean_data[f]) / sqrtf(variance_data[f] + epsilon);
                output_data[idx] = gamma_data[f] * normalized + beta_data[f];
            }
        }
        
        return 0;
    }
}

// Batch normalization backward pass - input gradients
int perform_batch_norm_backward_input(
    GPUPtr gradOutputPtr, long batchSize, long features,
    GPUPtr inputPtr, GPUPtr meanPtr, GPUPtr variancePtr,
    GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;
        id<MTLBuffer> variance_buffer = (__bridge id<MTLBuffer>)variancePtr;
        id<MTLBuffer> gamma_buffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !input_buffer || !mean_buffer || !variance_buffer || !gamma_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for batch normalization backward input.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *input_data = (float*)input_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        float *variance_data = (float*)variance_buffer.contents;
        float *gamma_data = (float*)gamma_buffer.contents;
        float *grad_input_data = (float*)grad_input_buffer.contents;
        
        // Batch normalization gradient computation
        // This is a simplified version - the full gradient involves terms for mean and variance gradients
        for (long f = 0; f < features; f++) {
            float var_eps = variance_data[f] + epsilon;
            float inv_std = 1.0f / sqrtf(var_eps);
            float gamma_val = gamma_data[f];
            
            // Compute gradient sums for this feature
            float grad_sum = 0.0f;
            float grad_dot_sum = 0.0f;
            
            for (long b = 0; b < batchSize; b++) {
                long idx = b * features + f;
                float grad_out = grad_output_data[idx];
                float x_hat = (input_data[idx] - mean_data[f]) * inv_std;
                
                grad_sum += grad_out;
                grad_dot_sum += grad_out * x_hat;
            }
            
            // Apply gradient formula
            float inv_batch_size = 1.0f / (float)batchSize;
            
            for (long b = 0; b < batchSize; b++) {
                long idx = b * features + f;
                float grad_out = grad_output_data[idx];
                float x_hat = (input_data[idx] - mean_data[f]) * inv_std;
                
                float grad_x_hat = grad_out - inv_batch_size * grad_sum - inv_batch_size * x_hat * grad_dot_sum;
                grad_input_data[idx] = gamma_val * inv_std * grad_x_hat;
            }
        }
        
        return 0;
    }
}

// Batch normalization backward pass - parameter gradients
int perform_batch_norm_backward_params(
    GPUPtr gradOutputPtr, long batchSize, long features,
    GPUPtr inputPtr, GPUPtr meanPtr, GPUPtr variancePtr,
    float epsilon,
    GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;
        id<MTLBuffer> variance_buffer = (__bridge id<MTLBuffer>)variancePtr;
        id<MTLBuffer> grad_gamma_buffer = (__bridge id<MTLBuffer>)gradGammaPtr;
        id<MTLBuffer> grad_beta_buffer = (__bridge id<MTLBuffer>)gradBetaPtr;

        if (!grad_output_buffer || !input_buffer || !mean_buffer || !variance_buffer || !grad_gamma_buffer || !grad_beta_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for batch normalization backward params.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *input_data = (float*)input_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        float *variance_data = (float*)variance_buffer.contents;
        float *grad_gamma_data = (float*)grad_gamma_buffer.contents;
        float *grad_beta_data = (float*)grad_beta_buffer.contents;
        
        // Initialize gradients to zero
        memset(grad_gamma_data, 0, features * sizeof(float));
        memset(grad_beta_data, 0, features * sizeof(float));
        
        // Compute parameter gradients
        for (long f = 0; f < features; f++) {
            float inv_std = 1.0f / sqrtf(variance_data[f] + epsilon);
            float mean_val = mean_data[f];
            
            for (long b = 0; b < batchSize; b++) {
                long idx = b * features + f;
                float grad_out = grad_output_data[idx];
                float x_normalized = (input_data[idx] - mean_val) * inv_std;
                
                grad_gamma_data[f] += grad_out * x_normalized;
                grad_beta_data[f] += grad_out;
            }
        }
        
        return 0;
    }
}

// Layer normalization forward pass
int perform_layer_norm_forward(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr gammaPtr, GPUPtr betaPtr,
    float epsilon,
    GPUPtr outputPtr, GPUPtr meanPtr, GPUPtr variancePtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> gamma_buffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> beta_buffer = (__bridge id<MTLBuffer>)betaPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;
        id<MTLBuffer> variance_buffer = (__bridge id<MTLBuffer>)variancePtr;

        if (!input_buffer || !gamma_buffer || !beta_buffer || !output_buffer || !mean_buffer || !variance_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for layer normalization forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *gamma_data = (float*)gamma_buffer.contents;
        float *beta_data = (float*)beta_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        float *variance_data = (float*)variance_buffer.contents;
        
        // For each sample in the batch, normalize across features
        for (long b = 0; b < batchSize; b++) {
            // Compute mean for this sample
            float sum = 0.0f;
            for (long f = 0; f < features; f++) {
                long idx = b * features + f;
                sum += input_data[idx];
            }
            float sample_mean = sum / (float)features;
            mean_data[b] = sample_mean;
            
            // Compute variance for this sample
            float sum_squared_diff = 0.0f;
            for (long f = 0; f < features; f++) {
                long idx = b * features + f;
                float diff = input_data[idx] - sample_mean;
                sum_squared_diff += diff * diff;
            }
            float sample_variance = sum_squared_diff / (float)features;
            variance_data[b] = sample_variance;
            
            // Apply layer normalization for this sample
            float inv_std = 1.0f / sqrtf(sample_variance + epsilon);
            for (long f = 0; f < features; f++) {
                long idx = b * features + f;
                float normalized = (input_data[idx] - sample_mean) * inv_std;
                output_data[idx] = gamma_data[f] * normalized + beta_data[f];
            }
        }
        
        return 0;
    }
}

// Layer normalization backward pass
int perform_layer_norm_backward(
    GPUPtr gradOutputPtr, long batchSize, long features,
    GPUPtr inputPtr, GPUPtr meanPtr, GPUPtr variancePtr,
    GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> mean_buffer = (__bridge id<MTLBuffer>)meanPtr;
        id<MTLBuffer> variance_buffer = (__bridge id<MTLBuffer>)variancePtr;
        id<MTLBuffer> gamma_buffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;
        id<MTLBuffer> grad_gamma_buffer = (__bridge id<MTLBuffer>)gradGammaPtr;
        id<MTLBuffer> grad_beta_buffer = (__bridge id<MTLBuffer>)gradBetaPtr;

        if (!grad_output_buffer || !input_buffer || !mean_buffer || !variance_buffer || 
            !gamma_buffer || !grad_input_buffer || !grad_gamma_buffer || !grad_beta_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for layer normalization backward.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *input_data = (float*)input_buffer.contents;
        float *mean_data = (float*)mean_buffer.contents;
        float *variance_data = (float*)variance_buffer.contents;
        float *gamma_data = (float*)gamma_buffer.contents;
        float *grad_input_data = (float*)grad_input_buffer.contents;
        float *grad_gamma_data = (float*)grad_gamma_buffer.contents;
        float *grad_beta_data = (float*)grad_beta_buffer.contents;
        
        // Initialize parameter gradients to zero
        memset(grad_gamma_data, 0, features * sizeof(float));
        memset(grad_beta_data, 0, features * sizeof(float));
        
        // Process each sample independently
        for (long b = 0; b < batchSize; b++) {
            float sample_mean = mean_data[b];
            float sample_variance = variance_data[b];
            float inv_std = 1.0f / sqrtf(sample_variance + epsilon);
            
            // Compute gradient sums for this sample
            float grad_sum = 0.0f;
            float grad_dot_sum = 0.0f;
            
            for (long f = 0; f < features; f++) {
                long idx = b * features + f;
                float grad_out = grad_output_data[idx];
                float x_hat = (input_data[idx] - sample_mean) * inv_std;
                
                grad_sum += grad_out * gamma_data[f];
                grad_dot_sum += grad_out * gamma_data[f] * x_hat;
                
                // Accumulate parameter gradients
                grad_gamma_data[f] += grad_out * x_hat;
                grad_beta_data[f] += grad_out;
            }
            
            // Compute input gradients for this sample
            float inv_features = 1.0f / (float)features;
            
            for (long f = 0; f < features; f++) {
                long idx = b * features + f;
                float grad_out = grad_output_data[idx];
                float x_hat = (input_data[idx] - sample_mean) * inv_std;
                
                float grad_x_hat = grad_out * gamma_data[f] - inv_features * grad_sum - inv_features * x_hat * grad_dot_sum;
                grad_input_data[idx] = inv_std * grad_x_hat;
            }
        }
        
        return 0;
    }
}

// Running statistics update
int perform_update_running_stats(
    GPUPtr runningMeanPtr, GPUPtr runningVarPtr,
    GPUPtr batchMeanPtr, GPUPtr batchVarPtr,
    float momentum, long features,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> running_mean_buffer = (__bridge id<MTLBuffer>)runningMeanPtr;
        id<MTLBuffer> running_var_buffer = (__bridge id<MTLBuffer>)runningVarPtr;
        id<MTLBuffer> batch_mean_buffer = (__bridge id<MTLBuffer>)batchMeanPtr;
        id<MTLBuffer> batch_var_buffer = (__bridge id<MTLBuffer>)batchVarPtr;

        if (!running_mean_buffer || !running_var_buffer || !batch_mean_buffer || !batch_var_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for running statistics update.");
            return -1;
        }

        float *running_mean_data = (float*)running_mean_buffer.contents;
        float *running_var_data = (float*)running_var_buffer.contents;
        float *batch_mean_data = (float*)batch_mean_buffer.contents;
        float *batch_var_data = (float*)batch_var_buffer.contents;
        
        // Update running statistics: running = momentum * running + (1 - momentum) * batch
        for (long f = 0; f < features; f++) {
            running_mean_data[f] = momentum * running_mean_data[f] + (1.0f - momentum) * batch_mean_data[f];
            running_var_data[f] = momentum * running_var_data[f] + (1.0f - momentum) * batch_var_data[f];
        }
        
        return 0;
    }
}

// Instance normalization forward pass
int perform_instance_norm_forward(
    GPUPtr inputPtr, long batchSize, long channels, long height, long width,
    GPUPtr gammaPtr, GPUPtr betaPtr,
    float epsilon,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> gamma_buffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> beta_buffer = (__bridge id<MTLBuffer>)betaPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !gamma_buffer || !beta_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for instance normalization forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *gamma_data = (float*)gamma_buffer.contents;
        float *beta_data = (float*)beta_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        long spatial_size = height * width;
        
        // Normalize each channel for each sample independently
        for (long b = 0; b < batchSize; b++) {
            for (long c = 0; c < channels; c++) {
                // Compute mean for this instance (batch, channel)
                float sum = 0.0f;
                for (long s = 0; s < spatial_size; s++) {
                    long idx = b * channels * spatial_size + c * spatial_size + s;
                    sum += input_data[idx];
                }
                float instance_mean = sum / (float)spatial_size;
                
                // Compute variance for this instance
                float sum_squared_diff = 0.0f;
                for (long s = 0; s < spatial_size; s++) {
                    long idx = b * channels * spatial_size + c * spatial_size + s;
                    float diff = input_data[idx] - instance_mean;
                    sum_squared_diff += diff * diff;
                }
                float instance_variance = sum_squared_diff / (float)spatial_size;
                
                // Apply instance normalization
                float inv_std = 1.0f / sqrtf(instance_variance + epsilon);
                for (long s = 0; s < spatial_size; s++) {
                    long idx = b * channels * spatial_size + c * spatial_size + s;
                    float normalized = (input_data[idx] - instance_mean) * inv_std;
                    output_data[idx] = gamma_data[c] * normalized + beta_data[c];
                }
            }
        }
        
        return 0;
    }
}

// Instance normalization backward pass (simplified version)
int perform_instance_norm_backward(
    GPUPtr gradOutputPtr, long batchSize, long channels, long height, long width,
    GPUPtr inputPtr, GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // This is a simplified implementation - full instance norm backward is complex
        set_c_error_message(err, @"Instance normalization backward pass not fully implemented yet.");
        return -1;
    }
}

// Group normalization forward pass
int perform_group_norm_forward(
    GPUPtr inputPtr, long batchSize, long channels, long height, long width,
    long numGroups, GPUPtr gammaPtr, GPUPtr betaPtr,
    float epsilon,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        if (channels % numGroups != 0) {
            set_c_error_message(err, @"Number of channels (%ld) must be divisible by number of groups (%ld).", channels, numGroups);
            return -1;
        }
        
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> gamma_buffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> beta_buffer = (__bridge id<MTLBuffer>)betaPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !gamma_buffer || !beta_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for group normalization forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *gamma_data = (float*)gamma_buffer.contents;
        float *beta_data = (float*)beta_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        long channels_per_group = channels / numGroups;
        long spatial_size = height * width;
        long group_size = channels_per_group * spatial_size;
        
        // Normalize each group for each sample
        for (long b = 0; b < batchSize; b++) {
            for (long g = 0; g < numGroups; g++) {
                long group_start_channel = g * channels_per_group;
                
                // Compute mean for this group
                float sum = 0.0f;
                for (long c = 0; c < channels_per_group; c++) {
                    for (long s = 0; s < spatial_size; s++) {
                        long channel_idx = group_start_channel + c;
                        long idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        sum += input_data[idx];
                    }
                }
                float group_mean = sum / (float)group_size;
                
                // Compute variance for this group
                float sum_squared_diff = 0.0f;
                for (long c = 0; c < channels_per_group; c++) {
                    for (long s = 0; s < spatial_size; s++) {
                        long channel_idx = group_start_channel + c;
                        long idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        float diff = input_data[idx] - group_mean;
                        sum_squared_diff += diff * diff;
                    }
                }
                float group_variance = sum_squared_diff / (float)group_size;
                
                // Apply group normalization
                float inv_std = 1.0f / sqrtf(group_variance + epsilon);
                for (long c = 0; c < channels_per_group; c++) {
                    long channel_idx = group_start_channel + c;
                    for (long s = 0; s < spatial_size; s++) {
                        long idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        float normalized = (input_data[idx] - group_mean) * inv_std;
                        output_data[idx] = gamma_data[channel_idx] * normalized + beta_data[channel_idx];
                    }
                }
            }
        }
        
        return 0;
    }
}

// Group normalization backward pass (simplified version)
int perform_group_norm_backward(
    GPUPtr gradOutputPtr, long batchSize, long channels, long height, long width,
    long numGroups, GPUPtr inputPtr, GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // This is a simplified implementation - full group norm backward is complex
        set_c_error_message(err, @"Group normalization backward pass not fully implemented yet.");
        return -1;
    }
}