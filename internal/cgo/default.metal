// Add these Metal compute shaders to your default.metal file

#include <metal_stdlib>
using namespace metal;

// Gradient accumulation kernel
kernel void gradient_accumulate(device float* existing_grad [[buffer(0)]],
                               device const float* new_grad [[buffer(1)]],
                               uint index [[thread_position_in_grid]],
                               uint size [[threads_per_grid]]) {
    if (index >= size) return;
    existing_grad[index] += new_grad[index];
}

// Tensor sum of squares kernel for gradient norm calculation
kernel void tensor_sum_squares(device const float* input [[buffer(0)]],
                              device float* result [[buffer(1)]],
                              constant long& size [[buffer(2)]],
                              uint index [[thread_position_in_grid]],
                              uint threads_per_grid [[threads_per_grid]]) {
    
    // Each thread accumulates a portion of the sum
    float local_sum = 0.0;
    
    // Process multiple elements per thread for better performance
    for (uint i = index; i < size; i += threads_per_grid) {
        float val = input[i];
        local_sum += val * val;
    }
    
    // Use threadgroup memory for reduction
    threadgroup float shared_data[256];
    uint local_index = index % 256;
    shared_data[local_index] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform reduction in threadgroup
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_data[local_index] += shared_data[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread in threadgroup writes result
    if (local_index == 0) {
        atomic_fetch_add_explicit((device atomic<float>*)result, shared_data[0], memory_order_relaxed);
    }
}

// Sum along axis kernel
kernel void sum_along_axis(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant int& axis [[buffer(2)]],
                          constant long& input_ndim [[buffer(3)]],
                          constant long* input_shape [[buffer(4)]],
                          constant long& output_ndim [[buffer(5)]],
                          constant long* output_shape [[buffer(6)]],
                          uint index [[thread_position_in_grid]]) {
    
    // Calculate output size
    long output_size = 1;
    for (int i = 0; i < output_ndim; i++) {
        output_size *= output_shape[i];
    }
    
    if (index >= output_size) return;
    
    // Convert linear index to multi-dimensional output coordinates
    long output_coords[8]; // Max 8 dimensions
    long temp_index = index;
    for (int i = output_ndim - 1; i >= 0; i--) {
        output_coords[i] = temp_index % output_shape[i];
        temp_index /= output_shape[i];
    }
    
    // Sum over the specified axis
    float sum = 0.0;
    long axis_size = input_shape[axis];
    
    for (long i = 0; i < axis_size; i++) {
        // Build input coordinates by inserting axis coordinate
        long input_coords[8];
        int input_idx = 0;
        for (int j = 0; j < input_ndim; j++) {
            if (j == axis) {
                input_coords[j] = i;
            } else {
                input_coords[j] = output_coords[input_idx];
                input_idx++;
            }
        }
        
        // Convert multi-dimensional coordinates to linear index
        long input_linear_index = 0;
        long stride = 1;
        for (int j = input_ndim - 1; j >= 0; j--) {
            input_linear_index += input_coords[j] * stride;
            stride *= input_shape[j];
        }
        
        sum += input[input_linear_index];
    }
    
    output[index] = sum;
}

// Dropout forward pass kernel
kernel void dropout_forward(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device float* mask [[buffer(2)]],
                           constant long& size [[buffer(3)]],
                           constant float& probability [[buffer(4)]],
                           constant uint& seed [[buffer(5)]],
                           uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    // Simple linear congruential generator for random numbers
    uint rng_state = seed + index * 1664525u + 1013904223u;
    rng_state = rng_state * 1664525u + 1013904223u;
    
    float random_val = float(rng_state) / float(UINT_MAX);
    
    if (random_val < probability) {
        // Drop this element
        output[index] = 0.0;
        mask[index] = 0.0;
    } else {
        // Keep this element and scale by 1/(1-p)
        float scale = 1.0 / (1.0 - probability);
        output[index] = input[index] * scale;
        mask[index] = scale;
    }
}

// Dropout backward pass kernel
kernel void dropout_backward(device const float* grad_output [[buffer(0)]],
                            device const float* mask [[buffer(1)]],
                            device float* grad_input [[buffer(2)]],
                            constant long& size [[buffer(3)]],
                            constant float& probability [[buffer(4)]],
                            uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    // Apply the same mask that was used in forward pass
    grad_input[index] = grad_output[index] * mask[index];
}

// Tensor fill kernel (for zeroing or setting to constant value)
kernel void tensor_fill(device float* tensor [[buffer(0)]],
                       constant long& size [[buffer(1)]],
                       constant float& value [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    tensor[index] = value;
}

// Advanced gradient operations

// Gradient clipping by norm kernel
kernel void gradient_clip_by_norm(device float* gradients [[buffer(0)]],
                                 constant float& max_norm [[buffer(1)]],
                                 constant float& current_norm [[buffer(2)]],
                                 constant long& size [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    if (current_norm > max_norm) {
        float scale_factor = max_norm / current_norm;
        gradients[index] *= scale_factor;
    }
}

// Gradient scaling kernel (for mixed precision training)
kernel void gradient_scale(device float* gradients [[buffer(0)]],
                          constant float& scale_factor [[buffer(1)]],
                          constant long& size [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    gradients[index] *= scale_factor;
}

// Element-wise gradient operations

// ReLU gradient kernel
kernel void relu_gradient(device const float* grad_output [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         device float* grad_input [[buffer(2)]],
                         constant long& size [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    grad_input[index] = input[index] > 0.0 ? grad_output[index] : 0.0;
}

// Sigmoid gradient kernel
kernel void sigmoid_gradient(device const float* grad_output [[buffer(0)]],
                            device const float* sigmoid_output [[buffer(1)]],
                            device float* grad_input [[buffer(2)]],
                            constant long& size [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    float s = sigmoid_output[index];
    grad_input[index] = grad_output[index] * s * (1.0 - s);
}

// Tanh gradient kernel
kernel void tanh_gradient(device const float* grad_output [[buffer(0)]],
                         device const float* tanh_output [[buffer(1)]],
                         device float* grad_input [[buffer(2)]],
                         constant long& size [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    float t = tanh_output[index];
    grad_input[index] = grad_output[index] * (1.0 - t * t);
}

// Advanced memory operations

// Memory copy with stride kernel
kernel void memory_copy_strided(device const float* source [[buffer(0)]],
                               device float* destination [[buffer(1)]],
                               constant long& source_stride [[buffer(2)]],
                               constant long& dest_stride [[buffer(3)]],
                               constant long& num_elements [[buffer(4)]],
                               uint index [[thread_position_in_grid]]) {
    
    if (index >= num_elements) return;
    destination[index * dest_stride] = source[index * source_stride];
}

// Tensor compression kernel (simple sparsification)
kernel void tensor_sparsify(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device bool* mask [[buffer(2)]],
                           constant float& threshold [[buffer(3)]],
                           constant long& size [[buffer(4)]],
                           uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    float val = input[index];
    if (abs(val) > threshold) {
        output[index] = val;
        mask[index] = true;
    } else {
        output[index] = 0.0;
        mask[index] = false;
    }
}

// Tensor decompression kernel
kernel void tensor_desparsify(device const float* compressed_input [[buffer(0)]],
                             device const bool* mask [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant long& size [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    if (mask[index]) {
        output[index] = compressed_input[index];
    } else {
        output[index] = 0.0;
    }
}

// Memory bandwidth optimization kernels

// Vectorized memory copy (4 floats at a time)
kernel void vectorized_copy(device const float4* source [[buffer(0)]],
                           device float4* destination [[buffer(1)]],
                           constant long& num_float4s [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    
    if (index >= num_float4s) return;
    destination[index] = source[index];
}

// Coalesced memory access pattern for matrix operations
kernel void coalesced_matrix_transpose(device const float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant long& rows [[buffer(2)]],
                                      constant long& cols [[buffer(3)]],
                                      uint2 index [[thread_position_in_grid]]) {
    
    if (index.x >= cols || index.y >= rows) return;
    
    // Read from input[row][col] and write to output[col][row]
    long input_idx = index.y * cols + index.x;
    long output_idx = index.x * rows + index.y;
    
    output[output_idx] = input[input_idx];
}

// Advanced reduction operations

// Parallel reduction for finding maximum value
kernel void parallel_max_reduction(device const float* input [[buffer(0)]],
                                  device float* result [[buffer(1)]],
                                  constant long& size [[buffer(2)]],
                                  uint index [[thread_position_in_grid]],
                                  uint threads_per_grid [[threads_per_grid]]) {
    
    threadgroup float shared_data[256];
    uint local_index = index % 256;
    
    // Each thread finds max in its portion
    float local_max = -INFINITY;
    for (uint i = index; i < size; i += threads_per_grid) {
        local_max = max(local_max, input[i]);
    }
    
    shared_data[local_index] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find global max
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_data[local_index] = max(shared_data[local_index], shared_data[local_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (local_index == 0) {
        atomic_fetch_max_explicit((device atomic<float>*)result, shared_data[0], memory_order_relaxed);
    }
}

// Parallel reduction for finding minimum value
kernel void parallel_min_reduction(device const float* input [[buffer(0)]],
                                  device float* result [[buffer(1)]],
                                  constant long& size [[buffer(2)]],
                                  uint index [[thread_position_in_grid]],
                                  uint threads_per_grid [[threads_per_grid]]) {
    
    threadgroup float shared_data[256];
    uint local_index = index % 256;
    
    // Each thread finds min in its portion
    float local_min = INFINITY;
    for (uint i = index; i < size; i += threads_per_grid) {
        local_min = min(local_min, input[i]);
    }
    
    shared_data[local_index] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find global min
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_data[local_index] = min(shared_data[local_index], shared_data[local_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (local_index == 0) {
        atomic_fetch_min_explicit((device atomic<float>*)result, shared_data[0], memory_order_relaxed);
    }
}