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

// ===== FUSED OPERATIONS KERNELS =====

// Fused Conv2D + BatchNorm + ReLU kernel
kernel void fused_conv_bn_relu(device const float* input [[buffer(0)]],
                               device const float* kernel [[buffer(1)]],
                               device const float* gamma [[buffer(2)]],
                               device const float* beta [[buffer(3)]],
                               device const float* bias [[buffer(4)]],
                               device const float* running_mean [[buffer(5)]],
                               device const float* running_var [[buffer(6)]],
                               device float* output [[buffer(7)]],
                               constant long& batch [[buffer(8)]],
                               constant long& in_height [[buffer(9)]],
                               constant long& in_width [[buffer(10)]],
                               constant long& in_channels [[buffer(11)]],
                               constant long& kernel_h [[buffer(12)]],
                               constant long& kernel_w [[buffer(13)]],
                               constant long& out_channels [[buffer(14)]],
                               constant long& stride_h [[buffer(15)]],
                               constant long& stride_w [[buffer(16)]],
                               constant long& pad_h [[buffer(17)]],
                               constant long& pad_w [[buffer(18)]],
                               constant float& epsilon [[buffer(19)]],
                               constant bool& training [[buffer(20)]],
                               constant long& out_height [[buffer(21)]],
                               constant long& out_width [[buffer(22)]],
                               uint3 gid [[thread_position_in_grid]]) {
    
    const uint b = gid.z;
    const uint oc = gid.y;
    const uint out_pos = gid.x;
    
    if (b >= batch || oc >= out_channels || out_pos >= out_height * out_width) return;
    
    const uint oh = out_pos / out_width;
    const uint ow = out_pos % out_width;
    
    // Step 1: Convolution
    float conv_result = 0.0f;
    
    for (long kh = 0; kh < kernel_h; kh++) {
        for (long kw = 0; kw < kernel_w; kw++) {
            for (long ic = 0; ic < in_channels; ic++) {
                long ih = oh * stride_h - pad_h + kh;
                long iw = ow * stride_w - pad_w + kw;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    long input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + ic;
                    long kernel_idx = ((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc;
                    conv_result += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    // Add bias if present
    if (bias) {
        conv_result += bias[oc];
    }
    
    // Step 2: Batch Normalization
    float normalized;
    if (training) {
        // During training, use batch statistics (would need to be computed)
        // For now, using running statistics
        float mean = running_mean ? running_mean[oc] : 0.0f;
        float var = running_var ? running_var[oc] : 1.0f;
        normalized = (conv_result - mean) / sqrt(var + epsilon);
    } else {
        // During inference, use running statistics
        float mean = running_mean ? running_mean[oc] : 0.0f;
        float var = running_var ? running_var[oc] : 1.0f;
        normalized = (conv_result - mean) / sqrt(var + epsilon);
    }
    
    // Apply gamma and beta
    if (gamma && beta) {
        normalized = normalized * gamma[oc] + beta[oc];
    }
    
    // Step 3: ReLU activation
    float activated = max(normalized, 0.0f);
    
    // Write output
    long output_idx = ((b * out_height + oh) * out_width + ow) * out_channels + oc;
    output[output_idx] = activated;
}

// Fused Linear + Activation kernel
kernel void fused_linear_activation(device const float* input [[buffer(0)]],
                                   device const float* weight [[buffer(1)]],
                                   device const float* bias [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant long& batch_size [[buffer(4)]],
                                   constant long& input_size [[buffer(5)]],
                                   constant long& output_size [[buffer(6)]],
                                   constant int& activation_type [[buffer(7)]],
                                   constant float& activation_param [[buffer(8)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    
    const uint batch = gid.y;
    const uint out_idx = gid.x;
    
    if (batch >= batch_size || out_idx >= output_size) return;
    
    // Step 1: Linear transformation
    float sum = 0.0f;
    for (long i = 0; i < input_size; i++) {
        sum += input[batch * input_size + i] * weight[i * output_size + out_idx];
    }
    
    // Add bias if present
    if (bias) {
        sum += bias[out_idx];
    }
    
    // Step 2: Apply activation
    float activated;
    switch (activation_type) {
        case 0: // ReLU
            activated = max(sum, 0.0f);
            break;
        case 1: // LeakyReLU
            activated = sum > 0 ? sum : sum * activation_param;
            break;
        case 2: // Sigmoid
            activated = 1.0f / (1.0f + exp(-sum));
            break;
        case 3: // Tanh
            activated = tanh(sum);
            break;
        case 4: // GELU
            activated = 0.5f * sum * (1.0f + tanh(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
            break;
        default:
            activated = sum; // No activation
    }
    
    output[batch * output_size + out_idx] = activated;
}

// Fused GELU + Dropout kernel
kernel void fused_gelu_dropout(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              device const uint* random_mask [[buffer(2)]],
                              constant long& size [[buffer(3)]],
                              constant float& dropout_rate [[buffer(4)]],
                              constant bool& training [[buffer(5)]],
                              uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    float x = input[index];
    
    // Step 1: GELU activation
    float gelu = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    
    // Step 2: Dropout (only during training)
    if (training && dropout_rate > 0.0f) {
        // Use pre-generated random mask
        float keep_prob = 1.0f - dropout_rate;
        bool keep = random_mask[index] > (uint)(dropout_rate * UINT_MAX);
        output[index] = keep ? (gelu / keep_prob) : 0.0f;
    } else {
        output[index] = gelu;
    }
}

// Fused LayerNorm + Linear kernel
kernel void fused_layer_norm_linear(device const float* input [[buffer(0)]],
                                   device const float* gamma [[buffer(1)]],
                                   device const float* beta [[buffer(2)]],
                                   device const float* weight [[buffer(3)]],
                                   device const float* bias [[buffer(4)]],
                                   device float* output [[buffer(5)]],
                                   device float* mean_out [[buffer(6)]],
                                   device float* var_out [[buffer(7)]],
                                   constant long& batch_size [[buffer(8)]],
                                   constant long& seq_len [[buffer(9)]],
                                   constant long& input_dim [[buffer(10)]],
                                   constant long& output_dim [[buffer(11)]],
                                   constant float& epsilon [[buffer(12)]],
                                   threadgroup float* shared_sum [[threadgroup(0)]],
                                   threadgroup float* shared_sum_sq [[threadgroup(1)]],
                                   uint3 gid [[thread_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
    
    const uint batch = gid.z;
    const uint seq = gid.y;
    const uint out_idx = gid.x;
    
    if (batch >= batch_size || seq >= seq_len || out_idx >= output_dim) return;
    
    // Step 1: Compute LayerNorm statistics (collaborative within threadgroup)
    if (tid == 0) {
        shared_sum[0] = 0.0f;
        shared_sum_sq[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread computes partial sum for layer norm
    if (tid < input_dim) {
        for (uint i = tid; i < input_dim; i += tg_size) {
            float val = input[(batch * seq_len + seq) * input_dim + i];
            atomic_fetch_add_explicit((threadgroup atomic<float>*)&shared_sum[0], val, memory_order_relaxed);
            atomic_fetch_add_explicit((threadgroup atomic<float>*)&shared_sum_sq[0], val * val, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute mean and variance
    float mean = shared_sum[0] / float(input_dim);
    float variance = (shared_sum_sq[0] / float(input_dim)) - (mean * mean);
    
    // Save statistics if needed
    if (tid == 0 && mean_out && var_out) {
        mean_out[batch * seq_len + seq] = mean;
        var_out[batch * seq_len + seq] = variance;
    }
    
    // Step 2: Apply LayerNorm and Linear transformation
    float sum = 0.0f;
    for (long i = 0; i < input_dim; i++) {
        float val = input[(batch * seq_len + seq) * input_dim + i];
        float normalized = (val - mean) / sqrt(variance + epsilon);
        
        // Apply gamma and beta
        if (gamma && beta) {
            normalized = normalized * gamma[i] + beta[i];
        }
        
        // Linear transformation
        sum += normalized * weight[i * output_dim + out_idx];
    }
    
    // Add bias if present
    if (bias) {
        sum += bias[out_idx];
    }
    
    output[(batch * seq_len + seq) * output_dim + out_idx] = sum;
}

// Fused Multi-Head Attention kernel
kernel void fused_attention(device const float* query [[buffer(0)]],
                           device const float* key [[buffer(1)]],
                           device const float* value [[buffer(2)]],
                           device float* output [[buffer(3)]],
                           device float* attention_weights [[buffer(4)]],
                           constant long& batch_size [[buffer(5)]],
                           constant long& seq_len [[buffer(6)]],
                           constant long& num_heads [[buffer(7)]],
                           constant long& head_dim [[buffer(8)]],
                           constant float& scale [[buffer(9)]],
                           constant bool& causal [[buffer(10)]],
                           threadgroup float* shared_scores [[threadgroup(0)]],
                           uint3 gid [[thread_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    
    const uint batch = gid.z;
    const uint head = gid.y;
    const uint query_pos = gid.x;
    
    if (batch >= batch_size || head >= num_heads || query_pos >= seq_len) return;
    
    // Compute attention scores for this query position
    for (uint key_pos = 0; key_pos < seq_len; key_pos++) {
        float score = 0.0f;
        
        // Apply causal mask if needed
        if (causal && key_pos > query_pos) {
            score = -INFINITY;
        } else {
            // Compute dot product between query and key
            for (uint d = 0; d < head_dim; d++) {
                uint q_idx = ((batch * seq_len + query_pos) * num_heads + head) * head_dim + d;
                uint k_idx = ((batch * seq_len + key_pos) * num_heads + head) * head_dim + d;
                score += query[q_idx] * key[k_idx];
            }
            score *= scale;
        }
        
        // Store score for softmax
        shared_scores[key_pos] = score;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute softmax
    float max_score = -INFINITY;
    for (uint i = 0; i < seq_len; i++) {
        max_score = max(max_score, shared_scores[i]);
    }
    
    float sum_exp = 0.0f;
    for (uint i = 0; i < seq_len; i++) {
        shared_scores[i] = exp(shared_scores[i] - max_score);
        sum_exp += shared_scores[i];
    }
    
    for (uint i = 0; i < seq_len; i++) {
        shared_scores[i] /= sum_exp;
    }
    
    // Save attention weights if requested
    if (attention_weights) {
        for (uint i = 0; i < seq_len; i++) {
            uint idx = ((batch * num_heads + head) * seq_len + query_pos) * seq_len + i;
            attention_weights[idx] = shared_scores[i];
        }
    }
    
    // Compute weighted sum of values
    for (uint d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (uint value_pos = 0; value_pos < seq_len; value_pos++) {
            uint v_idx = ((batch * seq_len + value_pos) * num_heads + head) * head_dim + d;
            sum += shared_scores[value_pos] * value[v_idx];
        }
        
        uint out_idx = ((batch * seq_len + query_pos) * num_heads + head) * head_dim + d;
        output[out_idx] = sum;
    }
}