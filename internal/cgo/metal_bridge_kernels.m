// metal_bridge_kernels.m - Optimized Kernel Functions Implementation
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Suppress deprecation warnings for CLAPACK
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <Accelerate/Accelerate.h>

#include "metal_bridge.h"
#include <stdlib.h>

// External declarations for global variables from main metal_bridge.m
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External declarations for helper functions from main metal_bridge.m
extern void set_c_error_message(CError *err, NSString *format, ...);
extern void free_c_error_message(char *message);

// Embedded Metal shader source for optimized kernels
static NSString* const METAL_SHADER_SOURCE = @R"(
#include <metal_stdlib>
using namespace metal;

// Optimized GEMM kernel with tiling and shared memory
kernel void optimized_gemm(device const float* A [[buffer(0)]],
                          device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]],
                          constant long& M [[buffer(3)]],
                          constant long& N [[buffer(4)]],
                          constant long& K [[buffer(5)]],
                          constant float& alpha [[buffer(6)]],
                          constant float& beta [[buffer(7)]],
                          threadgroup float* tileA [[threadgroup(0)]],
                          threadgroup float* tileB [[threadgroup(1)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 tg_size [[threads_per_threadgroup]]) {
    
    const uint row = gid.y;
    const uint col = gid.x;
    const uint tileSize = tg_size.x; // Assuming square threadgroup
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Iterate over tiles
    for (uint tile = 0; tile < (K + tileSize - 1) / tileSize; tile++) {
        // Load tile A into threadgroup memory
        uint aRow = row;
        uint aCol = tile * tileSize + tid.x;
        if (aRow < M && aCol < K) {
            tileA[tid.y * tileSize + tid.x] = A[aRow * K + aCol];
        } else {
            tileA[tid.y * tileSize + tid.x] = 0.0f;
        }
        
        // Load tile B into threadgroup memory
        uint bRow = tile * tileSize + tid.y;
        uint bCol = col;
        if (bRow < K && bCol < N) {
            tileB[tid.y * tileSize + tid.x] = B[bRow * N + bCol];
        } else {
            tileB[tid.y * tileSize + tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < tileSize; k++) {
            sum += tileA[tid.y * tileSize + k] * tileB[k * tileSize + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

// Batch matrix multiplication kernel
kernel void batch_matmul_optimized(device const float* A [[buffer(0)]],
                                  device const float* B [[buffer(1)]],
                                  device float* C [[buffer(2)]],
                                  constant long& batchSize [[buffer(3)]],
                                  constant long& M [[buffer(4)]],
                                  constant long& N [[buffer(5)]],
                                  constant long& K [[buffer(6)]],
                                  uint3 gid [[thread_position_in_grid]]) {
    
    const uint batch = gid.z;
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (batch >= batchSize || row >= M || col >= N) return;
    
    const uint batchOffsetA = batch * M * K;
    const uint batchOffsetB = batch * K * N;
    const uint batchOffsetC = batch * M * N;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[batchOffsetA + row * K + k] * B[batchOffsetB + k * N + col];
    }
    
    C[batchOffsetC + row * N + col] = sum;
}

// Optimized softmax kernel with numerical stability
kernel void softmax_optimized(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant long& batchSize [[buffer(2)]],
                             constant long& numClasses [[buffer(3)]],
                             threadgroup float* shared_max [[threadgroup(0)]],
                             threadgroup float* shared_sum [[threadgroup(1)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 tg_size [[threads_per_threadgroup]]) {
    
    const uint batch = gid.y;
    const uint class_idx = gid.x;
    
    if (batch >= batchSize) return;
    
    const uint inputOffset = batch * numClasses;
    
    // Phase 1: Find maximum for numerical stability (simple reduction)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x == 0 && tid.y == 0) {
        shared_max[0] = -INFINITY;
        for (uint i = 0; i < numClasses; i++) {
            shared_max[0] = max(shared_max[0], input[inputOffset + i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float max_val = shared_max[0];
    
    // Phase 2: Compute sum of exponentials (simple reduction)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x == 0 && tid.y == 0) {
        shared_sum[0] = 0.0f;
        for (uint i = 0; i < numClasses; i++) {
            shared_sum[0] += exp(input[inputOffset + i] - max_val);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float sum_exp = shared_sum[0];
    
    // Phase 3: Compute final softmax values
    if (class_idx < numClasses) {
        float exp_val = exp(input[inputOffset + class_idx] - max_val);
        output[inputOffset + class_idx] = exp_val / sum_exp;
    }
}

// Optimized 1x1 convolution kernel
kernel void conv1x1_optimized(device const float* input [[buffer(0)]],
                             device const float* weight [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant long& batch [[buffer(3)]],
                             constant long& height [[buffer(4)]],
                             constant long& width [[buffer(5)]],
                             constant long& inChannels [[buffer(6)]],
                             constant long& outChannels [[buffer(7)]],
                             uint3 gid [[thread_position_in_grid]]) {
    
    const uint b = gid.z;
    const uint y = gid.y;
    const uint x = gid.x;
    
    if (b >= batch || y >= height || x >= width) return;
    
    const uint inputOffset = ((b * height + y) * width + x) * inChannels;
    const uint outputOffset = ((b * height + y) * width + x) * outChannels;
    
    // Compute 1x1 convolution for all output channels
    for (uint oc = 0; oc < outChannels; oc++) {
        float sum = 0.0f;
        for (uint ic = 0; ic < inChannels; ic++) {
            sum += input[inputOffset + ic] * weight[oc * inChannels + ic];
        }
        output[outputOffset + oc] = sum;
    }
}

// Depthwise convolution kernel
kernel void depthwise_conv_optimized(device const float* input [[buffer(0)]],
                                   device const float* weights [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant long& batch [[buffer(3)]],
                                   constant long& inHeight [[buffer(4)]],
                                   constant long& inWidth [[buffer(5)]],
                                   constant long& channels [[buffer(6)]],
                                   constant long& kernelH [[buffer(7)]],
                                   constant long& kernelW [[buffer(8)]],
                                   constant long& strideH [[buffer(9)]],
                                   constant long& strideW [[buffer(10)]],
                                   constant long& padH [[buffer(11)]],
                                   constant long& padW [[buffer(12)]],
                                   uint3 gid [[thread_position_in_grid]]) {
    
    const uint b = gid.z;
    const uint oy = gid.y;
    const uint ox = gid.x;
    
    const uint outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
    const uint outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;
    
    if (b >= batch || oy >= outHeight || ox >= outWidth) return;
    
    // Depthwise convolution: each input channel has its own kernel
    for (uint c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (uint ky = 0; ky < kernelH; ky++) {
            for (uint kx = 0; kx < kernelW; kx++) {
                int iy = (int)(oy * strideH + ky) - (int)padH;
                int ix = (int)(ox * strideW + kx) - (int)padW;
                
                if (iy >= 0 && iy < inHeight && ix >= 0 && ix < inWidth) {
                    uint inputIdx = ((b * inHeight + iy) * inWidth + ix) * channels + c;
                    uint kernelIdx = (c * kernelH + ky) * kernelW + kx;
                    sum += input[inputIdx] * weights[kernelIdx];
                }
            }
        }
        
        uint outputIdx = ((b * outHeight + oy) * outWidth + ox) * channels + c;
        output[outputIdx] = sum;
    }
}

// Elementwise binary operations kernel
kernel void elementwise_binary_op_optimized(device const float* a [[buffer(0)]],
                                           device const float* b [[buffer(1)]],
                                           device float* output [[buffer(2)]],
                                           constant int& opType [[buffer(3)]],
                                           constant long& size [[buffer(4)]],
                                           uint index [[thread_position_in_grid]]) {
    
    if (index >= size) return;
    
    float valA = a[index];
    float valB = b[index];
    float result = 0.0f;
    
    switch (opType) {
        case 0: // Add
            result = valA + valB;
            break;
        case 1: // Subtract
            result = valA - valB;
            break;
        case 2: // Multiply
            result = valA * valB;
            break;
        case 3: // Divide
            result = valA / valB;
            break;
        default:
            result = valA;
    }
    
    output[index] = result;
}

// Reduction kernel
kernel void reduce_optimized(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant long& size [[buffer(2)]],
                           constant int& opType [[buffer(3)]],
                           threadgroup float* shared_data [[threadgroup(0)]],
                           uint index [[thread_position_in_grid]],
                           uint local_index [[thread_index_in_threadgroup]],
                           uint group_size [[threads_per_threadgroup]]) {
    
    // Load data into threadgroup memory
    float value = 0.0f;
    if (index < size) {
        value = input[index];
    }
    
    shared_data[local_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform reduction in threadgroup
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            float other = shared_data[local_index + stride];
            switch (opType) {
                case 0: // Sum
                    shared_data[local_index] += other;
                    break;
                case 1: // Max
                    shared_data[local_index] = max(shared_data[local_index], other);
                    break;
                case 2: // Min
                    shared_data[local_index] = min(shared_data[local_index], other);
                    break;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (local_index == 0) {
        output[index / group_size] = shared_data[0];
    }
}

// Layer normalization kernel
kernel void layer_norm_optimized(device const float* input [[buffer(0)]],
                                device const float* gamma [[buffer(1)]],
                                device const float* beta [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                device float* meanOut [[buffer(4)]],
                                device float* varOut [[buffer(5)]],
                                constant long& batchSize [[buffer(6)]],
                                constant long& featureSize [[buffer(7)]],
                                constant float& epsilon [[buffer(8)]],
                                threadgroup float* shared_sum [[threadgroup(0)]],
                                threadgroup float* shared_sum_sq [[threadgroup(1)]],
                                uint2 gid [[thread_position_in_grid]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group_size [[threads_per_threadgroup]]) {
    
    const uint batch = gid.y;
    const uint feature = gid.x;
    
    if (batch >= batchSize) return;
    
    const uint inputOffset = batch * featureSize;
    
    // Compute mean and variance
    if (tid.x == 0 && tid.y == 0) {
        shared_sum[0] = 0.0f;
        shared_sum_sq[0] = 0.0f;
        
        for (uint i = 0; i < featureSize; i++) {
            float val = input[inputOffset + i];
            shared_sum[0] += val;
            shared_sum_sq[0] += val * val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float mean = shared_sum[0] / float(featureSize);
    float variance = (shared_sum_sq[0] / float(featureSize)) - (mean * mean);
    
    // Save statistics
    if (tid.x == 0 && tid.y == 0 && meanOut && varOut) {
        meanOut[batch] = mean;
        varOut[batch] = variance;
    }
    
    // Normalize and apply scale/shift
    if (feature < featureSize) {
        float val = input[inputOffset + feature];
        float normalized = (val - mean) / sqrt(variance + epsilon);
        
        if (gamma && beta) {
            output[inputOffset + feature] = normalized * gamma[feature] + beta[feature];
        } else {
            output[inputOffset + feature] = normalized;
        }
    }
}
)";

// Global library cache to avoid recompilation
static id<MTLLibrary> _cached_library = nil;
static dispatch_once_t _library_once_token;

// Helper function to create compute pipeline state from embedded source
id<MTLComputePipelineState> createPipelineState(id<MTLDevice> device, NSString* kernelName, CError* err) {
    NSError* error = nil;
    
    // Create library from source (cached)
    dispatch_once(&_library_once_token, ^{
        NSError* compilationError = nil;
        _cached_library = [device newLibraryWithSource:METAL_SHADER_SOURCE options:nil error:&compilationError];
        if (!_cached_library && compilationError) {
            NSLog(@"Metal compilation error: %@", compilationError.localizedDescription);
        }
    });
    
    if (!_cached_library) {
        set_c_error_message(err, @"Could not compile Metal library from source");
        return nil;
    }
    
    // Get the kernel function
    id<MTLFunction> kernelFunction = [_cached_library newFunctionWithName:kernelName];
    if (!kernelFunction) {
        set_c_error_message(err, @"Could not find kernel function: %@", kernelName);
        return nil;
    }
    
    // Create compute pipeline state
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!pipelineState) {
        set_c_error_message(err, @"Could not create pipeline state for %@: %@", kernelName, error.localizedDescription);
        return nil;
    }
    
    return pipelineState;
}

// Optimized GEMM with tiling and shared memory
int perform_optimized_gemm(
    GPUPtr aPtr, GPUPtr bPtr, GPUPtr cPtr,
    long M, long N, long K,
    float alpha, float beta,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)aPtr;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)bPtr;
        id<MTLBuffer> bufferC = (__bridge id<MTLBuffer>)cPtr;
        
        if (!bufferA || !bufferB || !bufferC) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        // Create pipeline state
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"optimized_gemm", err);
        if (!pipelineState) {
            return -3;
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(long) atIndex:3];
        [encoder setBytes:&N length:sizeof(long) atIndex:4];
        [encoder setBytes:&K length:sizeof(long) atIndex:5];
        [encoder setBytes:&alpha length:sizeof(float) atIndex:6];
        [encoder setBytes:&beta length:sizeof(float) atIndex:7];
        
        // Set threadgroup memory for tiling
        NSUInteger tileSize = 16;
        NSUInteger threadgroupMemoryA = tileSize * tileSize * sizeof(float);
        NSUInteger threadgroupMemoryB = tileSize * tileSize * sizeof(float);
        [encoder setThreadgroupMemoryLength:threadgroupMemoryA atIndex:0];
        [encoder setThreadgroupMemoryLength:threadgroupMemoryB atIndex:1];
        
        // Configure thread execution
        MTLSize threadsPerThreadgroup = MTLSizeMake(tileSize, tileSize, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize, 1);
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"GEMM kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

// Optimized batch matrix multiplication
int perform_batch_matmul_optimized(
    GPUPtr aPtr, GPUPtr bPtr, GPUPtr cPtr,
    long batchSize, long M, long N, long K,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)aPtr;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)bPtr;
        id<MTLBuffer> bufferC = (__bridge id<MTLBuffer>)cPtr;
        
        if (!bufferA || !bufferB || !bufferC) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"batch_matmul_optimized", err);
        if (!pipelineState) {
            return -3;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBytes:&batchSize length:sizeof(long) atIndex:3];
        [encoder setBytes:&M length:sizeof(long) atIndex:4];
        [encoder setBytes:&N length:sizeof(long) atIndex:5];
        [encoder setBytes:&K length:sizeof(long) atIndex:6];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((N + 15) / 16, (M + 15) / 16, batchSize);
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Batch matmul kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

// Optimized 1x1 convolution
int perform_conv1x1_optimized(
    GPUPtr inputPtr, GPUPtr weightPtr, GPUPtr outputPtr,
    long batch, long height, long width,
    long inChannels, long outChannels,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)weightPtr;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputPtr;
        
        if (!inputBuffer || !weightBuffer || !outputBuffer) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"conv1x1_optimized", err);
        if (!pipelineState) {
            return -3;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&batch length:sizeof(long) atIndex:3];
        [encoder setBytes:&height length:sizeof(long) atIndex:4];
        [encoder setBytes:&width length:sizeof(long) atIndex:5];
        [encoder setBytes:&inChannels length:sizeof(long) atIndex:6];
        [encoder setBytes:&outChannels length:sizeof(long) atIndex:7];
        
        long totalPixels = height * width;
        MTLSize threadsPerThreadgroup = MTLSizeMake(32, 8, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((totalPixels + 31) / 32, (outChannels + 7) / 8, batch);
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Conv1x1 kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

// Optimized depthwise convolution
int perform_depthwise_conv_optimized(
    GPUPtr inputPtr, GPUPtr kernelPtr, GPUPtr outputPtr,
    long batch, long inHeight, long inWidth, long channels,
    long kernelH, long kernelW,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> kernelBuffer = (__bridge id<MTLBuffer>)kernelPtr;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputPtr;
        
        if (!inputBuffer || !kernelBuffer || !outputBuffer) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"depthwise_conv_optimized", err);
        if (!pipelineState) {
            return -3;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:kernelBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&batch length:sizeof(long) atIndex:3];
        [encoder setBytes:&inHeight length:sizeof(long) atIndex:4];
        [encoder setBytes:&inWidth length:sizeof(long) atIndex:5];
        [encoder setBytes:&channels length:sizeof(long) atIndex:6];
        [encoder setBytes:&kernelH length:sizeof(long) atIndex:7];
        [encoder setBytes:&kernelW length:sizeof(long) atIndex:8];
        [encoder setBytes:&strideH length:sizeof(long) atIndex:9];
        [encoder setBytes:&strideW length:sizeof(long) atIndex:10];
        [encoder setBytes:&padH length:sizeof(long) atIndex:11];
        [encoder setBytes:&padW length:sizeof(long) atIndex:12];
        
        long outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
        long outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;
        long totalOutputPositions = outHeight * outWidth;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(32, 8, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((totalOutputPositions + 31) / 32, (channels + 7) / 8, batch);
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Depthwise conv kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

// Optimized elementwise binary operations with broadcasting
int perform_elementwise_binary_op_optimized(
    GPUPtr aPtr, GPUPtr bPtr, GPUPtr outputPtr,
    int opType, long size,
    long aStride, long bStride,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)aPtr;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)bPtr;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputPtr;
        
        if (!bufferA || !bufferB || !outputBuffer) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"elementwise_binary_op", err);
        if (!pipelineState) {
            return -3;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&opType length:sizeof(int) atIndex:3];
        [encoder setBytes:&size length:sizeof(long) atIndex:4];
        [encoder setBytes:&aStride length:sizeof(long) atIndex:5];
        [encoder setBytes:&bStride length:sizeof(long) atIndex:6];
        
        NSUInteger threadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup;
        if (threadsPerThreadgroup > size) {
            threadsPerThreadgroup = size;
        }
        
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake((size + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Elementwise binary op kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

// Optimized reduction with threadgroup reduction
int perform_reduce_optimized(
    GPUPtr inputPtr, GPUPtr outputPtr,
    long size, int opType,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputPtr;
        
        if (!inputBuffer || !outputBuffer) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        // Create partial sums buffer for atomic operations
        id<MTLBuffer> partialSumsBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        if (!partialSumsBuffer) {
            set_c_error_message(err, @"Failed to create partial sums buffer");
            return -3;
        }
        
        // Initialize partial sums
        float* partialSumsPtr = (float*)partialSumsBuffer.contents;
        if (opType == 0) { // Sum
            *partialSumsPtr = 0.0f;
        } else if (opType == 1) { // Max
            *partialSumsPtr = -INFINITY;
        } else if (opType == 2) { // Min
            *partialSumsPtr = INFINITY;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"reduce_optimized", err);
        if (!pipelineState) {
            return -4;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:partialSumsBuffer offset:0 atIndex:2];
        [encoder setBytes:&size length:sizeof(long) atIndex:3];
        [encoder setBytes:&opType length:sizeof(int) atIndex:4];
        
        NSUInteger threadsPerThreadgroup = 256; // Optimal for reduction
        NSUInteger threadgroupMemory = threadsPerThreadgroup * sizeof(float);
        [encoder setThreadgroupMemoryLength:threadgroupMemory atIndex:0];
        
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake((size + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Reduce kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -5;
        }
        
        // Copy result from partial sums to output
        float* outputPtr_cpu = (float*)outputBuffer.contents;
        *outputPtr_cpu = *partialSumsPtr;
        
        return 0;
    }
}

// Optimized softmax with numerical stability
int perform_softmax_optimized(
    GPUPtr inputPtr, GPUPtr outputPtr,
    long batchSize, long numClasses,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputPtr;
        
        if (!inputBuffer || !outputBuffer) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"softmax_optimized", err);
        if (!pipelineState) {
            return -3;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBytes:&batchSize length:sizeof(long) atIndex:2];
        [encoder setBytes:&numClasses length:sizeof(long) atIndex:3];
        
        NSUInteger threadsPerThreadgroup = 256;
        NSUInteger threadgroupMemoryMax = threadsPerThreadgroup * sizeof(float);
        NSUInteger threadgroupMemorySum = threadsPerThreadgroup * sizeof(float);
        [encoder setThreadgroupMemoryLength:threadgroupMemoryMax atIndex:0];
        [encoder setThreadgroupMemoryLength:threadgroupMemorySum atIndex:1];
        
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(1, batchSize, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Softmax kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

// Optimized layer normalization
int perform_layer_norm_optimized(
    GPUPtr inputPtr, GPUPtr gammaPtr, GPUPtr betaPtr,
    GPUPtr outputPtr, GPUPtr meanOutPtr, GPUPtr varOutPtr,
    long batchSize, long featureSize, float epsilon,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized");
            return -1;
        }
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> gammaBuffer = (__bridge id<MTLBuffer>)gammaPtr;
        id<MTLBuffer> betaBuffer = (__bridge id<MTLBuffer>)betaPtr;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputPtr;
        id<MTLBuffer> meanOutBuffer = (__bridge id<MTLBuffer>)meanOutPtr;
        id<MTLBuffer> varOutBuffer = (__bridge id<MTLBuffer>)varOutPtr;
        
        if (!inputBuffer || !gammaBuffer || !betaBuffer || !outputBuffer) {
            set_c_error_message(err, @"Invalid buffer pointers");
            return -2;
        }
        
        id<MTLComputePipelineState> pipelineState = createPipelineState(device, @"layer_norm_optimized", err);
        if (!pipelineState) {
            return -3;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:gammaBuffer offset:0 atIndex:1];
        [encoder setBuffer:betaBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        
        // Optional mean and variance output buffers
        if (meanOutBuffer) {
            [encoder setBuffer:meanOutBuffer offset:0 atIndex:4];
        } else {
            [encoder setBuffer:inputBuffer offset:0 atIndex:4]; // Dummy buffer
        }
        
        if (varOutBuffer) {
            [encoder setBuffer:varOutBuffer offset:0 atIndex:5];
        } else {
            [encoder setBuffer:inputBuffer offset:0 atIndex:5]; // Dummy buffer
        }
        
        [encoder setBytes:&batchSize length:sizeof(long) atIndex:6];
        [encoder setBytes:&featureSize length:sizeof(long) atIndex:7];
        [encoder setBytes:&epsilon length:sizeof(float) atIndex:8];
        
        NSUInteger threadsPerThreadgroup = 256;
        NSUInteger threadgroupMemorySum = threadsPerThreadgroup * sizeof(float);
        NSUInteger threadgroupMemorySumSq = threadsPerThreadgroup * sizeof(float);
        [encoder setThreadgroupMemoryLength:threadgroupMemorySum atIndex:0];
        [encoder setThreadgroupMemoryLength:threadgroupMemorySumSq atIndex:1];
        
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(1, batchSize, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            set_c_error_message(err, @"Layer norm kernel execution failed: %@", commandBuffer.error.localizedDescription);
            return -4;
        }
        
        return 0;
    }
}

#pragma clang diagnostic pop