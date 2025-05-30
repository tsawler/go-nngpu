// metal_bridge.m - Fixed version with correct MPS APIs
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_bridge.h"
#include <stdlib.h>

// Global pointers for the Metal device and command queue
DevicePtr _global_mtl_device_ptr = NULL;
CommandQueuePtr _global_mtl_command_queue_ptr = NULL;

// Helper to set CError message
void set_c_error_message(CError *err, NSString *format, ...) {
    if (err) {
        va_list args;
        va_start(args, format);
        NSString *message = [[NSString alloc] initWithFormat:format arguments:args];
        va_end(args);
        err->message = strdup([message UTF8String]);
    }
}

void free_c_error_message(char *message) {
    if (message) {
        free(message);
    }
}

// --- GPU Buffer Management (existing code) ---

int create_gpu_buffer(float *data, long length_bytes, GPUPtr *outGPUPtr, DevicePtr *outDevicePtr, CError *err) {
    @autoreleasepool {
        if (!_global_mtl_device_ptr) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                set_c_error_message(err, @"Could not create Metal device.");
                return -1;
            }
            _global_mtl_device_ptr = (__bridge_retained void*)device;

            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                set_c_error_message(err, @"Could not create Metal command queue.");
                CFRelease(_global_mtl_device_ptr);
                _global_mtl_device_ptr = NULL;
                return -2;
            }
            _global_mtl_command_queue_ptr = (__bridge_retained void*)commandQueue;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)_global_mtl_device_ptr;

        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:data
                                                          length:length_bytes
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        if (!buffer) {
            set_c_error_message(err, @"Could not create Metal buffer with bytesNoCopy.");
            return -3;
        }

        *outGPUPtr = (__bridge_retained void*)buffer;
        *outDevicePtr = _global_mtl_device_ptr;

        return 0;
    }
}

int retrieve_gpu_buffer_data(GPUPtr gpuPtr, float *data, long length_bytes, CError *err) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)gpuPtr;
        if (!buffer) {
            set_c_error_message(err, @"Invalid GPU buffer pointer.");
            return -1;
        }
        if (buffer.length < length_bytes) {
            set_c_error_message(err, @"Buffer length mismatch during retrieval.");
            return -2;
        }

        memcpy(data, buffer.contents, length_bytes);
        return 0;
    }
}

int release_gpu_buffer(GPUPtr gpuPtr) {
    if (!gpuPtr) return 0;

    @autoreleasepool {
        CFRelease(gpuPtr);
    }
    return 0;
}

// --- Matrix Multiplication (existing) ---

int perform_mps_matrix_multiplication(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;

        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized.");
            return -1;
        }

        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -2;
        }

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                           columns:aCols
                                                                          rowBytes:aCols * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:A_buffer descriptor:descA];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:bRows
                                                                           columns:bCols
                                                                          rowBytes:bCols * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:B_buffer descriptor:descB];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:resultRows
                                                                           columns:resultCols
                                                                          rowBytes:resultCols * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:C_buffer descriptor:descC];

        if (!matrixA || !matrixB || !matrixC) {
            set_c_error_message(err, @"Failed to create MPSMatrix objects.");
            return -3;
        }

        MPSMatrixMultiplication *matrixMultiplication = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                                      transposeLeft:NO
                                                                                     transposeRight:NO
                                                                                         resultRows:resultRows
                                                                                      resultColumns:resultCols
                                                                                      interiorColumns:aCols
                                                                                              alpha:1.0
                                                                                               beta:0.0];

        if (!matrixMultiplication) {
            set_c_error_message(err, @"Failed to create MPSMatrixMultiplication kernel.");
            return -4;
        }

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            set_c_error_message(err, @"Could not create command buffer.");
            return -5;
        }

        [matrixMultiplication encodeToCommandBuffer:commandBuffer
                                         leftMatrix:matrixA
                                        rightMatrix:matrixB
                                       resultMatrix:matrixC];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error) {
            set_c_error_message(err, @"Metal command buffer error: %@", commandBuffer.error.localizedDescription);
            return -6;
        }

        return 0;
    }
}

// --- Matrix Transpose (existing) ---

int perform_mps_matrix_transpose(
    GPUPtr inputMatrixPtr, long inputRows, long inputCols,
    GPUPtr outputMatrixPtr, long outputRows, long outputCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;

        if (!device || !commandQueue) {
            set_c_error_message(err, @"Metal device or command queue not initialized.");
            return -1;
        }

        // Verify transpose dimensions
        if (inputRows != outputCols || inputCols != outputRows) {
            set_c_error_message(err, @"Invalid transpose dimensions.");
            return -2;
        }

        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputMatrixPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        // Simple CPU-based transpose for now (can be optimized with Metal compute shader later)
        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;

        for (long i = 0; i < inputRows; i++) {
            for (long j = 0; j < inputCols; j++) {
                output_data[j * outputCols + i] = input_data[i * inputCols + j];
            }
        }

        return 0;
    }
}

// --- Element-wise Operations (Phase 2) - Using CPU implementation for simplicity ---

int perform_mps_matrix_add(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify dimensions match
        if (aRows != bRows || aCols != bCols || aRows != resultRows || aCols != resultCols) {
            set_c_error_message(err, @"Matrices must have the same dimensions for element-wise addition.");
            return -2;
        }

        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        // Use direct CPU-based addition for simplicity and reliability
        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        long total_elements = aRows * aCols;
        
        // Perform element-wise addition: result[i] = a[i] + b[i]
        for (long i = 0; i < total_elements; i++) {
            result_data[i] = a_data[i] + b_data[i];
        }

        return 0;
    }
}

int perform_mps_matrix_subtract(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify dimensions match
        if (aRows != bRows || aCols != bCols || aRows != resultRows || aCols != resultCols) {
            set_c_error_message(err, @"Matrices must have the same dimensions for element-wise subtraction.");
            return -2;
        }

        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        // Use direct CPU-based subtraction: A - B
        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        long total_elements = aRows * aCols;
        
        // Perform element-wise subtraction: result[i] = a[i] - b[i]
        for (long i = 0; i < total_elements; i++) {
            result_data[i] = a_data[i] - b_data[i];
        }

        return 0;
    }
}

int perform_mps_matrix_element_multiply(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify dimensions match
        if (aRows != bRows || aCols != bCols || aRows != resultRows || aCols != resultCols) {
            set_c_error_message(err, @"Matrices must have the same dimensions for element-wise multiplication.");
            return -2;
        }

        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        // Use direct CPU-based element-wise multiplication (Hadamard product)
        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        long total_elements = aRows * aCols;
        
        // Perform element-wise multiplication: result[i] = a[i] * b[i]
        for (long i = 0; i < total_elements; i++) {
            result_data[i] = a_data[i] * b_data[i];
        }

        return 0;
    }
}

int perform_mps_matrix_element_divide(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify dimensions match
        if (aRows != bRows || aCols != bCols || aRows != resultRows || aCols != resultCols) {
            set_c_error_message(err, @"Matrices must have the same dimensions for element-wise division.");
            return -2;
        }

        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        // Use simple CPU-based element-wise division
        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        long total_elements = aRows * aCols;
        
        // Perform element-wise division: result[i] = a[i] / b[i]
        for (long i = 0; i < total_elements; i++) {
            if (b_data[i] == 0.0f) {
                set_c_error_message(err, @"Division by zero encountered at element %ld.", i);
                return -4;
            }
            result_data[i] = a_data[i] / b_data[i];
        }

        return 0;
    }
}

// --- Scalar Operations ---

int perform_mps_matrix_scalar_add(
    GPUPtr inputMatrixPtr, long rows, long cols,
    float scalar,
    GPUPtr resultMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!input_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *result_data = (float*)result_buffer.contents;
        
        long total_elements = rows * cols;
        
        // Perform scalar addition: result[i] = input[i] + scalar
        for (long i = 0; i < total_elements; i++) {
            result_data[i] = input_data[i] + scalar;
        }

        return 0;
    }
}

int perform_mps_matrix_scalar_multiply(
    GPUPtr inputMatrixPtr, long rows, long cols,
    float scalar,
    GPUPtr resultMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!input_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *result_data = (float*)result_buffer.contents;
        
        long total_elements = rows * cols;
        
        // Perform scalar multiplication: result[i] = input[i] * scalar
        for (long i = 0; i < total_elements; i++) {
            result_data[i] = input_data[i] * scalar;
        }

        return 0;
    }
}