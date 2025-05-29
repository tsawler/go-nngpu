// metal_bridge.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_bridge.h" // Include your C header
#include <stdlib.h> // For malloc, free, strdup

// Global pointers for the Metal device and command queue
// These are managed by Objective-C and referenced by Go
DevicePtr _global_mtl_device_ptr = NULL;
CommandQueuePtr _global_mtl_command_queue_ptr = NULL;

// Helper to set CError message
void set_c_error_message(CError *err, NSString *format, ...) {
    if (err) {
        va_list args;
        va_start(args, format);
        NSString *message = [[NSString alloc] initWithFormat:format arguments:args];
        va_end(args);
        err->message = strdup([message UTF8String]); // Duplicate to C string
    }
}

// Function to free the error message memory allocated by Objective-C
void free_c_error_message(char *message) {
    if (message) {
        free(message);
    }
}

// --- GPU Buffer Management ---

int create_gpu_buffer(float *data, long length_bytes, GPUPtr *outGPUPtr, DevicePtr *outDevicePtr, CError *err) {
    @autoreleasepool {
        // Lazily initialize Metal device and command queue
        if (!_global_mtl_device_ptr) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                set_c_error_message(err, @"Could not create Metal device.");
                return -1;
            }
            _global_mtl_device_ptr = (__bridge_retained void*)device; // Retain to manage lifetime

            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                set_c_error_message(err, @"Could not create Metal command queue.");
                // Release device if command queue creation fails
                CFRelease(_global_mtl_device_ptr);
                _global_mtl_device_ptr = NULL;
                return -2;
            }
            _global_mtl_command_queue_ptr = (__bridge_retained void*)commandQueue; // Retain
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)_global_mtl_device_ptr;

        // Create buffer using bytesNoCopy to wrap Go's memory
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:data
                                                          length:length_bytes
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil]; // Go manages the memory
        if (!buffer) {
            set_c_error_message(err, @"Could not create Metal buffer with bytesNoCopy.");
            return -3;
        }

        *outGPUPtr = (__bridge_retained void*)buffer; // Retain the buffer pointer
        *outDevicePtr = _global_mtl_device_ptr; // Pass back the device pointer for future ops

        return 0; // Success
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
        return 0; // Success
    }
}

int release_gpu_buffer(GPUPtr gpuPtr) {
    if (!gpuPtr) return 0; // Nothing to release

    @autoreleasepool {
        // We used __bridge_retained when passing the pointer to Go.
        // Now, we need to release that retain count.
        // This effectively releases the MTLBuffer.
        CFRelease(gpuPtr);
    }
    return 0;
}

// --- MPS Matrix Multiplication ---

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

        // Cast Go's opaque pointers back to Metal objects
        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -2;
        }

        // Create MPSMatrix objects
        // rowBytes is the stride between rows (num_cols * sizeof(float))
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

        // Create the MPSMatrixMultiplication kernel using the correct initializer
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

        // Create a command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            set_c_error_message(err, @"Could not create command buffer.");
            return -5;
        }

        // Encode the operation using the correct method
        [matrixMultiplication encodeToCommandBuffer:commandBuffer
                                         leftMatrix:matrixA
                                        rightMatrix:matrixB
                                       resultMatrix:matrixC];

        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted]; // Synchronous for simplicity, can be async

        if (commandBuffer.error) {
            set_c_error_message(err, @"Metal command buffer error: %@", commandBuffer.error.localizedDescription);
            return -6;
        }

        return 0; // Success
    }
}