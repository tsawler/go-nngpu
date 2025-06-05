// metal_bridge_matrix.m - Basic Matrix Operations
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

// External declarations for global variables (defined in metal_bridge_common.m)
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External declarations for common functions (defined in metal_bridge_common.m)
extern void set_c_error_message(CError *err, NSString *format, ...);

// --- Matrix Operations ---

int perform_mps_matrix_multiplication(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        
        if (!device) {
            set_c_error_message(err, @"Metal device not initialized.");
            return -1;
        }
        
        // Cache the device globally for consistent access
        if (!_global_mtl_device_ptr) {
            _global_mtl_device_ptr = mtlDevicePtr;
        }
        
        // Use global command queue for better performance
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        // If global queue is not set, create one and cache it
        if (!commandQueue) {
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                set_c_error_message(err, @"Failed to create command queue from device.");
                return -1;
            }
            commandQueue.label = @"go-nngpu.matmul";
            _global_mtl_command_queue_ptr = (__bridge void*)commandQueue;
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

int perform_mps_matrix_transpose(
    GPUPtr inputMatrixPtr, long inputRows, long inputCols,
    GPUPtr outputMatrixPtr, long outputRows, long outputCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        
        if (!device) {
            set_c_error_message(err, @"Metal device not initialized.");
            return -1;
        }
        
        // Note: This function currently uses CPU implementation
        // The command queue is not needed for CPU-based transpose

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

        // CPU-based transpose implementation
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
        
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        
        if (!device) {
            set_c_error_message(err, @"Metal device not initialized.");
            return -1;
        }
        
        // Cache the device globally for consistent access
        if (!_global_mtl_device_ptr) {
            _global_mtl_device_ptr = mtlDevicePtr;
        }
        
        // Use global command queue for better performance
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        // If global queue is not set, create one and cache it
        if (!commandQueue) {
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                set_c_error_message(err, @"Failed to create command queue from device.");
                return -1;
            }
            commandQueue.label = @"go-nngpu.shared";
            _global_mtl_command_queue_ptr = (__bridge void*)commandQueue;
        }

        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        // For now, use optimized CPU implementation until Metal compute shaders are ready
        long total_elements = aRows * aCols;
        
        // Optimized CPU addition with potential for vectorization
        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        // Use restrict keyword for compiler optimization hints
        float* restrict a_ptr = a_data;
        float* restrict b_ptr = b_data;
        float* restrict result_ptr = result_data;
        
        // Unrolled loop for better performance
        long i = 0;
        for (; i < total_elements - 3; i += 4) {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
            result_ptr[i+1] = a_ptr[i+1] + b_ptr[i+1];
            result_ptr[i+2] = a_ptr[i+2] + b_ptr[i+2];
            result_ptr[i+3] = a_ptr[i+3] + b_ptr[i+3];
        }
        // Handle remaining elements
        for (; i < total_elements; i++) {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }

        return 0;
    }
}

int perform_mps_matrix_add_broadcast(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> A_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> B_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> C_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!A_buffer || !B_buffer || !C_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -3;
        }

        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        // Handle broadcasting cases
        if (aRows == resultRows && aCols == resultCols && bRows == 1 && bCols == resultCols) {
            // Case: [M, N] + [1, N] -> [M, N] (bias broadcasting)
            for (long i = 0; i < resultRows; i++) {
                for (long j = 0; j < resultCols; j++) {
                    long a_idx = i * aCols + j;
                    long b_idx = j; // Broadcast along rows
                    long result_idx = i * resultCols + j;
                    result_data[result_idx] = a_data[a_idx] + b_data[b_idx];
                }
            }
        } else if (bRows == resultRows && bCols == resultCols && aRows == 1 && aCols == resultCols) {
            // Case: [1, N] + [M, N] -> [M, N] (bias broadcasting, swapped)
            for (long i = 0; i < resultRows; i++) {
                for (long j = 0; j < resultCols; j++) {
                    long a_idx = j; // Broadcast along rows
                    long b_idx = i * bCols + j;
                    long result_idx = i * resultCols + j;
                    result_data[result_idx] = a_data[a_idx] + b_data[b_idx];
                }
            }
        } else if (aRows == resultRows && aCols == resultCols && bRows == resultRows && bCols == 1) {
            // Case: [M, N] + [M, 1] -> [M, N] (broadcast along columns)
            for (long i = 0; i < resultRows; i++) {
                for (long j = 0; j < resultCols; j++) {
                    long a_idx = i * aCols + j;
                    long b_idx = i; // Broadcast along columns
                    long result_idx = i * resultCols + j;
                    result_data[result_idx] = a_data[a_idx] + b_data[b_idx];
                }
            }
        } else if (bRows == resultRows && bCols == resultCols && aRows == resultRows && aCols == 1) {
            // Case: [M, 1] + [M, N] -> [M, N] (broadcast along columns, swapped)
            for (long i = 0; i < resultRows; i++) {
                for (long j = 0; j < resultCols; j++) {
                    long a_idx = i; // Broadcast along columns
                    long b_idx = i * bCols + j;
                    long result_idx = i * resultCols + j;
                    result_data[result_idx] = a_data[a_idx] + b_data[b_idx];
                }
            }
        } else {
            set_c_error_message(err, @"Unsupported broadcasting pattern: A (%ldx%ld) + B (%ldx%ld) -> Result (%ldx%ld)", 
                                aRows, aCols, bRows, bCols, resultRows, resultCols);
            return -2;
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

        // Add device and command queue setup for GPU acceleration  
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevicePtr;
        
        if (!device) {
            set_c_error_message(err, @"Metal device not initialized.");
            return -1;
        }
        
        // Cache the device globally for consistent access
        if (!_global_mtl_device_ptr) {
            _global_mtl_device_ptr = mtlDevicePtr;
        }
        
        // Use global command queue for better performance
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_global_mtl_command_queue_ptr;
        
        // If global queue is not set, create one and cache it
        if (!commandQueue) {
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                set_c_error_message(err, @"Failed to create command queue from device.");
                return -1;
            }
            commandQueue.label = @"go-nngpu.shared";
            _global_mtl_command_queue_ptr = (__bridge void*)commandQueue;
        }
        
        // Use optimized CPU implementation for element-wise subtraction
        long total_elements = aRows * aCols;
        
        // Optimized CPU subtraction with vectorization hints
        float *a_data = (float*)A_buffer.contents;
        float *b_data = (float*)B_buffer.contents;
        float *result_data = (float*)C_buffer.contents;
        
        // Use restrict keyword for compiler optimization hints
        float* restrict a_ptr = a_data;
        float* restrict b_ptr = b_data;
        float* restrict result_ptr = result_data;
        
        // Unrolled loop for better performance
        long i = 0;
        for (; i < total_elements - 3; i += 4) {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
            result_ptr[i+1] = a_ptr[i+1] - b_ptr[i+1];
            result_ptr[i+2] = a_ptr[i+2] - b_ptr[i+2];
            result_ptr[i+3] = a_ptr[i+3] - b_ptr[i+3];
        }
        // Handle remaining elements
        for (; i < total_elements; i++) {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

int perform_matrix_inverse(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr resultMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify it's a square matrix
        if (rows != cols) {
            set_c_error_message(err, @"Matrix inverse requires a square matrix.");
            return -1;
        }

        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!input_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -2;
        }

        float *input_data = (float*)input_buffer.contents;
        float *result_data = (float*)result_buffer.contents;
        
        // Copy input to result buffer first (sgetrf modifies in-place)
        memcpy(result_data, input_data, rows * cols * sizeof(float));
        
        // Use LAPACK's sgetrf (LU factorization) + sgetri (compute inverse from LU)
        __CLPK_integer n = (__CLPK_integer)rows;
        __CLPK_integer lda = n;
        __CLPK_integer *ipiv = malloc(n * sizeof(__CLPK_integer));
        __CLPK_integer info;
        
        if (!ipiv) {
            set_c_error_message(err, @"Failed to allocate memory for pivot indices.");
            return -3;
        }
        
        // Step 1: LU factorization
        sgetrf_(&n, &n, result_data, &lda, ipiv, &info);
        
        if (info != 0) {
            free(ipiv);
            if (info > 0) {
                set_c_error_message(err, @"Matrix is singular and cannot be inverted (LU factorization failed).");
                return -4;
            } else {
                set_c_error_message(err, @"LU factorization failed with invalid parameter at position %d.", -info);
                return -5;
            }
        }
        
        // Step 2: Compute inverse from LU factorization
        __CLPK_integer lwork = n * n; // Workspace size
        float *work = malloc(lwork * sizeof(float));
        
        if (!work) {
            free(ipiv);
            set_c_error_message(err, @"Failed to allocate workspace for matrix inversion.");
            return -6;
        }
        
        sgetri_(&n, result_data, &lda, ipiv, work, &lwork, &info);
        
        free(ipiv);
        free(work);
        
        if (info != 0) {
            if (info > 0) {
                set_c_error_message(err, @"Matrix is singular and cannot be inverted (inversion failed).");
                return -7;
            } else {
                set_c_error_message(err, @"Matrix inversion failed with invalid parameter at position %d.", -info);
                return -8;
            }
        }
        
        return 0;
    }
}

int perform_matrix_determinant(
    GPUPtr inputMatrixPtr, long rows, long cols,
    float *determinant,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify it's a square matrix
        if (rows != cols) {
            set_c_error_message(err, @"Matrix determinant requires a square matrix.");
            return -1;
        }

        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;

        if (!input_buffer || !determinant) {
            set_c_error_message(err, @"Invalid input buffer pointer or determinant pointer.");
            return -2;
        }

        float *input_data = (float*)input_buffer.contents;
        
        // Create a copy since LU factorization modifies the matrix
        float *matrix_copy = malloc(rows * cols * sizeof(float));
        if (!matrix_copy) {
            set_c_error_message(err, @"Failed to allocate memory for matrix copy.");
            return -3;
        }
        
        memcpy(matrix_copy, input_data, rows * cols * sizeof(float));
        
        // Use LAPACK's sgetrf for LU factorization to compute determinant
        __CLPK_integer n = (__CLPK_integer)rows;
        __CLPK_integer lda = n;
        __CLPK_integer *ipiv = malloc(n * sizeof(__CLPK_integer));
        __CLPK_integer info;
        
        if (!ipiv) {
            free(matrix_copy);
            set_c_error_message(err, @"Failed to allocate memory for pivot indices.");
            return -4;
        }
        
        sgetrf_(&n, &n, matrix_copy, &lda, ipiv, &info);
        
        if (info != 0) {
            free(matrix_copy);
            free(ipiv);
            if (info > 0) {
                // Matrix is singular, determinant is 0
                *determinant = 0.0f;
                return 0;
            } else {
                set_c_error_message(err, @"LU factorization failed with invalid parameter at position %d.", -info);
                return -5;
            }
        }
        
        // Compute determinant from LU factorization
        // det(A) = det(P) * det(L) * det(U) = det(P) * 1 * product of diagonal elements of U
        // det(P) = (-1)^(number of row swaps)
        
        *determinant = 1.0f;
        int num_swaps = 0;
        
        // Count row swaps and compute product of diagonal elements
        for (int i = 0; i < n; i++) {
            if (ipiv[i] != i + 1) { // FORTRAN indexing is 1-based
                num_swaps++;
            }
            *determinant *= matrix_copy[i * n + i]; // Diagonal element of U
        }
        
        // Apply sign from permutation matrix
        if (num_swaps % 2 == 1) {
            *determinant = -*determinant;
        }
        
        free(matrix_copy);
        free(ipiv);
        
        return 0;
    }
}

#pragma clang diagnostic pop