// metal_bridge.m - Phase 4 with Advanced Decompositions - FIXED
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

// Helper function to convert row-major to column-major
void convert_row_to_col_major(float *row_major, float *col_major, long rows, long cols) {
    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            col_major[j * rows + i] = row_major[i * cols + j];
        }
    }
}

// Helper function to convert column-major to row-major
void convert_col_to_row_major(float *col_major, float *row_major, long rows, long cols) {
    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            row_major[i * cols + j] = col_major[j * rows + i];
        }
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

// --- Phase 3: Advanced Matrix Operations using CLAPACK (with warnings suppressed) ---

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

int perform_matrix_lu_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr lMatrixPtr, GPUPtr uMatrixPtr, 
    int *pivotIndices, // Array of pivot indices (size = min(rows, cols))
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> l_buffer = (__bridge id<MTLBuffer>)lMatrixPtr;
        id<MTLBuffer> u_buffer = (__bridge id<MTLBuffer>)uMatrixPtr;

        if (!input_buffer || !l_buffer || !u_buffer || !pivotIndices) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *l_data = (float*)l_buffer.contents;
        float *u_data = (float*)u_buffer.contents;
        
        // Create a copy for LU factorization
        float *lu_copy = malloc(rows * cols * sizeof(float));
        if (!lu_copy) {
            set_c_error_message(err, @"Failed to allocate memory for LU copy.");
            return -2;
        }
        
        memcpy(lu_copy, input_data, rows * cols * sizeof(float));
        
        // Use LAPACK's sgetrf for LU factorization
        __CLPK_integer m = (__CLPK_integer)rows;
        __CLPK_integer n = (__CLPK_integer)cols;
        __CLPK_integer lda = m;
        __CLPK_integer *ipiv = malloc(MIN(m, n) * sizeof(__CLPK_integer));
        __CLPK_integer info;
        
        if (!ipiv) {
            free(lu_copy);
            set_c_error_message(err, @"Failed to allocate memory for pivot indices.");
            return -3;
        }
        
        sgetrf_(&m, &n, lu_copy, &lda, ipiv, &info);
        
        if (info != 0) {
            free(lu_copy);
            free(ipiv);
            if (info > 0) {
                set_c_error_message(err, @"Matrix is singular at element (%d, %d).", info, info);
                return -4;
            } else {
                set_c_error_message(err, @"LU factorization failed with invalid parameter at position %d.", -info);
                return -5;
            }
        }
        
        // Extract L and U matrices from the packed LU result
        // Initialize L as identity matrix and U as zero matrix
        memset(l_data, 0, rows * rows * sizeof(float));
        memset(u_data, 0, cols * cols * sizeof(float));
        
        // Set L diagonal to 1
        for (long i = 0; i < rows && i < cols; i++) {
            l_data[i * rows + i] = 1.0f;
        }
        
        // Extract L (lower triangular) and U (upper triangular)
        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                if (i > j && i < rows && j < rows) {
                    // Lower triangular part goes to L
                    l_data[i * rows + j] = lu_copy[i * cols + j];
                } else if (i <= j) {
                    // Upper triangular part (including diagonal) goes to U
                    u_data[i * cols + j] = lu_copy[i * cols + j];
                }
            }
        }
        
        // Convert FORTRAN pivot indices (1-based) to C pivot indices (0-based)
        long min_dim = MIN(rows, cols);
        for (long i = 0; i < min_dim; i++) {
            pivotIndices[i] = ipiv[i] - 1;
        }
        
        free(lu_copy);
        free(ipiv);
        
        return 0;
    }
}

// --- Phase 4: Advanced Decompositions using CLAPACK - FIXED ---

int perform_matrix_qr_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr qMatrixPtr, GPUPtr rMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> q_buffer = (__bridge id<MTLBuffer>)qMatrixPtr;
        id<MTLBuffer> r_buffer = (__bridge id<MTLBuffer>)rMatrixPtr;

        if (!input_buffer || !q_buffer || !r_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *q_data = (float*)q_buffer.contents;
        float *r_data = (float*)r_buffer.contents;
        
        // Convert input from row-major to column-major for LAPACK
        float *col_major_input = malloc(rows * cols * sizeof(float));
        if (!col_major_input) {
            set_c_error_message(err, @"Failed to allocate memory for column-major input.");
            return -2;
        }
        
        convert_row_to_col_major(input_data, col_major_input, rows, cols);
        
        // Allocate tau array for Householder reflectors
        long min_dim = MIN(rows, cols);
        float *tau = malloc(min_dim * sizeof(float));
        if (!tau) {
            free(col_major_input);
            set_c_error_message(err, @"Failed to allocate memory for tau array.");
            return -3;
        }
        
        // First, compute QR factorization using sgeqrf
        __CLPK_integer m = (__CLPK_integer)rows;
        __CLPK_integer n = (__CLPK_integer)cols;
        __CLPK_integer lda = m;
        __CLPK_integer info;
        
        // Query optimal workspace size for sgeqrf
        float work_query_sgeqrf;
        __CLPK_integer lwork_sgeqrf = -1;
        sgeqrf_(&m, &n, col_major_input, &lda, tau, &work_query_sgeqrf, &lwork_sgeqrf, &info);
        
        if (info != 0) {
            free(col_major_input);
            free(tau);
            set_c_error_message(err, @"SGEQRF workspace query failed with error code %d.", info);
            return -4;
        }
        
        __CLPK_integer lwork_actual_sgeqrf = (__CLPK_integer)work_query_sgeqrf;
        float *work_sgeqrf = malloc(lwork_actual_sgeqrf * sizeof(float));
        if (!work_sgeqrf) {
            free(col_major_input);
            free(tau);
            set_c_error_message(err, @"Failed to allocate workspace for SGEQRF.");
            return -5;
        }
        
        // Perform QR factorization
        sgeqrf_(&m, &n, col_major_input, &lda, tau, work_sgeqrf, &lwork_actual_sgeqrf, &info);

        if (info != 0) {
            free(col_major_input);
            free(tau);
            free(work_sgeqrf);
            set_c_error_message(err, @"SGEQRF failed with error code %d.", info);
            return -6;
        }
        
        // Extract R matrix (upper triangular part) - convert back to row-major
        float *r_col_major = malloc(cols * cols * sizeof(float));
        if (!r_col_major) {
            free(col_major_input);
            free(tau);
            free(work_sgeqrf);
            set_c_error_message(err, @"Failed to allocate memory for R matrix.");
            return -7;
        }
        
        memset(r_col_major, 0, cols * cols * sizeof(float));
        for (long i = 0; i < MIN(rows, cols); i++) {
            for (long j = i; j < cols; j++) {
                r_col_major[j * cols + i] = col_major_input[j * rows + i];
            }
        }
        
        convert_col_to_row_major(r_col_major, r_data, cols, cols);
        free(r_col_major);
        
        // Generate Q matrix using sorgqr
        // First, prepare Q matrix data in column-major format
        float *q_col_major = malloc(rows * cols * sizeof(float));
        if (!q_col_major) {
            free(col_major_input);
            free(tau);
            free(work_sgeqrf);
            set_c_error_message(err, @"Failed to allocate memory for Q matrix.");
            return -8;
        }
        
        memcpy(q_col_major, col_major_input, rows * cols * sizeof(float));
        
        // Query workspace size for Q generation using sorgqr
        float work_query_sorgqr;
        __CLPK_integer lwork_sorgqr_query = -1;
        sorgqr_(&m, &n, &n, q_col_major, &lda, tau, &work_query_sorgqr, &lwork_sorgqr_query, &info);
        
        if (info != 0) {
            free(col_major_input);
            free(tau);
            free(work_sgeqrf);
            free(q_col_major);
            set_c_error_message(err, @"SORGQR workspace query failed with error code %d.", info);
            return -9;
        }
        
        __CLPK_integer lwork_actual_sorgqr = (__CLPK_integer)work_query_sorgqr;
        float *work_sorgqr = malloc(lwork_actual_sorgqr * sizeof(float));
        if (!work_sorgqr) {
            free(col_major_input);
            free(tau);
            free(work_sgeqrf);
            free(q_col_major);
            set_c_error_message(err, @"Failed to allocate workspace for SORGQR.");
            return -10;
        }
        
        // Generate Q matrix
        sorgqr_(&m, &n, &n, q_col_major, &lda, tau, work_sorgqr, &lwork_actual_sorgqr, &info);
        
        if (info != 0) {
            free(col_major_input);
            free(tau);
            free(work_sgeqrf);
            free(q_col_major);
            free(work_sorgqr);
            set_c_error_message(err, @"SORGQR computation failed with error code %d.", info);
            return -11;
        }
        
        // Convert Q back to row-major format
        convert_col_to_row_major(q_col_major, q_data, rows, cols);
        
        free(col_major_input);
        free(tau);
        free(work_sgeqrf);
        free(q_col_major);
        free(work_sorgqr);
        
        return 0;
    }
}

int perform_matrix_cholesky_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr lMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify it's a square matrix
        if (rows != cols) {
            set_c_error_message(err, @"Cholesky decomposition requires a square matrix.");
            return -1;
        }

        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> l_buffer = (__bridge id<MTLBuffer>)lMatrixPtr;

        if (!input_buffer || !l_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -2;
        }

        float *input_data = (float*)input_buffer.contents;
        float *l_data = (float*)l_buffer.contents;
        
        // Convert input from row-major to column-major for LAPACK
        float *col_major_matrix = malloc(rows * cols * sizeof(float));
        if (!col_major_matrix) {
            set_c_error_message(err, @"Failed to allocate memory for column-major matrix.");
            return -3;
        }
        
        convert_row_to_col_major(input_data, col_major_matrix, rows, cols);
        
        // Use LAPACK's spotrf for Cholesky decomposition
        char uplo = 'L'; // Lower triangular
        __CLPK_integer n = (__CLPK_integer)rows;
        __CLPK_integer lda = n;
        __CLPK_integer info;
        
        spotrf_(&uplo, &n, col_major_matrix, &lda, &info);
        
        if (info != 0) {
            free(col_major_matrix);
            if (info > 0) {
                set_c_error_message(err, @"Matrix is not positive definite. Leading minor of order %d is not positive.", info);
                return -4;
            } else {
                set_c_error_message(err, @"Cholesky decomposition failed with invalid parameter at position %d.", -info);
                return -5;
            }
        }
        
        // Zero out upper triangular part in column-major format
        for (long j = 0; j < cols; j++) {
            for (long i = 0; i < j; i++) {
                col_major_matrix[j * rows + i] = 0.0f;
            }
        }
        
        // Convert back to row-major format
        convert_col_to_row_major(col_major_matrix, l_data, rows, cols);
        
        free(col_major_matrix);
        
        return 0;
    }
}

int perform_matrix_eigenvalue_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr eigenvaluesPtr,
    GPUPtr eigenvectorsPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        // Verify it's a square matrix
        if (rows != cols) {
            set_c_error_message(err, @"Eigenvalue decomposition requires a square matrix.");
            return -1;
        }

        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> eigenvalues_buffer = (__bridge id<MTLBuffer>)eigenvaluesPtr;
        id<MTLBuffer> eigenvectors_buffer = (__bridge id<MTLBuffer>)eigenvectorsPtr;

        if (!input_buffer || !eigenvalues_buffer || !eigenvectors_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -2;
        }

        float *input_data = (float*)input_buffer.contents;
        float *eigenvalues = (float*)eigenvalues_buffer.contents;
        float *eigenvectors = (float*)eigenvectors_buffer.contents;
        
        // Convert input from row-major to column-major for LAPACK
        float *col_major_matrix = malloc(rows * cols * sizeof(float));
        if (!col_major_matrix) {
            set_c_error_message(err, @"Failed to allocate memory for column-major matrix.");
            return -3;
        }
        
        convert_row_to_col_major(input_data, col_major_matrix, rows, cols);
        
        // Use LAPACK's ssyev for symmetric eigenvalue decomposition
        char jobz = 'V'; // Compute eigenvalues and eigenvectors
        char uplo = 'L'; // Lower triangular part is referenced
        __CLPK_integer n = (__CLPK_integer)rows;
        __CLPK_integer lda = n;
        __CLPK_integer info;
        
        // Query optimal workspace size
        float work_query;
        __CLPK_integer lwork = -1;
        ssyev_(&jobz, &uplo, &n, col_major_matrix, &lda, eigenvalues, &work_query, &lwork, &info);
        
        if (info != 0) {
            free(col_major_matrix);
            set_c_error_message(err, @"Failed to query workspace size for eigenvalue decomposition.");
            return -4;
        }
        
        lwork = (__CLPK_integer)work_query;
        float *work = malloc(lwork * sizeof(float));
        if (!work) {
            free(col_major_matrix);
            set_c_error_message(err, @"Failed to allocate workspace for eigenvalue decomposition.");
            return -5;
        }
        
        // Perform eigenvalue decomposition
        ssyev_(&jobz, &uplo, &n, col_major_matrix, &lda, eigenvalues, work, &lwork, &info);
        
        if (info != 0) {
            free(col_major_matrix);
            free(work);
            if (info > 0) {
                set_c_error_message(err, @"Eigenvalue decomposition failed to converge. %d off-diagonal elements did not converge.", info);
                return -6;
            } else {
                set_c_error_message(err, @"Eigenvalue decomposition failed with invalid parameter at position %d.", -info);
                return -7;
            }
        }
        
        // Convert eigenvectors back to row-major format
        convert_col_to_row_major(col_major_matrix, eigenvectors, rows, cols);
        
        free(col_major_matrix);
        free(work);
        
        return 0;
    }
}

int perform_matrix_svd_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr uMatrixPtr,
    GPUPtr sVectorPtr,
    GPUPtr vtMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputMatrixPtr;
        id<MTLBuffer> u_buffer = (__bridge id<MTLBuffer>)uMatrixPtr;
        id<MTLBuffer> s_buffer = (__bridge id<MTLBuffer>)sVectorPtr;
        id<MTLBuffer> vt_buffer = (__bridge id<MTLBuffer>)vtMatrixPtr;

        if (!input_buffer || !u_buffer || !s_buffer || !vt_buffer) {
            set_c_error_message(err, @"Invalid input or output buffer pointers.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *u_data = (float*)u_buffer.contents;
        float *s_data = (float*)s_buffer.contents;
        float *vt_data = (float*)vt_buffer.contents;
        
        // Convert input from row-major to column-major for LAPACK
        float *col_major_input = malloc(rows * cols * sizeof(float));
        if (!col_major_input) {
            set_c_error_message(err, @"Failed to allocate memory for column-major input.");
            return -2;
        }
        
        convert_row_to_col_major(input_data, col_major_input, rows, cols);
        
        // Allocate column-major matrices for U and VT
        float *u_col_major = malloc(rows * rows * sizeof(float));
        float *vt_col_major = malloc(cols * cols * sizeof(float));
        
        if (!u_col_major || !vt_col_major) {
            free(col_major_input);
            if (u_col_major) free(u_col_major);
            if (vt_col_major) free(vt_col_major);
            set_c_error_message(err, @"Failed to allocate memory for U or VT matrices.");
            return -3;
        }
        
        // Use LAPACK's sgesvd for SVD
        char jobu = 'A';  // Compute all columns of U
        char jobvt = 'A'; // Compute all rows of V^T
        __CLPK_integer m = (__CLPK_integer)rows;
        __CLPK_integer n = (__CLPK_integer)cols;
        __CLPK_integer lda = m;
        __CLPK_integer ldu = m;
        __CLPK_integer ldvt = n;
        __CLPK_integer info;
        
        // Query optimal workspace size
        float work_query;
        __CLPK_integer lwork = -1;
        sgesvd_(&jobu, &jobvt, &m, &n, col_major_input, &lda, s_data, u_col_major, &ldu, vt_col_major, &ldvt, &work_query, &lwork, &info);
        
        if (info != 0) {
            free(col_major_input);
            free(u_col_major);
            free(vt_col_major);
            set_c_error_message(err, @"Failed to query workspace size for SVD.");
            return -4;
        }
        
        lwork = (__CLPK_integer)work_query;
        float *work = malloc(lwork * sizeof(float));
        if (!work) {
            free(col_major_input);
            free(u_col_major);
            free(vt_col_major);
            set_c_error_message(err, @"Failed to allocate workspace for SVD.");
            return -5;
        }
        
        // Perform SVD
        sgesvd_(&jobu, &jobvt, &m, &n, col_major_input, &lda, s_data, u_col_major, &ldu, vt_col_major, &ldvt, work, &lwork, &info);
        
        if (info != 0) {
            free(col_major_input);
            free(u_col_major);
            free(vt_col_major);
            free(work);
            if (info > 0) {
                set_c_error_message(err, @"SVD failed to converge. %d superdiagonals did not converge.", info);
                return -6;
            } else {
                set_c_error_message(err, @"SVD failed with invalid parameter at position %d.", -info);
                return -7;
            }
        }
        
        // Convert U and VT back to row-major format
        convert_col_to_row_major(u_col_major, u_data, rows, rows);
        convert_col_to_row_major(vt_col_major, vt_data, cols, cols);
        
        free(col_major_input);
        free(u_col_major);
        free(vt_col_major);
        free(work);
        
        return 0;
    }
}

// Add these imports at the top of metal_bridge.m (after existing imports)
// No additional imports needed - all required frameworks already included

// Add these functions at the end of metal_bridge.m (before the final closing brace)

// --- Phase 5: Sparse Matrix Operations ---

// Sparse-sparse matrix multiplication (CSR * CSC -> Dense)
int perform_sparse_sparse_matmul(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr bValuesPtr, int *bColPtr, long bColPtrLen, int *bRowIndices, long bRowIndicesLen,
    long bRows, long bCols, long bNNZ,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> aValues_buffer = (__bridge id<MTLBuffer>)aValuesPtr;
        id<MTLBuffer> bValues_buffer = (__bridge id<MTLBuffer>)bValuesPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!aValues_buffer || !bValues_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for sparse-sparse multiplication.");
            return -1;
        }

        // Get buffer data pointers
        float *aValues = (float*)aValues_buffer.contents;
        float *bValues = (float*)bValues_buffer.contents;
        float *result = (float*)result_buffer.contents;
        
        // Initialize result matrix to zero
        memset(result, 0, resultRows * resultCols * sizeof(float));
        
        // Perform CSR * CSC multiplication using CPU implementation
        // This is optimized for sparse matrices where GPU overhead might not be worth it
        for (long i = 0; i < aRows; i++) {
            long aStart = aRowPtr[i];
            long aEnd = aRowPtr[i + 1];
            
            for (long j = 0; j < bCols; j++) {
                long bStart = bColPtr[j];
                long bEnd = bColPtr[j + 1];
                
                float sum = 0.0f;
                long aIdx = aStart;
                long bIdx = bStart;
                
                // Merge-like operation to find matching indices
                while (aIdx < aEnd && bIdx < bEnd) {
                    int aCol = aColIndices[aIdx];
                    int bRow = bRowIndices[bIdx];
                    
                    if (aCol == bRow) {
                        sum += aValues[aIdx] * bValues[bIdx];
                        aIdx++;
                        bIdx++;
                    } else if (aCol < bRow) {
                        aIdx++;
                    } else {
                        bIdx++;
                    }
                }
                
                if (sum != 0.0f) {
                    result[i * resultCols + j] = sum;
                }
            }
        }
        
        return 0;
    }
}

// Sparse-dense matrix multiplication (CSR * Dense -> Dense)
int perform_sparse_dense_matmul(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> aValues_buffer = (__bridge id<MTLBuffer>)aValuesPtr;
        id<MTLBuffer> b_buffer = (__bridge id<MTLBuffer>)bMatrixPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!aValues_buffer || !b_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for sparse-dense multiplication.");
            return -1;
        }

        float *aValues = (float*)aValues_buffer.contents;
        float *bMatrix = (float*)b_buffer.contents;
        float *result = (float*)result_buffer.contents;
        
        // Initialize result matrix to zero
        memset(result, 0, resultRows * resultCols * sizeof(float));
        
        // Perform CSR * Dense multiplication
        for (long i = 0; i < aRows; i++) {
            long start = aRowPtr[i];
            long end = aRowPtr[i + 1];
            
            for (long k = start; k < end; k++) {
                int aCol = aColIndices[k];
                float aVal = aValues[k];
                
                // Multiply sparse element by corresponding row in dense matrix
                for (long j = 0; j < bCols; j++) {
                    result[i * resultCols + j] += aVal * bMatrix[aCol * bCols + j];
                }
            }
        }
        
        return 0;
    }
}

// Dense-sparse matrix multiplication (Dense * CSC -> Dense)
int perform_dense_sparse_matmul(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bValuesPtr, int *bColPtr, long bColPtrLen, int *bRowIndices, long bRowIndicesLen,
    long bRows, long bCols, long bNNZ,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> a_buffer = (__bridge id<MTLBuffer>)aMatrixPtr;
        id<MTLBuffer> bValues_buffer = (__bridge id<MTLBuffer>)bValuesPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultMatrixPtr;

        if (!a_buffer || !bValues_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for dense-sparse multiplication.");
            return -1;
        }

        float *aMatrix = (float*)a_buffer.contents;
        float *bValues = (float*)bValues_buffer.contents;
        float *result = (float*)result_buffer.contents;
        
        // Initialize result matrix to zero
        memset(result, 0, resultRows * resultCols * sizeof(float));
        
        // Perform Dense * CSC multiplication
        for (long j = 0; j < bCols; j++) {
            long start = bColPtr[j];
            long end = bColPtr[j + 1];
            
            for (long k = start; k < end; k++) {
                int bRow = bRowIndices[k];
                float bVal = bValues[k];
                
                // Multiply corresponding column in dense matrix by sparse element
                for (long i = 0; i < aRows; i++) {
                    result[i * resultCols + j] += aMatrix[i * aCols + bRow] * bVal;
                }
            }
        }
        
        return 0;
    }
}

// Sparse matrix addition (CSR + CSR -> COO)
int perform_sparse_add(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr bValuesPtr, int *bRowPtr, long bRowPtrLen, int *bColIndices, long bColIndicesLen,
    long bRows, long bCols, long bNNZ,
    int *resultRowIndices, int *resultColIndices, float *resultValues,
    long *actualNNZ, long maxResultNNZ,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        if (aRows != bRows || aCols != bCols) {
            set_c_error_message(err, @"Matrix dimensions must match for addition.");
            return -1;
        }

        id<MTLBuffer> aValues_buffer = (__bridge id<MTLBuffer>)aValuesPtr;
        id<MTLBuffer> bValues_buffer = (__bridge id<MTLBuffer>)bValuesPtr;

        if (!aValues_buffer || !bValues_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for sparse addition.");
            return -1;
        }

        float *aValues = (float*)aValues_buffer.contents;
        float *bValues = (float*)bValues_buffer.contents;
        
        long resultCount = 0;
        
        // Perform row-wise addition of two CSR matrices
        for (long i = 0; i < aRows; i++) {
            long aStart = aRowPtr[i];
            long aEnd = aRowPtr[i + 1];
            long bStart = bRowPtr[i];
            long bEnd = bRowPtr[i + 1];
            
            long aIdx = aStart;
            long bIdx = bStart;
            
            // Merge two sorted sequences
            while (aIdx < aEnd || bIdx < bEnd) {
                if (resultCount >= maxResultNNZ) {
                    set_c_error_message(err, @"Result matrix exceeds maximum allocated space.");
                    return -2;
                }
                
                int aCol = (aIdx < aEnd) ? aColIndices[aIdx] : INT_MAX;
                int bCol = (bIdx < bEnd) ? bColIndices[bIdx] : INT_MAX;
                
                if (aCol == bCol && aCol != INT_MAX) {
                    // Both matrices have elements at this position
                    float sum = aValues[aIdx] + bValues[bIdx];
                    if (sum != 0.0f) { // Only store non-zero results
                        resultRowIndices[resultCount] = (int)i;
                        resultColIndices[resultCount] = aCol;
                        resultValues[resultCount] = sum;
                        resultCount++;
                    }
                    aIdx++;
                    bIdx++;
                } else if (aCol < bCol) {
                    // Only matrix A has element at this position
                    resultRowIndices[resultCount] = (int)i;
                    resultColIndices[resultCount] = aCol;
                    resultValues[resultCount] = aValues[aIdx];
                    resultCount++;
                    aIdx++;
                } else {
                    // Only matrix B has element at this position
                    resultRowIndices[resultCount] = (int)i;
                    resultColIndices[resultCount] = bCol;
                    resultValues[resultCount] = bValues[bIdx];
                    resultCount++;
                    bIdx++;
                }
            }
        }
        
        *actualNNZ = resultCount;
        return 0;
    }
}

// Sparse matrix scalar multiplication (CSR -> CSR, in-place)
int perform_sparse_scalar_multiply(
    GPUPtr valuesPtr, int *rowPtr, long rowPtrLen, int *colIndices, long colIndicesLen,
    long rows, long cols, long nnz,
    float scalar,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> values_buffer = (__bridge id<MTLBuffer>)valuesPtr;

        if (!values_buffer) {
            set_c_error_message(err, @"Invalid buffer pointer for sparse scalar multiplication.");
            return -1;
        }

        float *values = (float*)values_buffer.contents;
        
        // Simple scalar multiplication - multiply all non-zero values
        for (long i = 0; i < nnz; i++) {
            values[i] *= scalar;
        }
        
        return 0;
    }
}

// Sparse matrix-vector multiplication (CSR * Vector -> Vector)
int perform_sparse_matvec(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr xVectorPtr, long xSize,
    GPUPtr resultVectorPtr, long resultSize,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        if (aCols != xSize || aRows != resultSize) {
            set_c_error_message(err, @"Incompatible dimensions for sparse matrix-vector multiplication.");
            return -1;
        }

        id<MTLBuffer> aValues_buffer = (__bridge id<MTLBuffer>)aValuesPtr;
        id<MTLBuffer> x_buffer = (__bridge id<MTLBuffer>)xVectorPtr;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)resultVectorPtr;

        if (!aValues_buffer || !x_buffer || !result_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for sparse matrix-vector multiplication.");
            return -1;
        }

        float *aValues = (float*)aValues_buffer.contents;
        float *x = (float*)x_buffer.contents;
        float *result = (float*)result_buffer.contents;
        
        // Initialize result vector to zero
        memset(result, 0, resultSize * sizeof(float));
        
        // Perform CSR * vector multiplication
        for (long i = 0; i < aRows; i++) {
            long start = aRowPtr[i];
            long end = aRowPtr[i + 1];
            
            float sum = 0.0f;
            for (long k = start; k < end; k++) {
                int col = aColIndices[k];
                sum += aValues[k] * x[col];
            }
            result[i] = sum;
        }
        
        return 0;
    }
}

// Sparse to dense conversion (CSR -> Dense)
int perform_sparse_to_dense(
    GPUPtr valuesPtr, int *rowPtr, long rowPtrLen, int *colIndices, long colIndicesLen,
    long rows, long cols, long nnz,
    GPUPtr denseMatrixPtr, long denseRows, long denseCols,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        if (rows != denseRows || cols != denseCols) {
            set_c_error_message(err, @"Dimension mismatch in sparse to dense conversion.");
            return -1;
        }

        id<MTLBuffer> values_buffer = (__bridge id<MTLBuffer>)valuesPtr;
        id<MTLBuffer> dense_buffer = (__bridge id<MTLBuffer>)denseMatrixPtr;

        if (!values_buffer || !dense_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for sparse to dense conversion.");
            return -1;
        }

        float *values = (float*)values_buffer.contents;
        float *dense = (float*)dense_buffer.contents;
        
        // Initialize dense matrix to zero
        memset(dense, 0, denseRows * denseCols * sizeof(float));
        
        // Fill dense matrix from CSR format
        for (long i = 0; i < rows; i++) {
            long start = rowPtr[i];
            long end = rowPtr[i + 1];
            
            for (long k = start; k < end; k++) {
                int col = colIndices[k];
                dense[i * denseCols + col] = values[k];
            }
        }
        
        return 0;
    }
}

// Dense to sparse conversion (Dense -> COO)
int perform_dense_to_sparse(
    GPUPtr denseMatrixPtr, long rows, long cols,
    float threshold,
    int *rowIndices, int *colIndices, float *values,
    long *actualNNZ, long maxNNZ,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> dense_buffer = (__bridge id<MTLBuffer>)denseMatrixPtr;

        if (!dense_buffer) {
            set_c_error_message(err, @"Invalid buffer pointer for dense to sparse conversion.");
            return -1;
        }

        float *dense = (float*)dense_buffer.contents;
        long count = 0;
        
        // Extract non-zero elements
        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                float val = dense[i * cols + j];
                
                if (fabsf(val) > threshold) {
                    if (count >= maxNNZ) {
                        set_c_error_message(err, @"Too many non-zero elements, exceeds allocated space.");
                        return -2;
                    }
                    
                    rowIndices[count] = (int)i;
                    colIndices[count] = (int)j;
                    values[count] = val;
                    count++;
                }
            }
        }
        
        *actualNNZ = count;
        return 0;
    }
}

#pragma clang diagnostic pop