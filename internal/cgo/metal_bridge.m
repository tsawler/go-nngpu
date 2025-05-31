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

// Sparse Matrix Operations

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

// --- Phase 6B: Loss Functions ---

// Mean Squared Error (MSE) loss: L = (1/n) * sum((pred - target)²)
int perform_loss_mse_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for MSE forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        // Compute MSE: (1/n) * sum((pred - target)²)
        float sum_squared_error = 0.0f;
        for (long i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            sum_squared_error += diff * diff;
        }
        
        *loss = sum_squared_error / (float)size;
        return 0;
    }
}

// MSE gradient: d/dpred = (2/n) * (pred - target)
int perform_loss_mse_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for MSE backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        // MSE gradient: (2/n) * (pred - target)
        float scale = 2.0f / (float)size;
        for (long i = 0; i < size; i++) {
            grad_input[i] = scale * (predictions[i] - targets[i]);
        }
        
        return 0;
    }
}

// Binary Cross-Entropy loss: L = -(1/n) * sum(target * log(pred) + (1-target) * log(1-pred))
int perform_loss_binary_crossentropy_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Binary CrossEntropy forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        const float epsilon = 1e-7f; // Small value to prevent log(0)
        float sum_loss = 0.0f;
        
        for (long i = 0; i < size; i++) {
            // Clamp predictions to prevent log(0) or log(1)
            float pred = fmaxf(epsilon, fminf(1.0f - epsilon, predictions[i]));
            float target = targets[i];
            
            sum_loss += -(target * logf(pred) + (1.0f - target) * logf(1.0f - pred));
        }
        
        *loss = sum_loss / (float)size;
        return 0;
    }
}

// Binary Cross-Entropy gradient: d/dpred = (1/n) * (pred - target) / (pred * (1 - pred))
int perform_loss_binary_crossentropy_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Binary CrossEntropy backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float epsilon = 1e-7f;
        float scale = 1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            // Clamp predictions to prevent division by zero
            float pred = fmaxf(epsilon, fminf(1.0f - epsilon, predictions[i]));
            float target = targets[i];
            
            // Simplified gradient: (pred - target) / (pred * (1 - pred))
            grad_input[i] = scale * (pred - target) / (pred * (1.0f - pred));
        }
        
        return 0;
    }
}

// Categorical Cross-Entropy loss: L = -(1/n) * sum_i sum_j target[i][j] * log(pred[i][j])
int perform_loss_categorical_crossentropy_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long batchSize, long numClasses,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Categorical CrossEntropy forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        const float epsilon = 1e-7f;
        float sum_loss = 0.0f;
        
        for (long b = 0; b < batchSize; b++) {
            for (long c = 0; c < numClasses; c++) {
                long idx = b * numClasses + c;
                float pred = fmaxf(epsilon, predictions[idx]); // Prevent log(0)
                float target = targets[idx];
                
                sum_loss += -target * logf(pred);
            }
        }
        
        *loss = sum_loss / (float)batchSize;
        return 0;
    }
}

// Categorical Cross-Entropy gradient: d/dpred[i][j] = -(1/n) * target[i][j] / pred[i][j]
int perform_loss_categorical_crossentropy_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long batchSize, long numClasses,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Categorical CrossEntropy backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float epsilon = 1e-7f;
        float scale = -1.0f / (float)batchSize;
        
        for (long b = 0; b < batchSize; b++) {
            for (long c = 0; c < numClasses; c++) {
                long idx = b * numClasses + c;
                float pred = fmaxf(epsilon, predictions[idx]);
                float target = targets[idx];
                
                grad_input[idx] = scale * target / pred;
            }
        }
        
        return 0;
    }
}

// Sparse Categorical Cross-Entropy loss: targets are class indices
int perform_loss_sparse_categorical_crossentropy_forward(
    GPUPtr predictionsPtr, int *targetIndices, long batchSize, long numClasses,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;

        if (!predictions_buffer || !targetIndices || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Sparse Categorical CrossEntropy forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        
        const float epsilon = 1e-7f;
        float sum_loss = 0.0f;
        
        for (long b = 0; b < batchSize; b++) {
            int target_class = targetIndices[b];
            
            if (target_class < 0 || target_class >= numClasses) {
                set_c_error_message(err, @"Target class index %d out of bounds [0, %ld) for batch %ld.", target_class, numClasses, b);
                return -2;
            }
            
            long idx = b * numClasses + target_class;
            float pred = fmaxf(epsilon, predictions[idx]);
            
            sum_loss += -logf(pred);
        }
        
        *loss = sum_loss / (float)batchSize;
        return 0;
    }
}

// Sparse Categorical Cross-Entropy gradient
int perform_loss_sparse_categorical_crossentropy_backward(
    GPUPtr predictionsPtr, int *targetIndices, long batchSize, long numClasses,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targetIndices || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Sparse Categorical CrossEntropy backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        const float epsilon = 1e-7f;
        float scale = -1.0f / (float)batchSize;
        
        // Initialize gradients to zero
        memset(grad_input, 0, batchSize * numClasses * sizeof(float));
        
        for (long b = 0; b < batchSize; b++) {
            int target_class = targetIndices[b];
            
            if (target_class < 0 || target_class >= numClasses) {
                set_c_error_message(err, @"Target class index %d out of bounds [0, %ld) for batch %ld.", target_class, numClasses, b);
                return -2;
            }
            
            long idx = b * numClasses + target_class;
            float pred = fmaxf(epsilon, predictions[idx]);
            
            grad_input[idx] = scale / pred;
        }
        
        return 0;
    }
}

// Huber loss: smooth combination of MSE and MAE
int perform_loss_huber_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float delta,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Huber forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        float sum_loss = 0.0f;
        
        for (long i = 0; i < size; i++) {
            float diff = fabsf(predictions[i] - targets[i]);
            
            if (diff <= delta) {
                // Quadratic region: 0.5 * diff²
                sum_loss += 0.5f * diff * diff;
            } else {
                // Linear region: delta * (diff - 0.5 * delta)
                sum_loss += delta * (diff - 0.5f * delta);
            }
        }
        
        *loss = sum_loss / (float)size;
        return 0;
    }
}

// Huber loss gradient
int perform_loss_huber_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float delta,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Huber backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        float scale = 1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            float abs_diff = fabsf(diff);
            
            if (abs_diff <= delta) {
                // Quadratic region: gradient = diff
                grad_input[i] = scale * diff;
            } else {
                // Linear region: gradient = delta * sign(diff)
                grad_input[i] = scale * delta * ((diff > 0.0f) ? 1.0f : -1.0f);
            }
        }
        
        return 0;
    }
}

// Mean Absolute Error (MAE) loss: L = (1/n) * sum(|pred - target|)
int perform_loss_mae_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for MAE forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        float sum_abs_error = 0.0f;
        for (long i = 0; i < size; i++) {
            sum_abs_error += fabsf(predictions[i] - targets[i]);
        }
        
        *loss = sum_abs_error / (float)size;
        return 0;
    }
}

// MAE gradient: d/dpred = (1/n) * sign(pred - target)
int perform_loss_mae_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for MAE backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        float scale = 1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            float diff = predictions[i] - targets[i];
            // Sign function: 1 if positive, -1 if negative, 0 if zero
            grad_input[i] = scale * ((diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f);
        }
        
        return 0;
    }
}

// Hinge loss: L = (1/n) * sum(max(0, 1 - target * pred))
int perform_loss_hinge_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;

        if (!predictions_buffer || !targets_buffer || !loss) {
            set_c_error_message(err, @"Invalid buffer pointers or loss pointer for Hinge forward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        
        float sum_loss = 0.0f;
        
        for (long i = 0; i < size; i++) {
            float margin = 1.0f - targets[i] * predictions[i];
            sum_loss += fmaxf(0.0f, margin);
        }
        
        *loss = sum_loss / (float)size;
        return 0;
    }
}

// Hinge loss gradient: d/dpred = -(1/n) * target if margin > 0, 0 otherwise
int perform_loss_hinge_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> predictions_buffer = (__bridge id<MTLBuffer>)predictionsPtr;
        id<MTLBuffer> targets_buffer = (__bridge id<MTLBuffer>)targetsPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!predictions_buffer || !targets_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Hinge backward.");
            return -1;
        }

        float *predictions = (float*)predictions_buffer.contents;
        float *targets = (float*)targets_buffer.contents;
        float *grad_input = (float*)grad_input_buffer.contents;
        
        float scale = -1.0f / (float)size;
        
        for (long i = 0; i < size; i++) {
            float margin = 1.0f - targets[i] * predictions[i];
            
            if (margin > 0.0f) {
                grad_input[i] = scale * targets[i];
            } else {
                grad_input[i] = 0.0f;
            }
        }
        
        return 0;
    }
}

// Convolution Operations ---

// 2D Convolution forward pass
int perform_conv2d_forward(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr kernelPtr, long kernelHeight, long kernelWidth, long kernelInputChannels, long kernelOutputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> kernel_buffer = (__bridge id<MTLBuffer>)kernelPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !kernel_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Conv2D forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *kernel_data = (float*)kernel_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Initialize output to zero
        memset(output_data, 0, outputBatch * outputHeight * outputWidth * outputChannels * sizeof(float));
        
        // Perform convolution using CPU implementation
        for (long b = 0; b < inputBatch; b++) {
            for (long oc = 0; oc < kernelOutputChannels; oc++) {
                for (long oh = 0; oh < outputHeight; oh++) {
                    for (long ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        
                        for (long ic = 0; ic < inputChannels; ic++) {
                            for (long kh = 0; kh < kernelHeight; kh++) {
                                for (long kw = 0; kw < kernelWidth; kw++) {
                                    long ih = oh * strideH - padH + kh;
                                    long iw = ow * strideW - padW + kw;
                                    
                                    // Check bounds
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        long input_idx = b * inputHeight * inputWidth * inputChannels +
                                                       ih * inputWidth * inputChannels +
                                                       iw * inputChannels + ic;
                                        
                                        long kernel_idx = kh * kernelWidth * kernelInputChannels * kernelOutputChannels +
                                                        kw * kernelInputChannels * kernelOutputChannels +
                                                        ic * kernelOutputChannels + oc;
                                        
                                        sum += input_data[input_idx] * kernel_data[kernel_idx];
                                    }
                                }
                            }
                        }
                        
                        long output_idx = b * outputHeight * outputWidth * outputChannels +
                                        oh * outputWidth * outputChannels +
                                        ow * outputChannels + oc;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
        
        return 0;
    }
}

// 2D Convolution backward pass - input gradients
int perform_conv2d_backward_input(
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr kernelPtr, long kernelHeight, long kernelWidth, long kernelInputChannels, long kernelOutputChannels,
    GPUPtr gradInputPtr, long gradInputBatch, long gradInputHeight, long gradInputWidth, long gradInputChannels,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> kernel_buffer = (__bridge id<MTLBuffer>)kernelPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !kernel_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Conv2D backward input.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *kernel_data = (float*)kernel_buffer.contents;
        float *grad_input_data = (float*)grad_input_buffer.contents;
        
        // Initialize gradient input to zero
        memset(grad_input_data, 0, gradInputBatch * gradInputHeight * gradInputWidth * gradInputChannels * sizeof(float));
        
        // Compute gradients with respect to input
        for (long b = 0; b < gradOutputBatch; b++) {
            for (long oh = 0; oh < gradOutputHeight; oh++) {
                for (long ow = 0; ow < gradOutputWidth; ow++) {
                    for (long oc = 0; oc < gradOutputChannels; oc++) {
                        long grad_output_idx = b * gradOutputHeight * gradOutputWidth * gradOutputChannels +
                                             oh * gradOutputWidth * gradOutputChannels +
                                             ow * gradOutputChannels + oc;
                        float grad_val = grad_output_data[grad_output_idx];
                        
                        for (long ic = 0; ic < kernelInputChannels; ic++) {
                            for (long kh = 0; kh < kernelHeight; kh++) {
                                for (long kw = 0; kw < kernelWidth; kw++) {
                                    long ih = oh * strideH - padH + kh;
                                    long iw = ow * strideW - padW + kw;
                                    
                                    // Check bounds
                                    if (ih >= 0 && ih < gradInputHeight && iw >= 0 && iw < gradInputWidth) {
                                        long grad_input_idx = b * gradInputHeight * gradInputWidth * gradInputChannels +
                                                             ih * gradInputWidth * gradInputChannels +
                                                             iw * gradInputChannels + ic;
                                        
                                        long kernel_idx = kh * kernelWidth * kernelInputChannels * kernelOutputChannels +
                                                        kw * kernelInputChannels * kernelOutputChannels +
                                                        ic * kernelOutputChannels + oc;
                                        
                                        grad_input_data[grad_input_idx] += grad_val * kernel_data[kernel_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return 0;
    }
}

// 2D Convolution backward pass - kernel gradients
int perform_conv2d_backward_kernel(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr gradKernelPtr, long kernelHeight, long kernelWidth, long kernelInputChannels, long kernelOutputChannels,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> grad_kernel_buffer = (__bridge id<MTLBuffer>)gradKernelPtr;

        if (!input_buffer || !grad_output_buffer || !grad_kernel_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Conv2D backward kernel.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *grad_kernel_data = (float*)grad_kernel_buffer.contents;
        
        // Initialize gradient kernel to zero
        memset(grad_kernel_data, 0, kernelHeight * kernelWidth * kernelInputChannels * kernelOutputChannels * sizeof(float));
        
        // Compute gradients with respect to kernel
        for (long kh = 0; kh < kernelHeight; kh++) {
            for (long kw = 0; kw < kernelWidth; kw++) {
                for (long ic = 0; ic < kernelInputChannels; ic++) {
                    for (long oc = 0; oc < kernelOutputChannels; oc++) {
                        float sum = 0.0f;
                        
                        for (long b = 0; b < inputBatch; b++) {
                            for (long oh = 0; oh < gradOutputHeight; oh++) {
                                for (long ow = 0; ow < gradOutputWidth; ow++) {
                                    long ih = oh * strideH - padH + kh;
                                    long iw = ow * strideW - padW + kw;
                                    
                                    // Check bounds
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        long input_idx = b * inputHeight * inputWidth * inputChannels +
                                                       ih * inputWidth * inputChannels +
                                                       iw * inputChannels + ic;
                                        
                                        long grad_output_idx = b * gradOutputHeight * gradOutputWidth * gradOutputChannels +
                                                             oh * gradOutputWidth * gradOutputChannels +
                                                             ow * gradOutputChannels + oc;
                                        
                                        sum += input_data[input_idx] * grad_output_data[grad_output_idx];
                                    }
                                }
                            }
                        }
                        
                        long kernel_idx = kh * kernelWidth * kernelInputChannels * kernelOutputChannels +
                                        kw * kernelInputChannels * kernelOutputChannels +
                                        ic * kernelOutputChannels + oc;
                        grad_kernel_data[kernel_idx] = sum;
                    }
                }
            }
        }
        
        return 0;
    }
}

// 2D Max Pooling forward pass
int perform_maxpool2d_forward(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    GPUPtr indicesPtr,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;
        id<MTLBuffer> indices_buffer = (__bridge id<MTLBuffer>)indicesPtr;

        if (!input_buffer || !output_buffer || !indices_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for MaxPool2D forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        float *indices_data = (float*)indices_buffer.contents;
        
        // Perform max pooling
        for (long b = 0; b < inputBatch; b++) {
            for (long c = 0; c < inputChannels; c++) {
                for (long oh = 0; oh < outputHeight; oh++) {
                    for (long ow = 0; ow < outputWidth; ow++) {
                        float max_val = -INFINITY;
                        long max_idx = -1;
                        
                        for (long ph = 0; ph < poolHeight; ph++) {
                            for (long pw = 0; pw < poolWidth; pw++) {
                                long ih = oh * strideH - padH + ph;
                                long iw = ow * strideW - padW + pw;
                                
                                // Check bounds
                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    long input_idx = b * inputHeight * inputWidth * inputChannels +
                                                   ih * inputWidth * inputChannels +
                                                   iw * inputChannels + c;
                                    
                                    if (input_data[input_idx] > max_val) {
                                        max_val = input_data[input_idx];
                                        max_idx = input_idx;
                                    }
                                }
                            }
                        }
                        
                        long output_idx = b * outputHeight * outputWidth * outputChannels +
                                        oh * outputWidth * outputChannels +
                                        ow * outputChannels + c;
                        
                        output_data[output_idx] = (max_val == -INFINITY) ? 0.0f : max_val;
                        indices_data[output_idx] = (float)max_idx; // Store index for backward pass
                    }
                }
            }
        }
        
        return 0;
    }
}

// 2D Max Pooling backward pass
int perform_maxpool2d_backward(
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr indicesPtr,
    GPUPtr gradInputPtr, long gradInputBatch, long gradInputHeight, long gradInputWidth, long gradInputChannels,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> indices_buffer = (__bridge id<MTLBuffer>)indicesPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !indices_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for MaxPool2D backward.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *indices_data = (float*)indices_buffer.contents;
        float *grad_input_data = (float*)grad_input_buffer.contents;
        
        // Initialize gradient input to zero
        memset(grad_input_data, 0, gradInputBatch * gradInputHeight * gradInputWidth * gradInputChannels * sizeof(float));
        
        // Propagate gradients back to input positions that were max
        for (long b = 0; b < gradOutputBatch; b++) {
            for (long c = 0; c < gradOutputChannels; c++) {
                for (long oh = 0; oh < gradOutputHeight; oh++) {
                    for (long ow = 0; ow < gradOutputWidth; ow++) {
                        long output_idx = b * gradOutputHeight * gradOutputWidth * gradOutputChannels +
                                        oh * gradOutputWidth * gradOutputChannels +
                                        ow * gradOutputChannels + c;
                        
                        long max_idx = (long)indices_data[output_idx];
                        if (max_idx >= 0) {
                            grad_input_data[max_idx] += grad_output_data[output_idx];
                        }
                    }
                }
            }
        }
        
        return 0;
    }
}

// 2D Average Pooling forward pass
int perform_avgpool2d_forward(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for AvgPool2D forward.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Perform average pooling
        for (long b = 0; b < inputBatch; b++) {
            for (long c = 0; c < inputChannels; c++) {
                for (long oh = 0; oh < outputHeight; oh++) {
                    for (long ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        long count = 0;
                        
                        for (long ph = 0; ph < poolHeight; ph++) {
                            for (long pw = 0; pw < poolWidth; pw++) {
                                long ih = oh * strideH - padH + ph;
                                long iw = ow * strideW - padW + pw;
                                
                                // Check bounds
                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    long input_idx = b * inputHeight * inputWidth * inputChannels +
                                                   ih * inputWidth * inputChannels +
                                                   iw * inputChannels + c;
                                    
                                    sum += input_data[input_idx];
                                    count++;
                                }
                            }
                        }
                        
                        long output_idx = b * outputHeight * outputWidth * outputChannels +
                                        oh * outputWidth * outputChannels +
                                        ow * outputChannels + c;
                        
                        output_data[output_idx] = (count > 0) ? (sum / (float)count) : 0.0f;
                    }
                }
            }
        }
        
        return 0;
    }
}

// 2D Average Pooling backward pass
int perform_avgpool2d_backward(
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr gradInputPtr, long gradInputBatch, long gradInputHeight, long gradInputWidth, long gradInputChannels,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> grad_output_buffer = (__bridge id<MTLBuffer>)gradOutputPtr;
        id<MTLBuffer> grad_input_buffer = (__bridge id<MTLBuffer>)gradInputPtr;

        if (!grad_output_buffer || !grad_input_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for AvgPool2D backward.");
            return -1;
        }

        float *grad_output_data = (float*)grad_output_buffer.contents;
        float *grad_input_data = (float*)grad_input_buffer.contents;
        
        // Initialize gradient input to zero
        memset(grad_input_data, 0, gradInputBatch * gradInputHeight * gradInputWidth * gradInputChannels * sizeof(float));
        
        // Propagate gradients back uniformly to all positions in the pool
        for (long b = 0; b < gradOutputBatch; b++) {
            for (long c = 0; c < gradOutputChannels; c++) {
                for (long oh = 0; oh < gradOutputHeight; oh++) {
                    for (long ow = 0; ow < gradOutputWidth; ow++) {
                        long output_idx = b * gradOutputHeight * gradOutputWidth * gradOutputChannels +
                                        oh * gradOutputWidth * gradOutputChannels +
                                        ow * gradOutputChannels + c;
                        
                        float grad_val = grad_output_data[output_idx];
                        
                        // Count valid positions in pool
                        long count = 0;
                        for (long ph = 0; ph < poolHeight; ph++) {
                            for (long pw = 0; pw < poolWidth; pw++) {
                                long ih = oh * strideH - padH + ph;
                                long iw = ow * strideW - padW + pw;
                                
                                if (ih >= 0 && ih < gradInputHeight && iw >= 0 && iw < gradInputWidth) {
                                    count++;
                                }
                            }
                        }
                        
                        // Distribute gradient uniformly
                        if (count > 0) {
                            float grad_per_element = grad_val / (float)count;
                            
                            for (long ph = 0; ph < poolHeight; ph++) {
                                for (long pw = 0; pw < poolWidth; pw++) {
                                    long ih = oh * strideH - padH + ph;
                                    long iw = ow * strideW - padW + pw;
                                    
                                    if (ih >= 0 && ih < gradInputHeight && iw >= 0 && iw < gradInputWidth) {
                                        long input_idx = b * gradInputHeight * gradInputWidth * gradInputChannels +
                                                       ih * gradInputWidth * gradInputChannels +
                                                       iw * gradInputChannels + c;
                                        
                                        grad_input_data[input_idx] += grad_per_element;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return 0;
    }
}

// Padding operations
int perform_pad2d(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long padTop, long padBottom, long padLeft, long padRight,
    float padValue,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Pad2D.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Initialize output with padding value
        for (long i = 0; i < outputBatch * outputHeight * outputWidth * outputChannels; i++) {
            output_data[i] = padValue;
        }
        
        // Copy input data to the center of output
        for (long b = 0; b < inputBatch; b++) {
            for (long c = 0; c < inputChannels; c++) {
                for (long ih = 0; ih < inputHeight; ih++) {
                    for (long iw = 0; iw < inputWidth; iw++) {
                        long input_idx = b * inputHeight * inputWidth * inputChannels +
                                       ih * inputWidth * inputChannels +
                                       iw * inputChannels + c;
                        
                        long oh = ih + padTop;
                        long ow = iw + padLeft;
                        
                        long output_idx = b * outputHeight * outputWidth * outputChannels +
                                        oh * outputWidth * outputChannels +
                                        ow * outputChannels + c;
                        
                        output_data[output_idx] = input_data[input_idx];
                    }
                }
            }
        }
        
        return 0;
    }
}

// Remove padding (crop)
int perform_unpad2d(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long padTop, long padBottom, long padLeft, long padRight,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Unpad2D.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Copy the center region (removing padding)
        for (long b = 0; b < outputBatch; b++) {
            for (long c = 0; c < outputChannels; c++) {
                for (long oh = 0; oh < outputHeight; oh++) {
                    for (long ow = 0; ow < outputWidth; ow++) {
                        long ih = oh + padTop;
                        long iw = ow + padLeft;
                        
                        long input_idx = b * inputHeight * inputWidth * inputChannels +
                                       ih * inputWidth * inputChannels +
                                       iw * inputChannels + c;
                        
                        long output_idx = b * outputHeight * outputWidth * outputChannels +
                                        oh * outputWidth * outputChannels +
                                        ow * outputChannels + c;
                        
                        output_data[output_idx] = input_data[input_idx];
                    }
                }
            }
        }
        
        return 0;
    }
}

// Im2Col operation for efficient convolution implementation
int perform_im2col(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr,
    long kernelHeight, long kernelWidth, long strideH, long strideW, long padH, long padW,
    long outputHeight, long outputWidth,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Im2Col.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Im2Col transforms: (batch, height, width, channels) -> (batch*out_h*out_w, kernel_h*kernel_w*channels)
        long col_idx = 0;
        
        for (long b = 0; b < inputBatch; b++) {
            for (long oh = 0; oh < outputHeight; oh++) {
                for (long ow = 0; ow < outputWidth; ow++) {
                    for (long kh = 0; kh < kernelHeight; kh++) {
                        for (long kw = 0; kw < kernelWidth; kw++) {
                            for (long c = 0; c < inputChannels; c++) {
                                long ih = oh * strideH - padH + kh;
                                long iw = ow * strideW - padW + kw;
                                
                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    long input_idx = b * inputHeight * inputWidth * inputChannels +
                                                   ih * inputWidth * inputChannels +
                                                   iw * inputChannels + c;
                                    output_data[col_idx] = input_data[input_idx];
                                } else {
                                    output_data[col_idx] = 0.0f; // Padding
                                }
                                col_idx++;
                            }
                        }
                    }
                }
            }
        }
        
        return 0;
    }
}

// Col2Im operation (inverse of Im2Col)
int perform_col2im(
    GPUPtr inputPtr,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long kernelHeight, long kernelWidth, long strideH, long strideW, long padH, long padW,
    long inputHeight, long inputWidth,
    DevicePtr mtlDevicePtr,
    CError *err
) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)inputPtr;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)outputPtr;

        if (!input_buffer || !output_buffer) {
            set_c_error_message(err, @"Invalid buffer pointers for Col2Im.");
            return -1;
        }

        float *input_data = (float*)input_buffer.contents;
        float *output_data = (float*)output_buffer.contents;
        
        // Initialize output to zero
        memset(output_data, 0, outputBatch * outputHeight * outputWidth * outputChannels * sizeof(float));
        
        // Col2Im transforms: (batch*out_h*out_w, kernel_h*kernel_w*channels) -> (batch, height, width, channels)
        long col_idx = 0;
        
        for (long b = 0; b < outputBatch; b++) {
            for (long oh = 0; oh < outputHeight; oh++) {
                for (long ow = 0; ow < outputWidth; ow++) {
                    for (long kh = 0; kh < kernelHeight; kh++) {
                        for (long kw = 0; kw < kernelWidth; kw++) {
                            for (long c = 0; c < outputChannels; c++) {
                                long ih = oh * strideH - padH + kh;
                                long iw = ow * strideW - padW + kw;
                                
                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    long output_idx = b * inputHeight * inputWidth * outputChannels +
                                                    ih * inputWidth * outputChannels +
                                                    iw * outputChannels + c;
                                    output_data[output_idx] += input_data[col_idx];
                                }
                                col_idx++;
                            }
                        }
                    }
                }
            }
        }
        
        return 0;
    }
}

// Batch Normalization Operations ---

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

#pragma clang diagnostic pop