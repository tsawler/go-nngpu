// metal_bridge_sparse.m - Sparse Matrix Operations
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

// External references to global device and command queue
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External declarations for common functions (defined in metal_bridge_common.m)
extern void set_c_error_message(CError *err, NSString *format, ...);
extern void free_c_error_message(char *message);

// ===== SPARSE MATRIX OPERATIONS =====

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