// metal_bridge.h
#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stddef.h> // For size_t

// Opaque pointer types for Go-C communication
typedef void* GPUPtr;   // Represents an MTLBuffer*
typedef void* DevicePtr; // Represents an MTLDevice*
typedef void* CommandQueuePtr; // Represents an MTLCommandQueue*

// C-compatible error struct
typedef struct {
    char *message;
} CError;

// Global device and command queue pointers (managed by Objective-C)
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// Function prototypes for GPU buffer management
int create_gpu_buffer(float *data, long length_bytes, GPUPtr *outGPUPtr, DevicePtr *outDevicePtr, CError *err);
int retrieve_gpu_buffer_data(GPUPtr gpuPtr, float *data, long length_bytes, CError *err);
int release_gpu_buffer(GPUPtr gpuPtr);

// Matrix operations that are actually implemented and working
int perform_mps_matrix_multiplication(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_mps_matrix_transpose(
    GPUPtr inputMatrixPtr, long inputRows, long inputCols,
    GPUPtr outputMatrixPtr, long outputRows, long outputCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Element-wise operations (Phase 2)
int perform_mps_matrix_add(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_mps_matrix_subtract(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_mps_matrix_element_multiply(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_mps_matrix_element_divide(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Scalar operations
int perform_mps_matrix_scalar_add(
    GPUPtr inputMatrixPtr, long rows, long cols,
    float scalar,
    GPUPtr resultMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_mps_matrix_scalar_multiply(
    GPUPtr inputMatrixPtr, long rows, long cols,
    float scalar,
    GPUPtr resultMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Phase 3: Matrix inverse using Accelerate framework
int perform_matrix_inverse(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr resultMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Matrix determinant using Accelerate framework
int perform_matrix_determinant(
    GPUPtr inputMatrixPtr, long rows, long cols,
    float *determinant,
    DevicePtr mtlDevicePtr,
    CError *err
);

// LU decomposition using Accelerate framework
int perform_matrix_lu_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr lMatrixPtr, GPUPtr uMatrixPtr, 
    int *pivotIndices, // Array of pivot indices (size = min(rows, cols))
    DevicePtr mtlDevicePtr,
    CError *err
);

// Phase 4: Advanced Decompositions using Accelerate framework

// QR decomposition using Accelerate framework
int perform_matrix_qr_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr qMatrixPtr, GPUPtr rMatrixPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Cholesky decomposition using Accelerate framework (for symmetric positive definite matrices)
int perform_matrix_cholesky_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr lMatrixPtr, // Lower triangular matrix
    DevicePtr mtlDevicePtr,
    CError *err
);

// Eigenvalue decomposition using Accelerate framework (for symmetric matrices)
int perform_matrix_eigenvalue_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr eigenvaluesPtr, // Vector of eigenvalues (size = rows)
    GPUPtr eigenvectorsPtr, // Matrix of eigenvectors (rows x cols)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Singular Value Decomposition (SVD) using Accelerate framework
int perform_matrix_svd_decomposition(
    GPUPtr inputMatrixPtr, long rows, long cols,
    GPUPtr uMatrixPtr, // Left singular vectors (rows x rows)
    GPUPtr sVectorPtr, // Singular values (min(rows, cols))
    GPUPtr vtMatrixPtr, // Right singular vectors transposed (cols x cols)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Add these function declarations to the existing metal_bridge.h file
// Insert after the existing SVD function declaration

// Phase 5: Sparse Matrix Operations

// Sparse-sparse matrix multiplication (CSR * CSC -> Dense)
int perform_sparse_sparse_matmul(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr bValuesPtr, int *bColPtr, long bColPtrLen, int *bRowIndices, long bRowIndicesLen,
    long bRows, long bCols, long bNNZ,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sparse-dense matrix multiplication (CSR * Dense -> Dense)
int perform_sparse_dense_matmul(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr bMatrixPtr, long bRows, long bCols,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Dense-sparse matrix multiplication (Dense * CSC -> Dense)
int perform_dense_sparse_matmul(
    GPUPtr aMatrixPtr, long aRows, long aCols,
    GPUPtr bValuesPtr, int *bColPtr, long bColPtrLen, int *bRowIndices, long bRowIndicesLen,
    long bRows, long bCols, long bNNZ,
    GPUPtr resultMatrixPtr, long resultRows, long resultCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

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
);

// Sparse matrix scalar multiplication (CSR -> CSR)
int perform_sparse_scalar_multiply(
    GPUPtr valuesPtr, int *rowPtr, long rowPtrLen, int *colIndices, long colIndicesLen,
    long rows, long cols, long nnz,
    float scalar,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sparse matrix-vector multiplication (CSR * Vector -> Vector)
int perform_sparse_matvec(
    GPUPtr aValuesPtr, int *aRowPtr, long aRowPtrLen, int *aColIndices, long aColIndicesLen,
    long aRows, long aCols, long aNNZ,
    GPUPtr xVectorPtr, long xSize,
    GPUPtr resultVectorPtr, long resultSize,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sparse to dense conversion (CSR -> Dense)
int perform_sparse_to_dense(
    GPUPtr valuesPtr, int *rowPtr, long rowPtrLen, int *colIndices, long colIndicesLen,
    long rows, long cols, long nnz,
    GPUPtr denseMatrixPtr, long denseRows, long denseCols,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Dense to sparse conversion (Dense -> COO)
int perform_dense_to_sparse(
    GPUPtr denseMatrixPtr, long rows, long cols,
    float threshold,
    int *rowIndices, int *colIndices, float *values,
    long *actualNNZ, long maxNNZ,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Function to free error message allocated by Objective-C
void free_c_error_message(char *message);


// Add these function declarations after the existing SVD and sparse matrix functions
// Insert before the final #endif

// Phase 6A: Activation Functions

// ReLU activation function and its derivative
int perform_activation_relu_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_relu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sigmoid activation function and its derivative
int perform_activation_sigmoid_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_sigmoid_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Tanh activation function and its derivative
int perform_activation_tanh_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_tanh_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Softmax activation function and its derivative (1D)
int perform_activation_softmax_1d_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_softmax_1d_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Softmax activation function and its derivative (2D - batch processing)
int perform_activation_softmax_2d_forward(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_softmax_2d_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long batchSize, long features,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Leaky ReLU activation function and its derivative
int perform_activation_leaky_relu_forward(
    GPUPtr inputPtr, long size,
    float alpha,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_leaky_relu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    float alpha,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// ELU (Exponential Linear Unit) activation function and its derivative
int perform_activation_elu_forward(
    GPUPtr inputPtr, long size,
    float alpha,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_elu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    float alpha,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Swish activation function and its derivative
int perform_activation_swish_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_swish_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// GELU (Gaussian Error Linear Unit) activation function and its derivative
int perform_activation_gelu_forward(
    GPUPtr inputPtr, long size,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_activation_gelu_backward(
    GPUPtr gradOutputPtr, GPUPtr activationOutputPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

#endif // METAL_BRIDGE_H