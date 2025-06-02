// metal_bridge.h
#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stddef.h> // For size_t
#include <stdbool.h> // Include this header to use 'bool' as a macro for '_Bool'

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

// Loss Functions

// Mean Squared Error (MSE) loss
int perform_loss_mse_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_mse_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Binary Cross-Entropy loss
int perform_loss_binary_crossentropy_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_binary_crossentropy_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Categorical Cross-Entropy loss (for softmax outputs)
int perform_loss_categorical_crossentropy_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long batchSize, long numClasses,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_categorical_crossentropy_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long batchSize, long numClasses,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sparse Categorical Cross-Entropy loss (targets are class indices)
int perform_loss_sparse_categorical_crossentropy_forward(
    GPUPtr predictionsPtr, int *targetIndices, long batchSize, long numClasses,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_sparse_categorical_crossentropy_backward(
    GPUPtr predictionsPtr, int *targetIndices, long batchSize, long numClasses,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Huber loss (robust regression loss)
int perform_loss_huber_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float delta,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_huber_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float delta,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Mean Absolute Error (MAE) loss
int perform_loss_mae_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_mae_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Hinge loss (for SVM-style classification)
int perform_loss_hinge_forward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    float *loss,
    DevicePtr mtlDevicePtr,
    CError *err
);

int perform_loss_hinge_backward(
    GPUPtr predictionsPtr, GPUPtr targetsPtr, long size,
    GPUPtr gradInputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Add these function declarations to metal_bridge.h after the existing loss functions
// Insert before the final #endif

// Phase 6C: Convolution Operations

// 2D Convolution forward pass
int perform_conv2d_forward(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr kernelPtr, long kernelHeight, long kernelWidth, long kernelInputChannels, long kernelOutputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// 2D Convolution backward pass - input gradients
int perform_conv2d_backward_input(
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr kernelPtr, long kernelHeight, long kernelWidth, long kernelInputChannels, long kernelOutputChannels,
    GPUPtr gradInputPtr, long gradInputBatch, long gradInputHeight, long gradInputWidth, long gradInputChannels,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// 2D Convolution backward pass - kernel gradients
int perform_conv2d_backward_kernel(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr gradKernelPtr, long kernelHeight, long kernelWidth, long kernelInputChannels, long kernelOutputChannels,
    long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// 2D Max Pooling forward pass
int perform_maxpool2d_forward(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    GPUPtr indicesPtr, // For backward pass - stores indices of max elements
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// 2D Max Pooling backward pass
int perform_maxpool2d_backward(
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr indicesPtr, // Indices from forward pass
    GPUPtr gradInputPtr, long gradInputBatch, long gradInputHeight, long gradInputWidth, long gradInputChannels,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// 2D Average Pooling forward pass
int perform_avgpool2d_forward(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// 2D Average Pooling backward pass
int perform_avgpool2d_backward(
    GPUPtr gradOutputPtr, long gradOutputBatch, long gradOutputHeight, long gradOutputWidth, long gradOutputChannels,
    GPUPtr gradInputPtr, long gradInputBatch, long gradInputHeight, long gradInputWidth, long gradInputChannels,
    long poolHeight, long poolWidth, long strideH, long strideW, long padH, long padW,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Padding operations
int perform_pad2d(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long padTop, long padBottom, long padLeft, long padRight,
    float padValue, // Value to fill padding with
    DevicePtr mtlDevicePtr,
    CError *err
);

// Remove padding (crop)
int perform_unpad2d(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long padTop, long padBottom, long padLeft, long padRight,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Im2Col operation for efficient convolution implementation
int perform_im2col(
    GPUPtr inputPtr, long inputBatch, long inputHeight, long inputWidth, long inputChannels,
    GPUPtr outputPtr, // Output is (batch * output_h * output_w) x (kernel_h * kernel_w * input_channels)
    long kernelHeight, long kernelWidth, long strideH, long strideW, long padH, long padW,
    long outputHeight, long outputWidth,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Col2Im operation (inverse of Im2Col)
int perform_col2im(
    GPUPtr inputPtr, // Input is (batch * output_h * output_w) x (kernel_h * kernel_w * input_channels)
    GPUPtr outputPtr, long outputBatch, long outputHeight, long outputWidth, long outputChannels,
    long kernelHeight, long kernelWidth, long strideH, long strideW, long padH, long padW,
    long inputHeight, long inputWidth, // Original input dimensions before im2col
    DevicePtr mtlDevicePtr,
    CError *err
);

// Batch Normalization Operations

// Batch mean computation (across batch dimension)
int perform_batch_mean(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr meanPtr, // Output: mean for each feature (size = features)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Batch variance computation (across batch dimension)
int perform_batch_variance(
    GPUPtr inputPtr, GPUPtr meanPtr, long batchSize, long features,
    GPUPtr variancePtr, // Output: variance for each feature (size = features)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Batch normalization forward pass
int perform_batch_norm_forward(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr meanPtr, GPUPtr variancePtr, // Running statistics (size = features)
    GPUPtr gammaPtr, GPUPtr betaPtr, // Learnable parameters (size = features)
    float epsilon, // Small constant for numerical stability
    GPUPtr outputPtr, // Normalized output (size = batchSize * features)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Batch normalization backward pass - input gradients
int perform_batch_norm_backward_input(
    GPUPtr gradOutputPtr, long batchSize, long features,
    GPUPtr inputPtr, GPUPtr meanPtr, GPUPtr variancePtr,
    GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, // Output: gradients w.r.t. input (size = batchSize * features)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Batch normalization backward pass - parameter gradients
int perform_batch_norm_backward_params(
    GPUPtr gradOutputPtr, long batchSize, long features,
    GPUPtr inputPtr, GPUPtr meanPtr, GPUPtr variancePtr,
    float epsilon,
    GPUPtr gradGammaPtr, GPUPtr gradBetaPtr, // Output: gradients w.r.t. gamma and beta (size = features)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Layer normalization forward pass (normalize across features for each sample)
int perform_layer_norm_forward(
    GPUPtr inputPtr, long batchSize, long features,
    GPUPtr gammaPtr, GPUPtr betaPtr, // Learnable parameters (size = features)
    float epsilon,
    GPUPtr outputPtr, GPUPtr meanPtr, GPUPtr variancePtr, // meanPtr and variancePtr are per-sample (size = batchSize)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Layer normalization backward pass
int perform_layer_norm_backward(
    GPUPtr gradOutputPtr, long batchSize, long features,
    GPUPtr inputPtr, GPUPtr meanPtr, GPUPtr variancePtr, // Per-sample statistics
    GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Running statistics update (for inference)
int perform_update_running_stats(
    GPUPtr runningMeanPtr, GPUPtr runningVarPtr, // Running statistics to update (size = features)
    GPUPtr batchMeanPtr, GPUPtr batchVarPtr, // Current batch statistics (size = features)
    float momentum, long features,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Instance normalization forward pass (normalize across spatial dimensions for each channel)
int perform_instance_norm_forward(
    GPUPtr inputPtr, long batchSize, long channels, long height, long width,
    GPUPtr gammaPtr, GPUPtr betaPtr, // Learnable parameters (size = channels)
    float epsilon,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Instance normalization backward pass
int perform_instance_norm_backward(
    GPUPtr gradOutputPtr, long batchSize, long channels, long height, long width,
    GPUPtr inputPtr, GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Group normalization forward pass (normalize within groups of channels)
int perform_group_norm_forward(
    GPUPtr inputPtr, long batchSize, long channels, long height, long width,
    long numGroups, // Number of groups to divide channels into
    GPUPtr gammaPtr, GPUPtr betaPtr, // Learnable parameters (size = channels)
    float epsilon,
    GPUPtr outputPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Group normalization backward pass
int perform_group_norm_backward(
    GPUPtr gradOutputPtr, long batchSize, long channels, long height, long width,
    long numGroups, GPUPtr inputPtr, GPUPtr gammaPtr, float epsilon,
    GPUPtr gradInputPtr, GPUPtr gradGammaPtr, GPUPtr gradBetaPtr,
    DevicePtr mtlDevicePtr,
    CError *err
);

// Gradient Computation Operations

// Gradient accumulation (for variables used multiple times in computation graph)
int perform_gradient_accumulate(
    GPUPtr existingGradPtr, // Existing accumulated gradient
    GPUPtr newGradPtr,      // New gradient to add
    long size,              // Number of elements
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sum of squares computation (for gradient norm calculation)
int perform_tensor_sum_squares(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float *sumSquares,      // Output: sum of squares
    DevicePtr mtlDevicePtr,
    CError *err
);

// Sum along specific axis (for gradient broadcasting)
int perform_sum_along_axis(
    GPUPtr inputPtr,        // Input tensor
    int axis,               // Axis to sum along
    long inputNDim,         // Number of input dimensions
    long *inputShape,       // Input tensor shape
    GPUPtr outputPtr,       // Output tensor (reduced along axis)
    long outputNDim,        // Number of output dimensions
    long *outputShape,      // Output tensor shape
    DevicePtr mtlDevicePtr,
    CError *err
);

// Broadcast gradient from reduced shape back to original shape
int perform_broadcast_gradient(
    GPUPtr gradPtr,         // Gradient tensor (reduced shape)
    long gradNDim,          // Number of gradient dimensions
    long *gradShape,        // Gradient tensor shape
    GPUPtr outputPtr,       // Output tensor (original shape)
    long outputNDim,        // Number of output dimensions
    long *outputShape,      // Output tensor shape
    DevicePtr mtlDevicePtr,
    CError *err
);

// Element-wise gradient scaling (for gradient clipping)
int perform_gradient_scale(
    GPUPtr gradPtr,         // Gradient tensor to scale
    long size,              // Number of elements
    float scale,            // Scale factor
    DevicePtr mtlDevicePtr,
    CError *err
);

// Set tensor elements to a specific value (for gradient zeroing)
int perform_tensor_fill(
    GPUPtr tensorPtr,       // Tensor to fill
    long size,              // Number of elements
    float value,            // Value to fill with
    DevicePtr mtlDevicePtr,
    CError *err
);

// Element-wise maximum (for gradient clipping operations)
int perform_tensor_clamp_max(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float maxValue,         // Maximum value to clamp to
    GPUPtr outputPtr,       // Output tensor
    DevicePtr mtlDevicePtr,
    CError *err
);

// Element-wise minimum (for gradient clipping operations)
int perform_tensor_clamp_min(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float minValue,         // Minimum value to clamp to
    GPUPtr outputPtr,       // Output tensor
    DevicePtr mtlDevicePtr,
    CError *err
);

// Combined clamp operation (clamp between min and max)
int perform_tensor_clamp(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float minValue,         // Minimum value
    float maxValue,         // Maximum value
    GPUPtr outputPtr,       // Output tensor
    DevicePtr mtlDevicePtr,
    CError *err
);

// Compute L2 norm of a tensor
int perform_tensor_l2_norm(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float *norm,            // Output: L2 norm
    DevicePtr mtlDevicePtr,
    CError *err
);

// Apply dropout mask during training (for gradient computation)
int perform_dropout_forward(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float probability,      // Dropout probability
    unsigned int seed,      // Random seed
    GPUPtr outputPtr,       // Output tensor
    GPUPtr maskPtr,         // Dropout mask (for backward pass)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Apply saved dropout mask during backward pass
int perform_dropout_backward(
    GPUPtr gradOutputPtr,   // Gradient from next layer
    GPUPtr maskPtr,         // Saved dropout mask from forward pass
    long size,              // Number of elements
    float probability,      // Dropout probability (for scaling)
    GPUPtr gradInputPtr,    // Output: gradient for input
    DevicePtr mtlDevicePtr,
    CError *err
);

// Copy tensor data (for gradient checkpointing and tensor cloning)
int perform_tensor_copy(
    GPUPtr srcPtr,          // Source tensor
    GPUPtr dstPtr,          // Destination tensor
    long size,              // Number of elements
    DevicePtr mtlDevicePtr,
    CError *err
);

// Tensor reduction operations for gradient computation

// Reduce sum across all elements
int perform_tensor_sum_all(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float *sum,             // Output: sum of all elements
    DevicePtr mtlDevicePtr,
    CError *err
);

// Reduce mean across all elements
int perform_tensor_mean_all(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float *mean,            // Output: mean of all elements
    DevicePtr mtlDevicePtr,
    CError *err
);

// Reduce max across all elements
int perform_tensor_max_all(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float *maxValue,        // Output: maximum element
    DevicePtr mtlDevicePtr,
    CError *err
);

// Reduce min across all elements
int perform_tensor_min_all(
    GPUPtr inputPtr,        // Input tensor
    long size,              // Number of elements
    float *minValue,        // Output: minimum element
    DevicePtr mtlDevicePtr,
    CError *err
);

// Advanced gradient operations

// Compute gradient of matrix determinant
int perform_det_backward(
    GPUPtr inputPtr,        // Original input matrix
    long rows, long cols,   // Matrix dimensions
    float detValue,         // Determinant value from forward pass
    GPUPtr gradOutputPtr,   // Gradient of loss w.r.t. determinant (scalar)
    GPUPtr gradInputPtr,    // Output: gradient w.r.t. input matrix
    DevicePtr mtlDevicePtr,
    CError *err
);

// Compute gradient of matrix inverse
int perform_inverse_backward(
    GPUPtr inversePtr,      // Matrix inverse from forward pass
    long rows, long cols,   // Matrix dimensions
    GPUPtr gradOutputPtr,   // Gradient of loss w.r.t. inverse
    GPUPtr gradInputPtr,    // Output: gradient w.r.t. original matrix
    DevicePtr mtlDevicePtr,
    CError *err
);

// Compute gradient of eigenvalue decomposition
int perform_eigen_backward(
    GPUPtr eigenvaluesPtr,  // Eigenvalues from forward pass
    GPUPtr eigenvectorsPtr, // Eigenvectors from forward pass
    long size,              // Matrix size (square matrix)
    GPUPtr gradEigenvaluesPtr,  // Gradient w.r.t. eigenvalues
    GPUPtr gradEigenvectorsPtr, // Gradient w.r.t. eigenvectors
    GPUPtr gradInputPtr,    // Output: gradient w.r.t. original matrix
    DevicePtr mtlDevicePtr,
    CError *err
);

// Memory management for gradient computation

// Allocate GPU memory for gradient computation workspace
int allocate_gradient_workspace(
    long workspaceSize,     // Size in bytes
    GPUPtr *workspacePtr,   // Output: pointer to allocated workspace
    DevicePtr mtlDevicePtr,
    CError *err
);

// Free gradient computation workspace
int free_gradient_workspace(
    GPUPtr workspacePtr,    // Workspace to free
    CError *err
);

// Synchronization operations for gradient computation

// Synchronize GPU computation (wait for completion)
int synchronize_gpu(
    DevicePtr mtlDevicePtr,
    CError *err
);

// Check if GPU computation is complete
int is_gpu_computation_complete(
    DevicePtr mtlDevicePtr,
    int *isComplete,        // Output: 1 if complete, 0 if still running
    CError *err
);

// Optimizers (SGD, Adam, RMSprop with GPU state)

// SGD (Stochastic Gradient Descent) optimizer
int perform_sgd_step(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // Gradients for parameters
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float momentum,         // Momentum coefficient (0.0 for no momentum)
    GPUPtr momentumBufferPtr, // Momentum buffer (can be NULL if momentum=0)
    float weightDecay,      // Weight decay coefficient (0.0 for no weight decay)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Adam optimizer step
int perform_adam_step(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // Gradients for parameters
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float beta1,            // First moment decay rate (typically 0.9)
    float beta2,            // Second moment decay rate (typically 0.999)
    float epsilon,          // Small constant for numerical stability (typically 1e-8)
    float weightDecay,      // Weight decay coefficient (0.0 for no weight decay)
    GPUPtr m_bufferPtr,     // First moment buffer (mean of gradients)
    GPUPtr v_bufferPtr,     // Second moment buffer (uncentered variance of gradients)
    long stepNumber,        // Current step number (for bias correction)
    DevicePtr mtlDevicePtr,
    CError *err
);

// RMSprop optimizer step
int perform_rmsprop_step(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // Gradients for parameters
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float alpha,            // Smoothing constant (typically 0.99)
    float epsilon,          // Small constant for numerical stability (typically 1e-8)
    float momentum,         // Momentum factor (0.0 for no momentum)
    float weightDecay,      // Weight decay coefficient (0.0 for no weight decay)
    GPUPtr squaredGradBufferPtr, // Running average of squared gradients
    GPUPtr momentumBufferPtr,    // Momentum buffer (can be NULL if momentum=0)
    DevicePtr mtlDevicePtr,
    CError *err
);

// AdaGrad optimizer step
int perform_adagrad_step(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // Gradients for parameters
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float epsilon,          // Small constant for numerical stability (typically 1e-8)
    float weightDecay,      // Weight decay coefficient (0.0 for no weight decay)
    GPUPtr accumulatedSquaredGradPtr, // Accumulated squared gradients
    DevicePtr mtlDevicePtr,
    CError *err
);

// Adadelta optimizer step
int perform_adadelta_step(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // Gradients for parameters
    long size,              // Number of parameters
    float rho,              // Decay rate for running averages (typically 0.9)
    float epsilon,          // Small constant for numerical stability (typically 1e-6)
    float weightDecay,      // Weight decay coefficient (0.0 for no weight decay)
    GPUPtr accumulatedGradPtr,    // Running average of squared gradients
    GPUPtr accumulatedDeltaPtr,   // Running average of squared parameter updates
    DevicePtr mtlDevicePtr,
    CError *err
);

// AdamW optimizer step (Adam with decoupled weight decay)
int perform_adamw_step(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // Gradients for parameters
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float beta1,            // First moment decay rate (typically 0.9)
    float beta2,            // Second moment decay rate (typically 0.999)
    float epsilon,          // Small constant for numerical stability (typically 1e-8)
    float weightDecay,      // Weight decay coefficient (0.0 for no weight decay)
    GPUPtr m_bufferPtr,     // First moment buffer
    GPUPtr v_bufferPtr,     // Second moment buffer
    long stepNumber,        // Current step number (for bias correction)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Learning rate schedulers

// Exponential decay scheduler
int perform_lr_exponential_decay(
    float *currentLR,       // Current learning rate (input/output)
    float initialLR,        // Initial learning rate
    float decayRate,        // Decay rate per step
    long stepNumber,        // Current step number
    long decaySteps,        // Number of steps between decay
    CError *err
);

// Step decay scheduler
int perform_lr_step_decay(
    float *currentLR,       // Current learning rate (input/output)
    float initialLR,        // Initial learning rate
    float gamma,            // Multiplicative factor of learning rate decay
    long stepNumber,        // Current step number
    long stepSize,          // Period of learning rate decay
    CError *err
);

// Cosine annealing scheduler
int perform_lr_cosine_annealing(
    float *currentLR,       // Current learning rate (input/output)
    float initialLR,        // Initial learning rate
    float minLR,            // Minimum learning rate
    long stepNumber,        // Current step number
    long totalSteps,        // Total number of steps
    CError *err
);

// Polynomial decay scheduler
int perform_lr_polynomial_decay(
    float *currentLR,       // Current learning rate (input/output)
    float initialLR,        // Initial learning rate
    float finalLR,          // Final learning rate
    long stepNumber,        // Current step number
    long totalSteps,        // Total number of steps
    float power,            // Power of polynomial (typically 1.0 for linear)
    CError *err
);

// Gradient clipping operations

// Gradient clipping by global norm
int perform_gradient_clip_by_norm(
    GPUPtr gradPtr,         // Gradient tensor to clip
    long size,              // Number of elements
    float maxNorm,          // Maximum norm allowed
    float *actualNorm,      // Output: actual norm before clipping
    DevicePtr mtlDevicePtr,
    CError *err
);

// Gradient clipping by value
int perform_gradient_clip_by_value(
    GPUPtr gradPtr,         // Gradient tensor to clip
    long size,              // Number of elements
    float minValue,         // Minimum value
    float maxValue,         // Maximum value
    DevicePtr mtlDevicePtr,
    CError *err
);

// Global gradient norm computation (for gradient clipping across multiple tensors)
int perform_global_gradient_norm(
    GPUPtr *gradPtrs,       // Array of gradient tensor pointers
    long *sizes,            // Array of sizes for each tensor
    long numTensors,        // Number of tensors
    float *globalNorm,      // Output: global gradient norm
    DevicePtr mtlDevicePtr,
    CError *err
);

// Apply gradient clipping to multiple tensors simultaneously
int perform_global_gradient_clip(
    GPUPtr *gradPtrs,       // Array of gradient tensor pointers
    long *sizes,            // Array of sizes for each tensor
    long numTensors,        // Number of tensors
    float maxNorm,          // Maximum norm allowed
    float *actualNorm,      // Output: actual norm before clipping
    DevicePtr mtlDevicePtr,
    CError *err
);

// Optimizer state management

// Initialize optimizer state buffers (zeros for most optimizers)
int initialize_optimizer_state(
    GPUPtr statePtr,        // State buffer to initialize
    long size,              // Number of elements
    float initValue,        // Initial value (typically 0.0)
    DevicePtr mtlDevicePtr,
    CError *err
);

// Copy optimizer state between buffers
int copy_optimizer_state(
    GPUPtr srcStatePtr,     // Source state buffer
    GPUPtr dstStatePtr,     // Destination state buffer
    long size,              // Number of elements
    DevicePtr mtlDevicePtr,
    CError *err
);

// Scale optimizer state (useful for momentum scheduling)
int scale_optimizer_state(
    GPUPtr statePtr,        // State buffer to scale
    long size,              // Number of elements
    float scale,            // Scale factor
    DevicePtr mtlDevicePtr,
    CError *err
);

// Zero out optimizer state
int zero_optimizer_state(
    GPUPtr statePtr,        // State buffer to zero
    long size,              // Number of elements
    DevicePtr mtlDevicePtr,
    CError *err
);

// Advanced optimizer features

// Compute effective learning rate after all adjustments
int compute_effective_learning_rate(
    float baseLR,           // Base learning rate
    float warmupFactor,     // Warmup factor (1.0 for no warmup)
    float schedulerFactor,  // Scheduler factor
    float *effectiveLR,     // Output: effective learning rate
    CError *err
);

// Learning rate warmup
int perform_lr_warmup(
    float *currentLR,       // Current learning rate (input/output)
    float targetLR,         // Target learning rate after warmup
    long stepNumber,        // Current step number
    long warmupSteps,       // Number of warmup steps
    int warmupType,         // 0: linear, 1: exponential
    CError *err
);

// Parameter statistics for monitoring
int compute_parameter_statistics(
    GPUPtr paramsPtr,       // Parameter tensor
    long size,              // Number of parameters
    float *mean,            // Output: mean of parameters
    float *variance,        // Output: variance of parameters
    float *minVal,          // Output: minimum parameter value
    float *maxVal,          // Output: maximum parameter value
    DevicePtr mtlDevicePtr,
    CError *err
);

// Gradient statistics for monitoring
int compute_gradient_statistics(
    GPUPtr gradPtr,         // Gradient tensor
    long size,              // Number of gradients
    float *mean,            // Output: mean of gradients
    float *variance,        // Output: variance of gradients
    float *minVal,          // Output: minimum gradient value
    float *maxVal,          // Output: maximum gradient value
    float *norm,            // Output: L2 norm of gradients
    DevicePtr mtlDevicePtr,
    CError *err
);

// Memory-efficient operations for large models

// Fused Adam step with gradient accumulation
int perform_fused_adam_step_with_accumulation(
    GPUPtr paramsPtr,       // Model parameters to update
    GPUPtr gradPtr,         // New gradients
    GPUPtr accumulatedGradPtr, // Accumulated gradients buffer
    long size,              // Number of parameters
    float learningRate,     // Learning rate
    float beta1, float beta2, float epsilon, float weightDecay,
    GPUPtr m_bufferPtr, GPUPtr v_bufferPtr,
    long stepNumber,
    float accumulationSteps, // Number of gradient accumulation steps
    int isAccumulationStep, // 1 if this is an accumulation step, 0 if optimizer step
    DevicePtr mtlDevicePtr,
    CError *err
);

// Mixed precision optimizer support
int perform_mixed_precision_sgd_step(
    GPUPtr params_fp32_Ptr,    // FP32 master parameters
    GPUPtr params_fp16_Ptr,    // FP16 working parameters
    GPUPtr grad_fp16_Ptr,      // FP16 gradients
    long size,                 // Number of parameters
    float learningRate,        // Learning rate
    float momentum,            // Momentum coefficient
    GPUPtr momentumBufferPtr,  // Momentum buffer (FP32)
    float weightDecay,         // Weight decay coefficient
    float gradScale,           // Gradient scale factor for mixed precision
    DevicePtr mtlDevicePtr,
    CError *err
);

// Optimizer checkpoint operations
int save_optimizer_checkpoint(
    GPUPtr *stateBuffers,      // Array of optimizer state buffers
    long *bufferSizes,         // Array of buffer sizes
    long numBuffers,           // Number of state buffers
    float *hyperparameters,    // Array of hyperparameters to save
    long numHyperparameters,   // Number of hyperparameters
    char *checkpointPath,      // Path to save checkpoint
    DevicePtr mtlDevicePtr,
    CError *err
);

int load_optimizer_checkpoint(
    GPUPtr *stateBuffers,      // Array of optimizer state buffers to load into
    long *bufferSizes,         // Array of buffer sizes
    long numBuffers,           // Number of state buffers
    float *hyperparameters,    // Array to load hyperparameters into
    long numHyperparameters,   // Number of hyperparameters
    char *checkpointPath,      // Path to load checkpoint from
    DevicePtr mtlDevicePtr,
    CError *err
);

// Memory Pool Management Functions

// Initialize memory pool with maximum size
int initialize_memory_pool(
    long maxSize,           // Maximum memory pool size in bytes
    CError *err
);

// Allocate GPU memory from pool
int allocate_gpu_memory(
    long size,              // Size in bytes to allocate
    GPUPtr *outPtr,         // Output: pointer to allocated memory
    CError *err
);

// Free GPU memory back to pool
int free_gpu_memory(
    GPUPtr ptr,             // Pointer to free
    CError *err
);

// Clean up memory pool
int cleanup_memory_pool(
    CError *err
);

// Memory profiling and monitoring
int get_gpu_memory_usage(
    long *currentUsage,     // Output: current memory usage
    long *peakUsage,        // Output: peak memory usage
    CError *err
);

// Memory compaction and defragmentation
int compact_gpu_memory(
    CError *err
);

// Set memory allocation strategy
int set_memory_allocation_strategy(
    int strategy,           // 0: first fit, 1: best fit, 2: worst fit
    CError *err
);

// New function declarations for Phase 7D - Automatic differentiation helpers

// Fused operations
int perform_fused_conv_bn_relu(
    GPUPtr input, long batch_size, long input_h, long input_w, long input_channels,
    GPUPtr kernel, long kernel_h, long kernel_w, long output_channels,
    GPUPtr gamma, GPUPtr beta, GPUPtr bias,
    long stride_h, long stride_w, long pad_h, long pad_w,
    float epsilon, bool training,
    GPUPtr output, long output_h, long output_w,
    DevicePtr device, CError* error
);

int perform_fused_linear_activation(
    GPUPtr input, long batch_size, long input_size,
    GPUPtr weight, long output_size, GPUPtr bias,
    int activation_type, float alpha,
    GPUPtr output, DevicePtr device, CError* error
);

int perform_fused_attention(
    GPUPtr query, GPUPtr key, GPUPtr value,
    long batch_size, long seq_len, long model_dim,
    int num_heads, float scale, float dropout_rate, bool causal,
    GPUPtr output, DevicePtr device, CError* error
);

int perform_fused_gelu_dropout(
    GPUPtr input, long size, float dropout_rate, unsigned int seed,
    GPUPtr output, DevicePtr device, CError* error
);

int perform_fused_layer_norm_linear(
    GPUPtr input, long batch_size, long seq_len, long input_dim,
    GPUPtr gamma, GPUPtr beta, float epsilon,
    GPUPtr weight, long output_dim, GPUPtr bias,
    GPUPtr output, DevicePtr device, CError* error
);

int perform_fused_residual_block(
    GPUPtr input, long batch_size, long height, long width, long channels,
    GPUPtr conv1_weight, GPUPtr bn1_gamma, GPUPtr bn1_beta,
    GPUPtr conv2_weight, GPUPtr bn2_gamma, GPUPtr bn2_beta,
    float epsilon, GPUPtr output, DevicePtr device, CError* error
);

// Memory management functions
int initialize_memory_pool(long max_memory, CError* error);
int allocate_gpu_memory(long size, GPUPtr* ptr, CError* error);
int free_gpu_memory(GPUPtr ptr, CError* error);
int cleanup_memory_pool(CError* error);

// Gradient operations
int perform_gradient_accumulate(
    GPUPtr existing_grad, GPUPtr new_grad, long size,
    DevicePtr device, CError* error
);

int perform_tensor_sum_squares(
    GPUPtr tensor, long size, float* result,
    DevicePtr device, CError* error
);

int perform_sum_along_axis(
    GPUPtr input, int axis, long input_ndim, long* input_shape,
    GPUPtr output, long output_ndim, long* output_shape,
    DevicePtr device, CError* error
);

int perform_tensor_fill(
    GPUPtr tensor, long size, float value,
    DevicePtr device, CError* error
);

// Dropout operations
int perform_dropout_forward(
    GPUPtr input, long size, float probability, unsigned int seed,
    GPUPtr output, GPUPtr mask, DevicePtr device, CError* error
);

int perform_dropout_backward(
    GPUPtr grad_output, GPUPtr mask, long size, float probability,
    GPUPtr grad_input, DevicePtr device, CError* error
);

#endif // METAL_BRIDGE_H