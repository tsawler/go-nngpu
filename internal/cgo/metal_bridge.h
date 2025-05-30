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

// Function to free error message allocated by Objective-C
void free_c_error_message(char *message);

#endif // METAL_BRIDGE_H