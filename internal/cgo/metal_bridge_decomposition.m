// metal_bridge_decomposition.m - Matrix Decomposition Operations
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
extern void convert_row_to_col_major(float *row_major, float *col_major, long rows, long cols);
extern void convert_col_to_row_major(float *col_major, float *row_major, long rows, long cols);

// --- Matrix Decompositions ---

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

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

#pragma clang diagnostic pop