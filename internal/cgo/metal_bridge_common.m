// metal_bridge_common.m - Common utilities and globals for Metal bridge
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