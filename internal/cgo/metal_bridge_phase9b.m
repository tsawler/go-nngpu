#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "metal_bridge.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/task_info.h>

// Phase 9B: Real Metal Bridge Implementation

// Command Queue Management
void* createCommandQueue(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) {
            NSLog(@"Error: Invalid Metal device in createCommandQueue");
            return NULL;
        }
        
        // Create command queue with optimal configuration
        id<MTLCommandQueue> commandQueue = [mtlDevice newCommandQueue];
        if (!commandQueue) {
            NSLog(@"Error: Failed to create command queue");
            return NULL;
        }
        
        // Set label for debugging
        commandQueue.label = @"go-nngpu.commandQueue";
        
        // Retain and return as bridged pointer
        return (__bridge_retained void*)commandQueue;
    }
}

// Synchronize command queue
void synchronizeCommandQueue(void* queue) {
    @autoreleasepool {
        if (!queue) return;
        
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // Create command buffer and commit to ensure synchronization
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        commandBuffer.label = @"Synchronization";
        
        // Add completion handler for true synchronization
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buffer) {
            // Synchronization complete
        }];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// Create Metal event for synchronization
void* createMetalEvent(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) {
            NSLog(@"Error: Invalid Metal device in createMetalEvent");
            return NULL;
        }
        
        // Create shared event for CPU-GPU synchronization
        id<MTLSharedEvent> event = [mtlDevice newSharedEvent];
        if (!event) {
            NSLog(@"Error: Failed to create Metal event");
            return NULL;
        }
        
        event.label = @"go-nngpu.syncEvent";
        
        return (__bridge_retained void*)event;
    }
}

// Create MPS Graph
void* createMPSGraph(void) {
    @autoreleasepool {
        if (@available(macOS 11.0, *)) {
            MPSGraph *graph = [[MPSGraph alloc] init];
            if (!graph) {
                NSLog(@"Error: Failed to create MPS Graph");
                return NULL;
            }
            
            return (__bridge_retained void*)graph;
        } else {
            NSLog(@"Error: MPS Graph requires macOS 11.0 or later");
            return NULL;
        }
    }
}

// Unified Memory Management

// Allocate shared memory accessible by both CPU and GPU
void* allocateSharedMemory(long size) {
    @autoreleasepool {
        if (size <= 0) {
            NSLog(@"Error: Invalid size for shared memory allocation: %ld", size);
            return NULL;
        }
        
        // Use posix_memalign for page-aligned memory
        void* memory = NULL;
        size_t alignment = 16384; // 16KB alignment for optimal performance
        int result = posix_memalign(&memory, alignment, size);
        
        if (result != 0 || !memory) {
            NSLog(@"Error: Failed to allocate aligned memory of size %ld", size);
            return NULL;
        }
        
        // Touch pages to ensure allocation
        memset(memory, 0, size);
        
        return memory;
    }
}

// Create GPU buffer from shared memory
void* createGPUBufferFromSharedMemory(void* device, void* sharedMemory, long size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice || !sharedMemory || size <= 0) {
            NSLog(@"Error: Invalid parameters for GPU buffer creation");
            return NULL;
        }
        
        // Create buffer with shared storage mode
        id<MTLBuffer> buffer = [mtlDevice newBufferWithBytesNoCopy:sharedMemory
                                                             length:size
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
        
        if (!buffer) {
            NSLog(@"Error: Failed to create GPU buffer from shared memory");
            return NULL;
        }
        
        buffer.label = @"go-nngpu.sharedBuffer";
        
        return (__bridge_retained void*)buffer;
    }
}

// Make buffer resident on GPU
void makeBufferGPUResident(void* buffer) {
    @autoreleasepool {
        if (!buffer) return;
        
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        
        // On Apple Silicon, all buffers are automatically resident
        // This is a no-op but kept for API compatibility
        // In the future, this could handle resource hints
    }
}

// Memory-mapped file support
void* memoryMapFile(const char* path, long* size, CError* error) {
    @autoreleasepool {
        if (!path || !size) {
            if (error) {
                error->message = strdup("Invalid parameters for memory mapping");
            }
            return NULL;
        }
        
        NSString *filePath = [NSString stringWithUTF8String:path];
        NSError *nsError = nil;
        
        // Get file size
        NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:&nsError];
        if (nsError) {
            if (error) {
                error->message = strdup([[nsError localizedDescription] UTF8String]);
            }
            return NULL;
        }
        
        NSNumber *fileSize = attributes[NSFileSize];
        *size = [fileSize longValue];
        
        // Open file for memory mapping
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            if (error) {
                error->message = strdup("Failed to open file for memory mapping");
            }
            return NULL;
        }
        
        // Memory map the file
        void* mappedMemory = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        
        if (mappedMemory == MAP_FAILED) {
            if (error) {
                error->message = strdup("Failed to memory map file");
            }
            return NULL;
        }
        
        return mappedMemory;
    }
}

// Unmap memory
void unmapMemory(void* memory, long size) {
    if (memory && size > 0) {
        munmap(memory, size);
    }
}

// Performance Monitoring

// Get GPU performance counters
void getGPUCounters(GPUCounters* counters) {
    @autoreleasepool {
        if (!counters) return;
        
        // Initialize default values
        counters->computeUtilization = 0.0;
        counters->memoryBandwidth = 0.0;
        counters->cacheHitRate = 0.0;
        
        // Note: Real GPU counters require private APIs or Metal System Trace
        // This is a simplified implementation
        
        // Get process info for basic metrics
        struct task_basic_info info;
        mach_msg_type_number_t size = sizeof(info);
        kern_return_t kerr = task_info(mach_task_self(),
                                      TASK_BASIC_INFO,
                                      (task_info_t)&info,
                                      &size);
        
        if (kerr == KERN_SUCCESS) {
            // Estimate based on process metrics
            counters->computeUtilization = (float)info.resident_size / (1024.0 * 1024.0 * 1024.0);
        }
    }
}

// Get GPU memory usage
void getGPUMemoryUsage(long* totalBytes, long* usedBytes) {
    @autoreleasepool {
        if (!totalBytes || !usedBytes) return;
        
        // Get recommended working set size (approximation)
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            *totalBytes = device.recommendedMaxWorkingSetSize;
            
            // Estimate used memory from current allocations
            struct task_vm_info vmInfo;
            mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
            kern_return_t result = task_info(mach_task_self(),
                                           TASK_VM_INFO,
                                           (task_info_t)&vmInfo,
                                           &count);
            
            if (result == KERN_SUCCESS) {
                *usedBytes = vmInfo.phys_footprint;
            } else {
                *usedBytes = 0;
            }
        } else {
            *totalBytes = 0;
            *usedBytes = 0;
        }
    }
}

// Get GPU power usage (simplified)
float getGPUPowerUsage(void) {
    // Note: Real power monitoring requires IOKit or private APIs
    // Return a placeholder value
    return 0.0f;
}

// Get GPU temperature (simplified)
float getGPUTemperature(void) {
    // Note: Real temperature monitoring requires IOKit or private APIs
    // Return a placeholder value
    return 0.0f;
}

// MPS Graph Operations

// Create matrix multiplication node
void* createMatMulNode(void* graph, void* inputA, void* inputB) {
    @autoreleasepool {
        if (@available(macOS 11.0, *)) {
            MPSGraph *mpsGraph = (__bridge MPSGraph*)graph;
            MPSGraphTensor *tensorA = (__bridge MPSGraphTensor*)inputA;
            MPSGraphTensor *tensorB = (__bridge MPSGraphTensor*)inputB;
            
            if (!mpsGraph || !tensorA || !tensorB) {
                NSLog(@"Error: Invalid parameters for matmul node");
                return NULL;
            }
            
            MPSGraphTensor *result = [mpsGraph matrixMultiplicationWithPrimaryTensor:tensorA
                                                                    secondaryTensor:tensorB
                                                                               name:@"MatMul"];
            
            return (__bridge_retained void*)result;
        } else {
            return NULL;
        }
    }
}

// Create convolution node
void* createConv2DNode(void* graph, void* input, void* weights, long strideH, long strideW, long padH, long padW) {
    @autoreleasepool {
        if (@available(macOS 11.0, *)) {
            MPSGraph *mpsGraph = (__bridge MPSGraph*)graph;
            MPSGraphTensor *inputTensor = (__bridge MPSGraphTensor*)input;
            MPSGraphTensor *weightsTensor = (__bridge MPSGraphTensor*)weights;
            
            if (!mpsGraph || !inputTensor || !weightsTensor) {
                NSLog(@"Error: Invalid parameters for conv2d node");
                return NULL;
            }
            
            // Create convolution descriptor
            MPSGraphConvolution2DOpDescriptor *descriptor = [[MPSGraphConvolution2DOpDescriptor alloc] init];
            descriptor.strideInX = strideW;
            descriptor.strideInY = strideH;
            descriptor.paddingLeft = padW;
            descriptor.paddingRight = padW;
            descriptor.paddingTop = padH;
            descriptor.paddingBottom = padH;
            descriptor.paddingStyle = MPSGraphPaddingStyleExplicit;
            descriptor.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
            descriptor.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
            
            MPSGraphTensor *result = [mpsGraph convolution2DWithSourceTensor:inputTensor
                                                              weightsTensor:weightsTensor
                                                                 descriptor:descriptor
                                                                       name:@"Conv2D"];
            
            return (__bridge_retained void*)result;
        } else {
            return NULL;
        }
    }
}

// Create ReLU activation node
void* createReLUNode(void* graph, void* input) {
    @autoreleasepool {
        if (@available(macOS 11.0, *)) {
            MPSGraph *mpsGraph = (__bridge MPSGraph*)graph;
            MPSGraphTensor *inputTensor = (__bridge MPSGraphTensor*)input;
            
            if (!mpsGraph || !inputTensor) {
                NSLog(@"Error: Invalid parameters for ReLU node");
                return NULL;
            }
            
            MPSGraphTensor *result = [mpsGraph reLUWithTensor:inputTensor name:@"ReLU"];
            
            return (__bridge_retained void*)result;
        } else {
            return NULL;
        }
    }
}

// Compile MPS Graph
void* compileGraph(void* graph, void* device) {
    @autoreleasepool {
        if (@available(macOS 11.0, *)) {
            MPSGraph *mpsGraph = (__bridge MPSGraph*)graph;
            id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
            
            if (!mpsGraph || !mtlDevice) {
                NSLog(@"Error: Invalid parameters for graph compilation");
                return NULL;
            }
            
            // Note: Full graph compilation requires more setup
            // For now, return the graph itself as a placeholder
            NSLog(@"Graph compilation placeholder - requires full implementation");
            
            return (__bridge_retained void*)mpsGraph;
        } else {
            return NULL;
        }
    }
}

// Execute compiled graph
void executeCompiledGraph(void* compiledGraph, void* commandBuffer, void** inputs, long numInputs, void** outputs, long numOutputs) {
    @autoreleasepool {
        if (@available(macOS 12.0, *)) {
            // Simplified execution - full implementation would handle tensor mappings
            NSLog(@"Graph execution placeholder - requires full implementation");
        }
    }
}

// Stream Management Helper Functions

// Get number of available GPU cores
int getGPUCoreCount(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) return 0;
        
        // Note: This is an approximation based on device family
        NSString *deviceName = mtlDevice.name;
        
        // M4 Pro: 20 GPU cores, M4 Max: 40 GPU cores
        if ([deviceName containsString:@"M4 Max"]) {
            return 40;
        } else if ([deviceName containsString:@"M4 Pro"]) {
            return 20;
        } else if ([deviceName containsString:@"M4"]) {
            return 10; // Base M4
        }
        
        // Default for unknown devices
        return 8;
    }
}

// Create optimized command queue with priority
void* createPriorityCommandQueue(void* device, int priority) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) return NULL;
        
        // Note: Priority queues require specific Metal features
        // For now, create standard queue with label indicating priority
        id<MTLCommandQueue> commandQueue = [mtlDevice newCommandQueue];
        if (commandQueue) {
            NSString *priorityStr = @"normal";
            switch (priority) {
                case 0:
                    priorityStr = @"background";
                    break;
                case 2:
                    priorityStr = @"high";
                    break;
            }
            commandQueue.label = [NSString stringWithFormat:@"go-nngpu.%@PriorityQueue", priorityStr];
        }
        
        return (__bridge_retained void*)commandQueue;
    }
}

// Batch command buffer execution
void executeBatchCommands(void* queue, void** commandBuffers, long count) {
    @autoreleasepool {
        if (!queue || !commandBuffers || count <= 0) return;
        
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // Commit all command buffers
        for (long i = 0; i < count; i++) {
            if (commandBuffers[i]) {
                id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)commandBuffers[i];
                [buffer commit];
            }
        }
        
        // Wait for the last one to complete (ensures all are done)
        if (commandBuffers[count-1]) {
            id<MTLCommandBuffer> lastBuffer = (__bridge id<MTLCommandBuffer>)commandBuffers[count-1];
            [lastBuffer waitUntilCompleted];
        }
    }
}