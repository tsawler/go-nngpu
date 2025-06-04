// metal_bridge_convolution.m - Convolution-related functions extracted from metal_bridge.m
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

// External declarations for global variables and helper functions
extern DevicePtr _global_mtl_device_ptr;
extern CommandQueuePtr _global_mtl_command_queue_ptr;

// External helper function declarations
extern void set_c_error_message(CError *err, NSString *format, ...);

// --- Convolution Operations ---

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