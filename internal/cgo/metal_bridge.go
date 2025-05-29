package cgo

/*
#cgo CFLAGS: -x objective-c -ObjC -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework Accelerate -framework MetalPerformanceShaders
#include "metal_bridge.h"
*/
import "C"

// This package ensures that the metal_bridge.m file gets compiled by CGO.