// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

#include <stdlib.h>
#include <stdbool.h>

typedef unsigned long uint_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long long uint64_t;

struct Device {
	void *       Device;
	bool         Headless;
	bool         LowPower;
	bool         Removable;
	uint64_t     RegistryID;
	const char * Name;
};

struct Size {
	uint_t Width;
	uint_t Height;
	uint_t Depth;
};

struct Library {
	void *       Library;
	const char * Error;
};
struct CompileOption {
	uint_t languageVersion;
};

struct ComputePipelineState {
	void *       ComputePipelineState;
	const char * Error;
};

struct Device CreateSystemDefaultDevice();
struct Library Device_MakeLibrary(void * device, const char *source, struct CompileOption opt);
struct ComputePipelineState Device_MakeComputePipelineState(void *device, void *function);

// CommandQueue
void *Device_MakeCommandQueue(void *device) ;
void *CommandQueue_MakeCommandBuffer(void *commandQueue);
void  CommandQueue_Release(void *commandQueue);
// CommandEncoder, ComputeCommandEncoder
void  CommandEncoder_EndEncoding(void *commandEncoder);
void *CommandBuffer_MakeComputeCommandEncoder(void *commandBuffer);
void  ComputeCommandEncoder_SetComputePipelineState(void *computeCommandEncoder, void * computePipelineState);
void  ComputeCommandEncoder_SetBytes(void *computeCommandEncoder, void *bytes, int length, int index);
void  ComputeCommandEncoder_SetBuffer(void *computeCommandEncoder, void *buffer, int offset, int index);
void  ComputeCommandEncoder_DispatchThreads(void *computeCommandEncoder, struct Size threadsPerGrid, struct Size threadsPerThreadgroup);
// CommandBuffer
void  CommandBuffer_WaitUntilCompleted(void * commandBuffer);
void  CommandBuffer_Commit(void * commandBuffer);
void  CommandBuffer_Release(void *commandBuffer);
// Buffer
void *Device_MakeBuffer(void *device, const void *bytes, size_t length, uint16_t options);
void *Buffer_Content(void *buffer);
void  Buffer_Release(void *buffer);
// Function
void *Library_MakeFunction(void *library, const char *name);