#include "nvinfer_lean_c.hpp"

// ICudaEngine

nvinfer1_IExecutionContext *nvinfer1_ICudaEngine_createExecutionContext(nvinfer1_ICudaEngine *engine,
                                                                  nvinfer1_ExecutionContextAllocationStrategy strategy)
{
    return reinterpret_cast<nvinfer1_IExecutionContext *>(
        reinterpret_cast<nvinfer1::ICudaEngine *>(engine)->createExecutionContext(
            static_cast<nvinfer1::ExecutionContextAllocationStrategy>(strategy)));
}

nvinfer1::Dims nvinfer1_ICudaEngine_getTensorShape(const nvinfer1_ICudaEngine *engine, const char *tensorName)
{
    return reinterpret_cast<const nvinfer1::ICudaEngine *>(engine)->getTensorShape(tensorName);
}

nvinfer1_TensorIOMode nvinfer1_ICudaEngine_getTensorIOMode(const nvinfer1_ICudaEngine *engine, const char *tensorName)
{
    return static_cast<nvinfer1_TensorIOMode>(
        reinterpret_cast<const nvinfer1::ICudaEngine *>(engine)->getTensorIOMode(tensorName));
}

int32_t nvinfer1_ICudaEngine_getNbIOTensors(const nvinfer1_ICudaEngine *engine)
{
    return reinterpret_cast<const nvinfer1::ICudaEngine *>(engine)->getNbIOTensors();
}

const char *nvinfer1_ICudaEngine_getIOTensorName(const nvinfer1_ICudaEngine *engine, int32_t index)
{
    return reinterpret_cast<const nvinfer1::ICudaEngine *>(engine)->getIOTensorName(index);
}

void nvinfer1_ICudaEngine_destroy(nvinfer1_ICudaEngine *engine)
{
    delete reinterpret_cast<nvinfer1::ICudaEngine *>(engine);
}

// IExecutionContext

bool nvinfer1_IExecutionContext_setTensorAddress(nvinfer1_IExecutionContext *context, const char *tensorName,
                                                 void *data)
{
    return reinterpret_cast<nvinfer1::IExecutionContext *>(context)->setTensorAddress(tensorName, data);
}

bool nvinfer1_IExecutionContext_enqueueV3(nvinfer1_IExecutionContext *context, cudaStream_t stream)
{
    return reinterpret_cast<nvinfer1::IExecutionContext *>(context)->enqueueV3(stream);
}

void nvinfer1_IExecutionContext_destroy(nvinfer1_IExecutionContext *context)
{
    delete reinterpret_cast<nvinfer1::IExecutionContext *>(context);
}

// IRuntime

nvinfer1_ICudaEngine *nvinfer1_IRuntime_deserializeCudaEngine(nvinfer1_IRuntime *runtime, const void *blob, size_t size)
{
    return reinterpret_cast<nvinfer1_ICudaEngine *>(
        reinterpret_cast<nvinfer1::IRuntime *>(runtime)->deserializeCudaEngine(blob, size));
}

void nvinfer1_IRuntime_destroy(nvinfer1_IRuntime *runtime)
{
    delete reinterpret_cast<nvinfer1::IRuntime *>(runtime);
}
