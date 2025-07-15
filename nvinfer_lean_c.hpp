#ifndef nvinfer_lean_c
#define nvinfer_lean_c

#include <NvInfer.h>

//!
//! \enum ExecutionContextAllocationStrategy
//!
//! \brief Different memory allocation behaviors for IExecutionContext.
//!
//! IExecutionContext requires a block of device memory for internal activation tensors during inference. The user can
//! either let the execution context manage the memory in various ways or allocate the memory themselves.
//!
//! \see ICudaEngine::createExecutionContext()
//! \see IExecutionContext::setDeviceMemory()
//!
typedef enum nvinfer1_ExecutionContextAllocationStrategy
{
    kSTATIC = 0,            //!< Default static allocation with the maximum size across all profiles.
    kON_PROFILE_CHANGE = 1, //!< Reallocate for a profile when it's selected.
    kUSER_MANAGED = 2,      //!< The user supplies custom allocation to the execution context.
} nvinfer1_ExecutionContextAllocationStrategy;

//!
//! \enum TensorIOMode
//!
//! \brief Definition of tensor IO Mode.
//!
typedef enum nvinfer1_TensorIOMode
{
    kNONE = 0,   //! Tensor is not an input or output.
    kINPUT = 1,  //! Tensor is input to the engine.
    kOUTPUT = 2, //! Tensor is output by the engine.
} nvinfer1_TensorIOMode;

//!
//! \enum Severity
//!
//! \brief The severity corresponding to a log message.
//!
typedef enum nvinfer1_Severity
{
    kINTERNAL_ERROR = 0, //! An internal error has occurred. Execution is unrecoverable.
    kERROR = 1,          //! An application error has occurred.
    kWARNING = 2, //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
    kINFO = 3,    //! Informational messages with instructional information.
    kVERBOSE = 4, //! Verbose messages with debugging information.
} nvinfer1_Severity;

//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, refitter and runtime.
//!
//! The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects created through
//! that interface. The logger must be valid until all objects created are released.
//!
//! The Logger object implementation must be thread safe. All locking and synchronization is pushed to the
//! interface implementation and TensorRT does not hold any synchronization primitives when calling the interface
//! functions.
//!
struct nvinfer1_ILogger;

//!
//! \class Logger
//!
//! \brief Implements a global ILogger that allows setting the callback using setCallback.
//!
class Logger : public nvinfer1::ILogger
{
  public:
    //!
    //! \brief Log a message with a given severity.
    //!
    //! \param severity The severity of the log message.
    //! \param msg The message to log.
    //!
    void log(nvinfer1_Severity severity, const char *msg) noexcept override
    {
        if (callback)
        {
            callback(severity, msg);
        }
    }

    //!
    //! \brief Set a custom logging callback function.
    //!
    //! \param callback A C-style function pointer to handle logging.
    //!           The function should accept nvinfer1_Severity and const char* as parameters.
    //!
    //! By default, the callback does nothing.
    //!
    void setCallback(void (*callback)(nvinfer1_Severity, const char *)) noexcept
    {
        callback = callback;
    }

  private:
    void (*callback)(nvinfer1_Severity, const char *) = nullptr; //!< Pointer to the custom logging callback function.
};

//!
//! \brief Global logger instance.
//!
Logger LOGGER;

//!
//! \class ICudaEngine
//!
//! \brief An engine for executing inference on a built network, with functionally unsafe features.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
struct nvinfer1_ICudaEngine;

//!
//! \class IExecutionContext
//!
//! \brief Context for executing inference using an engine, with functionally unsafe features.
//!
//! Multiple execution contexts may exist for one ICudaEngine instance, allowing the same
//! engine to be used for the execution of multiple batches simultaneously. If the engine supports
//! dynamic shapes, each execution context in concurrent use must use a separate optimization profile.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
struct nvinfer1_IExecutionContext;

//!
//! \class IRuntime
//!
//! \brief Allows a serialized functionally unsafe engine to be deserialized.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
struct nvinfer1_IRuntime;

//!
//! \brief Create an execution context and specify the strategy for allocating internal activation memory.
//!
//! The default value for the allocation strategy is ExecutionContextAllocationStrategy::kSTATIC, which means the
//! context will pre-allocate a block of device memory that is sufficient for all profiles. The newly created
//! execution context will be assigned optimization profile 0. If an error recorder has been set for the engine, it
//! will also be passed to the execution context.
//!
//! \see IExecutionContext
//! \see IExecutionContext::setOptimizationProfileAsync()
//! \see ExecutionContextAllocationStrategy
//!
extern "C" nvinfer1_IExecutionContext *nvinfer1_ICudaEngine_createExecutionContext(
    nvinfer1_ICudaEngine *engine, nvinfer1_ExecutionContextAllocationStrategy strategy);

//!
//! \brief Return the shape of the given input or output.
//!
//! \param tensorName The name of an input or output tensor.
//!
//! Return Dims{-1, {}} if the provided name does not map to an input or output tensor.
//! Otherwise return the shape of the input or output tensor.
//!
//! A dimension in an input tensor will have a -1 wildcard value if all the following are true:
//!  * setInputShape() has not yet been called for this tensor
//!  * The dimension is a runtime dimension that is not implicitly constrained to be a single value.
//!
//! A dimension in an output tensor will have a -1 wildcard value if the dimension depends
//! on values of execution tensors OR if all the following are true:
//!  * It is a runtime dimension.
//!  * setInputShape() has NOT been called for some input tensor(s) with a runtime shape.
//!  * setTensorAddress() has NOT been called for some input tensor(s) with isShapeInferenceIO() = true.
//!
//! An output tensor may also have -1 wildcard dimensions if its shape depends on values of tensors supplied to
//! enqueueV3().
//!
//! If the request is for the shape of an output tensor with runtime dimensions,
//! all input tensors with isShapeInferenceIO() = true should have their value already set,
//! since these values might be needed to compute the output shape.
//!
//! Examples of an input dimension that is implicitly constrained to a single value:
//! * The optimization profile specifies equal min and max values.
//! * The dimension is named and only one value meets the optimization profile requirements
//!   for dimensions with that name.
//!
//! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
//!
extern "C" nvinfer1::Dims nvinfer1_ICudaEngine_getTensorShape(const nvinfer1_ICudaEngine *engine,
                                                              const char *tensorName);

//!
//! \brief Determine whether a tensor is an input or output tensor.
//!
//! \param tensorName The name of an input or output tensor.
//!
//! \return kINPUT if tensorName is an input, kOUTPUT if tensorName is an output, or kNONE if neither.
//!
//! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
//!
extern "C" nvinfer1_TensorIOMode nvinfer1_ICudaEngine_getTensorIOMode(const nvinfer1_ICudaEngine *engine,
                                                                      const char *tensorName);

//!
//! \brief Return number of IO tensors.
//!
//! It is the number of input and output tensors for the network from which the engine was built.
//! The names of the IO tensors can be discovered by calling getIOTensorName(i) for i in 0 to getNbIOTensors()-1.
//!
//! \see getIOTensorName()
//!
extern "C" int32_t nvinfer1_ICudaEngine_getNbIOTensors(const nvinfer1_ICudaEngine *engine);

//!
//! \brief Return name of an IO tensor.
//!
//! \param index value between 0 and getNbIOTensors()-1
//!
//! \see getNbIOTensors()
//!
extern "C" const char *nvinfer1_ICudaEngine_getIOTensorName(const nvinfer1_ICudaEngine *engine, int32_t index);

//!
//! \brief Destroy the ICudaEngine instance.
//!
extern "C" void nvinfer1_ICudaEngine_destroy(nvinfer1_ICudaEngine *engine);

//!
//! \brief Set memory address for given input or output tensor.
//!
//! \param tensorName The name of an input or output tensor.
//! \param data The pointer (void*) to the data owned by the user.
//!
//! \return True on success, false if error occurred.
//!
//! An address defaults to nullptr.
//! Pass data=nullptr to reset to the default state.
//!
//! Return false if the provided name does not map to an input or output tensor.
//!
//! If an input pointer has type (void const*), use setInputTensorAddress() instead.
//!
//! Before calling enqueueV3(), each input must have a non-null address and
//! each output must have a non-null address or an IOutputAllocator to set it later.
//!
//! If the TensorLocation of the tensor is kHOST:
//! - The pointer must point to a host buffer of sufficient size.
//! - Data representing shape values is not copied until enqueueV3 is invoked.
//!
//! If the TensorLocation of the tensor is kDEVICE:
//! - The pointer must point to a device buffer of sufficient size and alignment, or
//! - Be nullptr if the tensor is an output tensor that will be allocated by IOutputAllocator.
//!
//! If getTensorShape(name) reports a -1 for any dimension of an output after all
//! input shapes have been set, use setOutputAllocator() to associate an IOutputAllocator
//! to which the dimensions will be reported when known.
//!
//! Calling both setTensorAddress and setOutputAllocator() for the same output is allowed,
//! and can be useful for preallocating memory, and then reallocating if it's not big enough.
//!
//! The pointer must have at least 256-byte alignment.
//!
//! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
//!
//! \see setInputTensorAddress() setOutputTensorAddress() getTensorShape() setOutputAllocator() IOutputAllocator
//!
extern "C" bool nvinfer1_IExecutionContext_setTensorAddress(nvinfer1_IExecutionContext *context, const char *tensorName,
                                                            void *data);

//!
//! \brief Enqueue inference on a stream.
//!
//! \param stream A CUDA stream on which the inference kernels will be enqueued.
//!
//! \return True if the kernels were enqueued successfully, false otherwise.
//!
//! Modifying or releasing memory that has been registered for the tensors before stream
//! synchronization or the event passed to setInputConsumedEvent has been being triggered results in undefined
//! behavior.
//! Input tensor can be released after the setInputConsumedEvent whereas output tensors require stream
//! synchronization.
//!
//! \warning Using default stream may lead to performance issues due to additional cudaDeviceSynchronize() calls by
//!          TensorRT to ensure correct synchronizations. Please use non-default stream instead.
//!
//! \warning If the Engine is streaming weights, enqueueV3 will become synchronous, and
//!          the graph will not be capturable.
//!
extern "C" bool nvinfer1_IExecutionContext_enqueueV3(nvinfer1_IExecutionContext *context, cudaStream_t stream);

//!
//! \brief Destroy the IExecutionContext instance.
//!
extern "C" void nvinfer1_IExecutionContext_destroy(nvinfer1_IExecutionContext *context);

//!
//! \brief Create an instance of an IRuntime class.
//!
//! \param logger The logging class for the runtime.
//!
extern "C" nvinfer1_IRuntime *nvinfer1_createInferRuntime(nvinfer1_ILogger *logger);

//!
//! \brief Deserialize an engine from host memory.
//!
//! If an error recorder has been set for the runtime, it will also be passed to the engine.
//!
//! \warning Destroying the IRuntime before destroying all associated ICudaEngine instances results in undefined
//! behavior.
//!
//! \param blob The memory that holds the serialized engine.
//! \param size The size of the memory.
//!
//! \return The engine, or nullptr if it could not be deserialized.
//!
extern "C" nvinfer1_ICudaEngine *nvinfer1_IRuntime_deserializeCudaEngine(nvinfer1_IRuntime *runtime, const void *blob,
                                                                         size_t size);

//!
//! \brief Destroy the IRuntime instance.
//!
extern "C" void nvinfer1_IRuntime_destroy(nvinfer1_IRuntime *runtime);

//!
//! \brief Set the callback function for the LOGGER.
//!
//! \param callback A C-style function pointer to handle logging.
//!                 The function should accept nvinfer1_Severity and const char* as parameters.
//!
extern "C" void nvinfer1_setLoggerCallback(void (*callback)(nvinfer1_Severity, const char *));

#endif // nvinfer_lean_c
