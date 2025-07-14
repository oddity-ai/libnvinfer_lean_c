# libnvinfer_lean_c

`libnvinfer_lean_c` is a wrapper around `libnvinfer_lean` that exports C
functions.

The original TensorRT libraries do not export C functions. They can only be
linked to from a C++ application. This library provides an interface to TensorRT
over a C ABI. This makes it easier to use TensorRT from other languages.
