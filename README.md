# libnvinfer_lean_c

`libnvinfer_lean_c` is a replacement for normal `libnvinfer_lean` that contains
C exported functions for part of the API.

The reason this is useful is because the original TensorRT distribution does not
export C symbols. This makes foreign function interfaces a real pain to
implement, since TensorRT is a C++ library and C++ does not have a stable ABI.

The solution is a new C++ library that wholly contains the original
`libnvinfer_lean_static` (statically linked) and re-exports a portion of the API
as C functions. The library itself is a shared library and can be even be loaded
at runtime with `dlopen`.
