OCR
===

# Installation

The CMake Build System is enabled. If you already know how
that works, you can go ahead and do the following:

```bash

mkdir build;
cd build;
cmake ..;
make 

```

## Generated Files

* `libnnl.so` - Graves' neural net library.
* `libocr.so` - OCR implemented as a facade to `libnnl`.
* `_pyocr.so`, `ocr.py` - Python module and shared library
  generated for interface.




