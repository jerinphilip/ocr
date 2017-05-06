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


# Interfaces

## Python

Post building, put `ports/python/test.py` inside the same folder as
`pyocr.py` and `_pyocr.so`.

In the directory run

```bash
    LD_LIBRARY_PATH=. python test.py
```

Take a look at `test.py`, for figuring out how to use the API.

