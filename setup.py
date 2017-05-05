#!/usr/bin/env python

"""
setup.py file for SWIG api
"""

from distutils.core import setup, Extension


api_module = Extension('_ocr',
                           sources=['api_wrap.cxx', "api.cpp"],
                           )

setup (name = 'ocr',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig api from docs""",
       ext_modules = [api_module],
       py_modules = ["ocr"],
       )

