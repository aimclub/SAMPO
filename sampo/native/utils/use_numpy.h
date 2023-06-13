//
// Created by stasb on 29.05.2023.
//

#ifndef NATIVE_USE_NUMPY_H
#define NATIVE_USE_NUMPY_H

//your fancy name for the dedicated PyArray_API-symbol
#define PY_ARRAY_UNIQUE_SYMBOL MY_PyArray_API

//this macro must be defined for the translation unit
#ifndef INIT_NUMPY_ARRAY_CPP
    #define NO_IMPORT_ARRAY //for usual translation units
#endif

//now, everything is setup, just include the numpy-arrays:
#include <numpy/arrayobject.h>

#endif //NATIVE_USE_NUMPY_H
