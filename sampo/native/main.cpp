#include <iostream>
#include "Python.h"

static PyObject* evaluate(PyObject *self, PyObject *args) {
    std::cout << "Hello, World!" << std::endl;
    return nullptr;
}
