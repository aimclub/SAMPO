#include <iostream>
#include "Python.h"

// GLOBAL TODOS
// TODO Make all classes with encapsulation - remove public fields
// TODO Make deserializer
// TODO Make parallel runtime
// TODO Performance measurements
// TODO Cache data in C++ memory - parse Python WG and Contractors once per-scheduling
// TODO Split data types' definition and implementation

static PyObject* evaluate(PyObject *self, PyObject *args) {
    std::cout << "Hello, World!" << std::endl;
    return nullptr;
}
