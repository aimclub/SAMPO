#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "native.h"
#include "workgraph.h"
#include "contractor.h"
#include "python_deserializer.h"

#include <iostream>

// GLOBAL TODOS
// TODO Make all classes with encapsulation - remove public fields
// TODO Make parallel runtime
// TODO Performance measurements
// TODO Cache data in C++ memory - parse Python WG and Contractors once per-scheduling
// TODO Split data types' definition and implementation

static PyObject* evaluate(PyObject *self, PyObject *args) {
    WorkGraph* workGraph = PythonDeserializer::workGraph(PyTuple_GetItem(args, 1));
    vector<Contractor*> contractors = PythonDeserializer::contractors(PyTuple_GetItem(args, 2));


    return nullptr;
}

static PyMethodDef nativeMethods[] = {
        { "evaluate", evaluate, METH_VARARGS,
              "Evaluates the chromosome using Just-In-Time-Timeline" },
        {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef nativeModule = {
        PyModuleDef_HEAD_INIT,
        "native",
        "The high-efficient native implementation of sampo modules",
        -1,
        nativeMethods
};

PyMODINIT_FUNC
PyInit_native(void) {
    import_numpy();
    return PyModule_Create(&nativeModule);
}

int main() {
    return 0;
}