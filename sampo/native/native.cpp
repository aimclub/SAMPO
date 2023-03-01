#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include "native.h"
#include "workgraph.h"
#include "contractor.h"
#include "python_deserializer.h"
#include "pycodec.h"
#include "chromosome_evaluator.h"

#include <iostream>

// GLOBAL TODOS
// TODO Make all classes with encapsulation - remove public fields
// TODO Make parallel runtime
// TODO Performance measurements
// TODO Cache data in C++ memory - parse Python WG and Contractors once per-scheduling
// TODO Split data types' definition and implementation

static inline int PyLong_AsInt(PyObject* object) {
    return (int) PyLong_AsLong(object);
}

static inline vector<int> decodeIntList(PyObject* object) {
    return PyCodec::fromList(object, PyLong_AsInt)
}

static inline PyObject* pyObjectIdentity(PyObject* object) {
    return object;
}

static PyObject* evaluate(PyObject *self, PyObject *args) {
    PyObject* it = PyObject_GetIter(args);
    if (it == nullptr) {
        return nullptr;
    }
    auto pythonWrapper = it = PyIter_Next(it);
    auto parents = PyCodec::fromList(it = PyIter_Next(it), decodeIntList);
    auto inseparables = PyCodec::fromList(it = PyIter_Next(it), decodeIntList);
    auto workers = PyCodec::fromList(it = PyIter_Next(it), decodeIntList);
    int totalWorksCount = PyLong_AsInt(it = PyIter_Next(it));
    auto chromosomes = PyCodec::fromList(it = PyIter_Next(it), pyObjectIdentity);

    Py_DECREF(it);
    auto evaluator = ChromosomeEvaluator(parents, inseparables, workers, totalWorksCount, pythonWrapper);

    vector<int> results = evaluator.evaluate(chromosomes);

    PyObject* pyList = PyList_New(results.size());
    for (int i = 0; i < results.size(); i++) {
        PyList_SetItem(pyList, i, Py_BuildValue("i", results[i]));
    }
    return pyList;
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
    assert(! PyErr_Occurred());
    // Initialise Numpy
    import_array()
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&nativeModule);
}

int main() {
    return 0;
}
