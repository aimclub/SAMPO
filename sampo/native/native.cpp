#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#include "chromosome_evaluator.h"
#include "pycodec.h"

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

static vector<int> decodeIntList(PyObject* object) {
    return PyCodec::fromList(object, PyLong_AsInt);
}

static PyObject* pyObjectIdentity(PyObject* object) {
    return object;
}

static PyObject* evaluate(PyObject *self, PyObject *args) {
//    cerr << "Called" << endl;

    PyObject* pythonWrapper;
    PyObject* pyParents;
    PyObject* pyInseparables;
    PyObject* pyWorkers;
    int totalWorksCount;
    PyObject* pyChromosomes;

    if (!PyArg_ParseTuple(args, "OOOOiO",
                          &pythonWrapper, &pyParents, &pyInseparables,
                          &pyWorkers, &totalWorksCount, &pyChromosomes)) {
        cout << "Can't parse arguments" << endl;
    }
    auto parents = PyCodec::fromList(pyParents, decodeIntList);
    auto inseparables = PyCodec::fromList(pyInseparables, decodeIntList);
    auto workers = PyCodec::fromList(pyWorkers, decodeIntList);
    auto chromosomes = PyCodec::fromList(pyChromosomes, pyObjectIdentity);

//    cout << "Called" << endl << flush;

    ChromosomeEvaluator evaluator(parents, inseparables, workers, totalWorksCount, pythonWrapper);

    vector<int> results = evaluator.evaluate(chromosomes);

//    return PyLong_FromLong(evaluator.evaluate(chromosomes[0]));
    PyObject* pyList = PyList_New(results.size());
    for (int i = 0; i < results.size(); i++) {
        PyObject* pyInt = Py_BuildValue("i", results[i]);
        PyList_SetItem(pyList, i, pyInt);
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
        return nullptr;
    }
    return PyModule_Create(&nativeModule);
}

// the test function
int main() {
    // real data from Python
    vector<vector<int>> parents      = { {  }, { 0 }, { 0 }, { 1 }, { 2 }, { 1, 10 },
                                         { 4, 5 }, { 3, 5 }, { 5 }, { 7, 8, 13 },
                                         { 2 }, { 4 }, { 11, 10 }, { 6, 12 }};
    vector<vector<int>> inseparables = { { 0 }, { 1 }, { 2, 10 }, { 3 }, { 4, 11, 12 }, { 5 },
                                         { 6, 13 }, { 7 }, { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 },};
    vector<vector<int>> workers      = { { 50, 50, 50, 50, 50, 50 } };               // one contractor with 6 types of workers

    vector<int> chromosomeOrder = { 0, 1, 2, 3, 5, 7, 4, 8, 6, 9 };
    vector<vector<int>> chromosomeResources = {
            { 0,  0,  0,  0,  0,  0, 0},
            {49, 49,  0,  0,  0,  0, 0},
            {49, 49, 49, 49, 49, 49, 0},
            { 0, 49, 49,  0,  0, 49, 0},
            { 0, 49, 49,  0,  0, 49, 0},
            {49, 49, 49,  0, 49, 49, 0},
            {49, 49, 49, 49,  0,  0, 0},
            {49,  0, 49, 49, 49, 49, 0},
            {49, 49, 49, 49,  0, 49, 0},
            { 0,  0,  0,  0,  0,  0, 0}
    };

    ChromosomeEvaluator evaluator(parents, inseparables, workers, parents.size(), nullptr);
    int res = evaluator.testEvaluate(chromosomeOrder, chromosomeResources);

    cout << "Result: " << res << endl;

    return 0;
}
