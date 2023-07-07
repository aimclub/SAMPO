#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <iostream>

#include "chromosome_evaluator.h"
#include "pycodec.h"
#include "genetic.h"
#include "python_deserializer.h"


#include <chrono>

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

static string decodeString(PyObject* object) {
    return string {PyUnicode_AsUTF8(object) };
}

static float decodeFloat(PyObject* object) {
    return (float) PyFloat_AsDouble(object);
}

static PyObject* evaluate(PyObject *self, PyObject *args) {
    EvaluateInfo* infoPtr;
    PyObject* pyChromosomes;
    if (!PyArg_ParseTuple(args, "LO", &infoPtr, &pyChromosomes)) {
        cout << "Can't parse arguments" << endl;
    }
    auto chromosomes = PythonDeserializer::decodeChromosomes(pyChromosomes);

    ChromosomeEvaluator evaluator(infoPtr);

    evaluator.evaluate(chromosomes);

    PyObject* pyList = PyList_New(chromosomes.size());
    Py_INCREF(pyList);
    for (int i = 0; i < chromosomes.size(); i++) {
        PyObject* pyInt = Py_BuildValue("i", chromosomes[i]->fitness);
        PyList_SetItem(pyList, i, pyInt);
    }
    return pyList;
}

static PyObject* runGenetic(PyObject* self, PyObject* args) {
    EvaluateInfo* infoPtr;
    PyObject* pyChromosomes;
    float mutateOrderProb, mutateResourcesProb, mutateContractorsProb;
    float crossOrderProb, crossResourcesProb, crossContractorsProb;
    int sizeSelection;

    if (!PyArg_ParseTuple(args, "LOffffffi", &infoPtr, &pyChromosomes,
                          &mutateOrderProb, &mutateResourcesProb, &mutateContractorsProb,
                          &crossOrderProb, &crossResourcesProb, &crossContractorsProb, &sizeSelection)) {
        cout << "Can't parse arguments" << endl;
    }
    auto start = chrono::high_resolution_clock::now();
    auto chromosomes = PythonDeserializer::decodeChromosomes(pyChromosomes);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Chromosomes decoded in " << duration.count() << " ms" << endl;

    ChromosomeEvaluator evaluator(infoPtr);
    Genetic g(infoPtr->minReq,
              mutateOrderProb, mutateResourcesProb, mutateContractorsProb,
              crossOrderProb, crossResourcesProb, crossContractorsProb,
              sizeSelection, evaluator);
    Chromosome* result;
//    Py_BEGIN_ALLOW_THREADS;
    result = g.run(chromosomes);
//    Py_END_ALLOW_THREADS;
    auto pyResult = PythonDeserializer::encodeChromosome(result);
    delete result;
    return pyResult;
}

static PyObject* decodeEvaluationInfo(PyObject *self, PyObject *args) {
    PyObject* pythonWrapper;
    PyObject* pyParents;
    PyObject* pyHeadParents;
    PyObject* pyInseparables;
    PyObject* pyWorkers;
    int totalWorksCount;
    bool usePythonWorkEstimator;
    bool useExternalWorkEstimator;
    PyObject* volume;
    PyObject* minReq;
    PyObject* maxReq;
    PyObject* id2work;
    PyObject* id2res;

    if (!PyArg_ParseTuple(args, "OOOOOippOOOOO",
                          &pythonWrapper, &pyParents, &pyHeadParents, &pyInseparables,
                          &pyWorkers, &totalWorksCount, &usePythonWorkEstimator, &useExternalWorkEstimator,
                          &volume, &minReq, &maxReq, &id2work, &id2res)) {
        cout << "Can't parse arguments" << endl;
    }

    auto* info = new EvaluateInfo {
        pythonWrapper,
        PyCodec::fromList(pyParents, decodeIntList),
        PyCodec::fromList(pyHeadParents, decodeIntList),
        PyCodec::fromList(pyInseparables, decodeIntList),
        PyCodec::fromList(pyWorkers, decodeIntList),
        PyCodec::fromList(volume, decodeFloat),
        PyCodec::fromList(minReq, decodeIntList),
        PyCodec::fromList(maxReq, decodeIntList),
        PyCodec::fromList(id2work, decodeString),
        PyCodec::fromList(id2res, decodeString),
        "aaaa", // TODO Propagate workEstimatorPath from Python
        totalWorksCount,
        usePythonWorkEstimator,
        useExternalWorkEstimator
    };

    auto* res = PyLong_FromVoidPtr(info);
    Py_INCREF(res);
    return res;
}

static PyObject* freeEvaluationInfo(PyObject *self, PyObject *args) {
    EvaluateInfo* infoPtr;
    if (!PyArg_ParseTuple(args, "L", &infoPtr)) {
        cout << "Can't parse arguments" << endl;
    }
    delete infoPtr;
    Py_RETURN_NONE;
}

static PyMethodDef nativeMethods[] = {
        {"evaluate", evaluate, METH_VARARGS,
                "Evaluates the chromosome using Just-In-Time-Timeline" },
        {"runGenetic", runGenetic, METH_VARARGS,
                "Runs the whole genetic cycle" },
        {"decodeEvaluationInfo", decodeEvaluationInfo, METH_VARARGS,
                "Uploads the scheduling info to C++ memory and caches it" },
        {"freeEvaluationInfo", freeEvaluationInfo, METH_VARARGS,
                "Frees C++ scheduling cache. Must be called in the end of scheduling to avoid memory leaks." },
        {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef nativeModule = {
        PyModuleDef_HEAD_INIT,
        "native",
        "The high-efficient native implementation of SAMPO modules",
        -1,
        nativeMethods
};

PyMODINIT_FUNC
PyInit_native(void) {
    assert(! PyErr_Occurred());
    // Initialise Numpy
    import_array();
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
    vector<vector<int>> workers      = { { 50, 50, 50, 50, 50, 50 } };  // one contractor with 6 types of workers

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

    string v = "aaaa";

    auto* info = new EvaluateInfo {
            nullptr,
            parents,
            vector<vector<int>>(),
            inseparables,
            workers,
            vector<float>(),
            vector<vector<int>>(),
            vector<vector<int>>(),
            vector<string>(),
            vector<string>(),
            v, // TODO Propagate workEstimatorPath from Python
            0,
            false,
            true
    };
//
    ChromosomeEvaluator evaluator(info);
//    int res = evaluator.testEvaluate(chromosomeOrder, chromosomeResources);
//
//    cout << "Result: " << res << endl;

    return 0;
}
