#include <chrono>
#include <iostream>

#include "native/scheduler/chromosome_evaluator.h"
#include "native/scheduler/genetic.h"
#include "native/pycodec.h"
#include "native/python_deserializer.h"

// GLOBAL TODOS
// TODO Make all classes with encapsulation - remove public fields
// TODO Make parallel runtime
// TODO Performance measurements
// TODO Cache data in C++ memory - parse Python WG and Contractors once
// per-scheduling
// TODO Split data types' definition and implementation

inline int PyLong_AsInt(const py::handle &object) {
    return object.cast<int>();
}

vector<int> decodeIntList(const py::handle &object) {
    return PyCodec::fromList(object, PyLong_AsInt);
}

vector<vector<float>> evaluate(size_t info_ptr_orig, const py::object &py_chromosomes) {
    EvaluateInfo *info_ptr = (EvaluateInfo *) info_ptr_orig;

    auto chromosomes = PythonDeserializer::decodeChromosomes(py_chromosomes);

    ChromosomeEvaluator evaluator(info_ptr);

    evaluator.evaluate(chromosomes);

    vector<vector<float>> fitness;
    fitness.resize(chromosomes.size());
    for (int i = 0; i < chromosomes.size(); i++) {
        fitness[i] = { chromosomes[i]->fitness };
        delete chromosomes[i];
    }
    return fitness;
}

//static PyObject *runGenetic(PyObject *self, PyObject *args) {
//    EvaluateInfo *infoPtr;
//    PyObject *pyChromosomes;
//    float mutateOrderProb, mutateResourcesProb, mutateContractorsProb;
//    float crossOrderProb, crossResourcesProb, crossContractorsProb;
//    int sizeSelection;
//
//    if (!PyArg_ParseTuple(
//            args,
//            "LOOOffffffi",
//            &infoPtr,
//            &pyChromosomes,
//            &mutateOrderProb,
//            &mutateResourcesProb,
//            &mutateContractorsProb,
//            &crossOrderProb,
//            &crossResourcesProb,
//            &crossContractorsProb,
//            &sizeSelection
//        )) {
//        cout << "Can't parse arguments" << endl;
//    }
//    auto start       = chrono::high_resolution_clock::now();
//    auto chromosomes = PythonDeserializer::decodeChromosomes(pyChromosomes);
//    auto stop        = chrono::high_resolution_clock::now();
//    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
//    cout << "Chromosomes decoded in " << duration.count() << " ms" << endl;
//
//    ChromosomeEvaluator evaluator(infoPtr);
//    Genetic g(
//        infoPtr->minReq,
//        mutateOrderProb,
//        mutateResourcesProb,
//        mutateContractorsProb,
//        crossOrderProb,
//        crossResourcesProb,
//        crossContractorsProb,
//        sizeSelection,
//        evaluator
//    );
//    Chromosome *result;
//    //    Py_BEGIN_ALLOW_THREADS;
////    result = g.run(chromosomes);
//    //    Py_END_ALLOW_THREADS;
//    auto pyResult = PythonDeserializer::encodeChromosome(result);
//    delete result;
//    return pyResult;
//}

//static PyObject *ddd(PyObject *self, PyObject *args) {
//    PyObject *pyWorkGraph;
//
//    if (!PyArg_ParseTuple(
//            args,
//            "O",
//            &pyWorkGraph
//    )) {
//        cout << "Can't parse arguments" << endl;
//        Py_RETURN_NONE;
//    }
//
//    auto* wg = PythonDeserializer::workGraph(pyWorkGraph);
//    cout << "Decoded wg size: " << wg->nodes.size() << endl;
//    delete wg;
//    Py_RETURN_NONE;
//}

size_t decodeEvaluationInfo(const py::object &pythonWrapper,
                            const py::object &pyWorkGraph,
                            const py::object &pyContractors,
                            const string &pyWorkEstimatorPath,
                            const py::object &pyParents,
                            const py::object &pyHeadParents,
                            const py::object &pyInseparables,
                            const py::object &pyWorkers,
                            int totalWorksCount,
                            bool usePythonWorkEstimator,
                            bool useExternalWorkEstimator,
                            const vector<float> &volume,
                            const py::object &minReq,
                            const py::object &maxReq,
                            const vector<string> &id2work,
                            const vector<string> &id2res) {

//    cout << pyWorkGraph->ob_refcnt << endl;
//    cout << pyContractors->ob_refcnt << endl;

//    auto* wg = PythonDeserializer::workGraph(pyWorkGraph);
//    cout << "Decoded wg size: " << wg->nodes.size() << endl;
//    delete wg;
//    PythonDeserializer::contractors(pyContractors);

    return (size_t) new EvaluateInfo {
            pythonWrapper,
            PythonDeserializer::workGraph(pyWorkGraph),
            PythonDeserializer::contractors(pyContractors),
            PyCodec::fromList(pyParents, decodeIntList),
            PyCodec::fromList(pyHeadParents, decodeIntList),
            PyCodec::fromList(pyInseparables, decodeIntList),
            PyCodec::fromList(pyWorkers, decodeIntList),
            volume,
            PyCodec::fromList(minReq, decodeIntList),
            PyCodec::fromList(maxReq, decodeIntList),
            id2work,
            id2res,
            pyWorkEstimatorPath,
            LandscapeConfiguration(),
            new DefaultWorkTimeEstimator(),
            totalWorksCount,
            usePythonWorkEstimator,
            useExternalWorkEstimator
    };
}

void freeEvaluationInfo(size_t info_ptr_orig) {
    EvaluateInfo *info_ptr = (EvaluateInfo *) info_ptr_orig;
    delete info_ptr->work_estimator;
    delete info_ptr;
}

//static PyMethodDef nativeMethods[] = {
//    {            "evaluate",
//     evaluate, METH_VARARGS,
//     "Evaluates the chromosome using Just-In-Time-Timeline"                             },
//    {          "runGenetic", runGenetic, METH_VARARGS,    "Runs the whole genetic cycle"},
//    {"decodeEvaluationInfo",
//     decodeEvaluationInfo, METH_VARARGS,
//     "Uploads the scheduling info to C++ memory and caches it"                          },
//    {"ddd",
//            ddd, METH_VARARGS,
//            "Uploads the scheduling info to C++ memory and caches it"                          },
//    {  "freeEvaluationInfo",
//     freeEvaluationInfo, METH_VARARGS,
//     "Frees C++ scheduling cache. Must be called in the end of scheduling to "
//     "avoid memory leaks."                                                              },
//    {               nullptr,    nullptr,            0,                           nullptr}
//};
//
//static PyModuleDef nativeModule = { PyModuleDef_HEAD_INIT,
//                                    "native",
//                                    "The high-efficient native implementation of SAMPO modules",
//                                    -1,
//                                    nativeMethods };

PYBIND11_MODULE(native, m) {
    m.def(
          "decodeEvaluationInfo",
          decodeEvaluationInfo,
          "Uploads the scheduling info to C++ memory and caches it"
    );
    m.def(
            "freeEvaluationInfo",
            freeEvaluationInfo,
            "Frees C++ scheduling cache. Must be called in the end of scheduling to avoid memory leaks"
    );
    m.def(
            "evaluate",
            evaluate,
            "Evaluates the chromosome using Just-In-Time-Timeline"
    );
}
