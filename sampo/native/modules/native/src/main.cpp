#include <iostream>

#include "Python.h"
//#include <Python.h>

using namespace std;

int main() {
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue, *sys, *path;

    Py_Initialize();

//    sys  = PyImport_ImportModule("sys");
//    path = PyObject_GetAttrString(sys, "path");
//    PyList_Append(path, PyUnicode_FromString("."));

    pModule = PyImport_ImportModule("extra.test");

    if (!pModule) {
        PyErr_Print();
        cout << "ERROR in pModule" << endl;
        exit(1);
    }

    pFunc = PyObject_GetAttrString(pModule, "get_args");

    if (!pFunc) {
        PyErr_Print();
        cout << "ERROR in pFunc" << endl;
        exit(1);
    }

    pArgs = PyTuple_New(0);

    PyObject* wg = PyObject_CallObject(pFunc, pArgs);

    if (!wg) {
        PyErr_Print();
        cout << "ERROR in WorkGraph#loadf" << endl;
        exit(1);
    }

    Py_Finalize();
    return 0;
}
