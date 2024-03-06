//#include <iostream>
//
//#include "Python.h"
////#include <Python.h>
//
//using namespace std;
//
//int main() {
//    PyObject *pName, *pModule, *pFunc;
//    PyObject *pArgs, *pValue, *sys, *path;
//
//    Py_Initialize();
//
////    sys  = PyImport_ImportModule("sampo.schemas");
////    path = PyObject_GetAttrString(sys, "WorkGraph");
////    PyList_Append(path, PyUnicode_FromString("."));
//
//    pModule = PyImport_ImportModule("sampo.schemas.WorkGraph");
//
//    if (!pModule) {
//        PyErr_Print();
//        cout << "ERROR in pModule" << endl;
//        exit(1);
//    }
//
//    pFunc = PyObject_GetAttrString(pModule, "loadf");
//    pArgs = PyTuple_New(1);
//    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString("wg.json"));
//
//    PyObject* wg = PyObject_CallObject(pFunc, pArgs);
//
//    if (!wg) {
//        PyErr_Print();
//        cout << "ERROR in WorkGraph#loadf" << endl;
//        exit(1);
//    }
//
//    Py_Finalize();
//    return 0;
//}
