#ifndef PYCODEC_H
#define PYCODEC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "evaluator_types.h"

#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

// There are coders and decoders for Python built-in data structures
namespace PyCodec {

    // ===========================
    // ====== Coder section ======
    // ===========================

    inline PyObject *toPrimitive(int value) {
        return PyLong_FromLong(value);
    }

    inline PyObject *toPrimitive(long value) {
        return PyLong_FromLong(value);
    }

    inline PyObject *toPrimitive(float value) {
        return PyFloat_FromDouble(value);
    }

    inline PyObject *toPrimitive(double value) {
        return PyFloat_FromDouble(value);
    }

    inline PyObject *toPrimitive(const string &value) {
        return PyUnicode_FromString(value.c_str());
    }

    template<typename T>
    PyObject *toList(const vector<T> &data, PyObject* (*encodeValue)(T)) {
        PyObject *listObj = PyList_New(data.size());
        if (!listObj) throw logic_error("Unable to allocate memory for Python list");
        for (size_t i = 0; i < data.size(); i++) {
            PyObject *num = encodeValue(data[i]);
            if (!num) {
                Py_DECREF(listObj);
                throw logic_error("Unable to allocate memory for Python list");
            }
            PyList_SET_ITEM(listObj, i, num);
        }
        return listObj;
    }

    template<typename T>
    PyObject *toPrimitiveList(const vector<T> &data) {
        return toList(data, toPrimitive);
    }

    // =============================
    // ====== Decoder section ======
    // =============================

    inline int fromPrimitive(PyObject* incoming, int typeref) {
        return (int) PyLong_AsLong(incoming);
    }

    inline long fromPrimitive(PyObject* incoming, long typeref) {
        return PyLong_AsLong(incoming);
    }

    inline float fromPrimitive(PyObject* incoming, float typeref) {
        return (float) PyFloat_AsDouble(incoming);
    }

    inline double fromPrimitive(PyObject* incoming, double typeref) {
        return PyFloat_AsDouble(incoming);
    }

    inline string fromPrimitive(PyObject* incoming, const string& typeref) {
        return { PyUnicode_AsUTF8(incoming) };
    }

    template<typename T>
    inline vector<T> fromList(PyObject *incoming, T (*decodeValue)(PyObject*)) {
        vector<T> data;
        if (PyTuple_Check(incoming)) {
            for (Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
                PyObject *value = PyTuple_GetItem(incoming, i);
                data.push_back(decodeValue(value));
            }
        } else {
            if (PyList_Check(incoming)) {
                for (Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
                    PyObject *value = PyList_GetItem(incoming, i);
                    data.push_back(decodeValue(value));
                }
            } else {
                throw logic_error("Passed PyObject pointer was not a list or tuple!");
            }
        }
        return data;
    }

    template<typename T>
    inline vector<T> fromList(PyObject *incoming, T typeref) {  // typeref used for T recognition
        return fromList(incoming, [typeref](PyObject* value) { return fromPrimitive(value, typeref); });
    }

    inline static int& Py_GET1D(const PyObject* list, size_t ind) {
        return * (int*) PyArray_GETPTR1(list, ind);
    }

    inline static int& Py_GET2D(const PyObject* resources, size_t work, size_t worker) {
        return * (int*) PyArray_GETPTR2(resources, work, worker);
    }

    // ============================
    // ====== Helper section ======
    // ============================

    inline PyObject* getAttr(PyObject* incoming, const char *name) {
        return PyObject_GetAttr(incoming, PyUnicode_FromString(name));
    }

    template<typename T>
    inline T getAttr(PyObject* incoming, const char *name, T (*decodeValue)(PyObject*)) {
        return decodeValue(PyObject_GetAttr(incoming, PyUnicode_FromString(name)));
    }

    inline int getAttrInt(PyObject* incoming, const char *name) {
        return fromPrimitive(getAttr(incoming, name), 0);
    }

    inline long getAttrLong(PyObject* incoming, const char *name) {
        return fromPrimitive(getAttr(incoming, name), 0L);
    }

    inline float getAttrFloat(PyObject* incoming, const char *name) {
        return fromPrimitive(getAttr(incoming, name), 0.0f);
    }

    inline double getAttrDouble(PyObject* incoming, const char *name) {
        return fromPrimitive(getAttr(incoming, name), 0.0);
    }

    inline bool getAttrBool(PyObject* incoming, const char *name) {
        return PyObject_IsTrue(getAttr(incoming, name));
    }

    inline string getAttrString(PyObject* incoming, const char *name) {
        return fromPrimitive(getAttr(incoming, name), "");
    }
}

#endif //PYCODEC_H
