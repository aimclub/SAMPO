#include "pycodec.h"

// There are coders and decoders for Python built-in data structures
namespace PyCodec {

    // ===========================
    // ====== Coder section ======
    // ===========================

    PyObject *toPrimitive(int value) {
        return PyLong_FromLong(value);
    }

    PyObject *toPrimitive(long value) {
        return PyLong_FromLong(value);
    }

    PyObject *toPrimitive(float value) {
        return PyFloat_FromDouble(value);
    }

    PyObject *toPrimitive(double value) {
        return PyFloat_FromDouble(value);
    }

    PyObject *toPrimitive(const string &value) {
        return PyUnicode_FromString(value.c_str());
    }

    template<typename T>
    PyObject *toPyList(const vector<T> &data) {
        PyObject *listObj = PyList_New(data.size());
        if (!listObj) throw logic_error("Unable to allocate memory for Python list");
        for (size_t i = 0; i < data.size(); i++) {
            PyObject *num = toPrimitive(data[i]);
            if (!num) {
                Py_DECREF(listObj);
                throw logic_error("Unable to allocate memory for Python list");
            }
            PyList_SET_ITEM(listObj, i, num);
        }
        return listObj;
    }

    // =============================
    // ====== Decoder section ======
    // =============================

    int fromPrimitive(PyObject* incoming, int typeref) {
        return (int) PyLong_AsLong(incoming);
    }

    long fromPrimitive(PyObject* incoming, long typeref) {
        return PyLong_AsLong(incoming);
    }

    float fromPrimitive(PyObject* incoming, float typeref) {
        return (float) PyFloat_AsDouble(incoming);
    }

    double fromPrimitive(PyObject* incoming, double typeref) {
        return PyFloat_AsDouble(incoming);
    }

    string fromPrimitive(PyObject* incoming, string typeref) {
        return { PyUnicode_AsUTF8(incoming) };
    }

    template<typename T>
    vector<T> fromList(PyObject *incoming, T typeref) {  // typeref used for T recognition
        vector<float> data;
        if (PyTuple_Check(incoming)) {
            for (Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
                PyObject *value = PyTuple_GetItem(incoming, i);
                data.push_back(fromPrimitive(value, typeref));
            }
        } else {
            if (PyList_Check(incoming)) {
                for (Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
                    PyObject *value = PyList_GetItem(incoming, i);
                    data.push_back(fromPrimitive(value, typeref));
                }
            } else {
                throw logic_error("Passed PyObject pointer was not a list or tuple!");
            }
        }
        return data;
    }
}
