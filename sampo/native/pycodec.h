#include "Python.h"
#include <vector>
#include <stdexcept>

using namespace std;

// There are coders and decoders for Python built-in data structures
namespace PyCodec {

    // ===========================
    // ====== Coder section ======
    // ===========================

    PyObject *toPrimitive(int value);

    PyObject *toPrimitive(long value);

    PyObject *toPrimitive(float value);

    PyObject *toPrimitive(double value);

    PyObject *toPrimitive(const string &value);

    template<typename T>
    PyObject *toPyList(const vector<T> &data);

    // =============================
    // ====== Decoder section ======
    // =============================

    int fromPrimitive(PyObject* incoming, int typeref);

    long fromPrimitive(PyObject* incoming, long typeref);

    float fromPrimitive(PyObject* incoming, float typeref);

    double fromPrimitive(PyObject* incoming, double typeref);

    string fromPrimitive(PyObject* incoming, string typeref);

    template<typename T>
    vector<T> fromList(PyObject *incoming, T typeref);  // typeref used for T recognition
}
