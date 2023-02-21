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

    string fromPrimitive(PyObject* incoming, const string& typeref);

    template<typename T>
    vector<T> fromList(PyObject *incoming, T (*decodeValue)(PyObject*));

    template<typename T>
    vector<T> fromList(PyObject *incoming, T typeref);  // typeref used for T recognition

    // ============================
    // ====== Helper section ======
    // ============================

    PyObject* getAttr(PyObject* incoming, const char *name;

    int getAttrInt(PyObject* incoming, const char *name);

    long getAttrLong(PyObject* incoming, const char *name);

    float getAttrFloat(PyObject* incoming, const char *name);

    double getAttrDouble(PyObject* incoming, const char *name);

    string getAttrString(PyObject* incoming, const char *name);
}
