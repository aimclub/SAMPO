#ifndef PYCODEC_H
#define PYCODEC_H

#include <iostream>
#include <stdexcept>
#include <vector>

#include "native/schemas/evaluator_types.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;

// There are coders and decoders for Python built-in data structures
namespace PyCodec {

// ===========================
// ====== Coder section ======
// ===========================

template <typename T>
inline py::list toList(const vector<T> &data, py::object (*encodeValue)(T)) {
    py::list list_obj(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        list_obj[i] = encodeValue(data[i]);
    }
    return list_obj;
}

template <typename T>
inline py::list toList(const vector<T> &data) {
    py::list list_obj(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        list_obj[i] = data[i];
    }
    return list_obj;
}

// =============================
// ====== Decoder section ======
// =============================

template <typename T>
inline vector<T> fromList(const py::handle &incoming, T (*decodeValue)(const py::handle &)) {
    vector<T> data;

    auto list = incoming.cast<py::list>();
    for (const auto& item : list) {
        data.push_back(decodeValue(item));
    }

    return data;
}

//template <typename T>
//inline vector<T> fromList(const vector<py::handle> &incoming, T (*decodeValue)(const py::handle &)) {
//    vector<T> data;
//
//    for (const auto& item : incoming) {
//        data.push_back(decodeValue(item));
//    }
//
//    return data;
//}

inline vector<py::object> fromList(const py::object &incoming) {
    vector<py::object> data;
    auto list = incoming.cast<py::list>();

    for (const auto& item : list) {
        data.push_back(item.cast<py::object>());
    }

    return data;
}

}    // namespace PyCodec

#endif    // PYCODEC_H
