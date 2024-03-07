#include <iostream>
#include <string>

#include <pybind11/embed.h>

#include "native/python_deserializer.h"

namespace py = pybind11;
using namespace std;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    try {
        py::module test_module = py::module_::import("extra.test");

//        auto result = test_module.attr("get_args")().cast<py::tuple>();
//
//        cout << result[0].attr("vertex_count").cast<int>() << endl;
        // PythonDeserializer::workGraph(result[0].ptr());

    } catch (exception &e) {
        cout << e.what();
    }
    return 0;
}
