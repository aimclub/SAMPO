#include <iostream>
#include <string>

#include <pybind11/embed.h>

#include "native/scheduler/chromosome_evaluator.h"
#include "native/python_deserializer.h"

namespace py = pybind11;
//using namespace std;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    try {
        py::module test_module = py::module_::import("extra.test");

        auto result = test_module.attr("get_cached_args")().cast<py::tuple>();

        cout << "Python size: " << result[0].attr("vertex_count").cast<int>() << endl;
        WorkGraph* wg = PythonDeserializer::workGraph(result[0]);
        cout << "Decoded size: " << wg->nodes.size() << endl;

        auto contractors = PythonDeserializer::contractors(result[1]);
        cout << "Contractors number: " << contractors.size() << endl;

        auto chromosomes = PythonDeserializer::decodeChromosomes(test_module.attr("get_sample_chromosomes")());

//        cout << "-----------" << endl;
//        for (auto* node : wg->nodes) {
//            for (const auto *inseparable_node: node->getInseparableChainWithSelf()) {
//                cout << inseparable_node->getWorkUnit()->id << " ";
//            }
//            cout << endl;
//        }
//        cout << "-----------" << endl;

        ChromosomeEvaluator evaluator(wg, contractors, ScheduleSpec(), new DefaultWorkTimeEstimator());

//        cout << "-----------" << endl;
//        for (auto* node : wg->nodes) {
//            for (const auto *inseparable_node: node->getInseparableChainWithSelf()) {
//                cout << inseparable_node->getWorkUnit()->name << " ";
//            }
//            cout << endl;
//        }
//        cout << "-----------" << endl;
        evaluator.evaluate(chromosomes);
        cout << chromosomes[0]->fitness << endl;

//        delete wg;
//        for (auto* contractor : contractors) {
//            delete contractor;
//        }
//        for (auto* chromosome : chromosomes) {
//            delete chromosome;
//        }

    } catch (exception &e) {
        cout << e.what();
    }
    return 0;
}
