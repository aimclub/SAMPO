#ifndef PYTHON_DESERIALIZER_H
#define PYTHON_DESERIALIZER_H

#include "native/schemas/evaluator_types.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/chromosome.h"
#include "native/schemas/contractor.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace PythonDeserializer {
    WorkGraph *workGraph(const py::handle &pyWorkGraph);

    vector<Contractor *> contractors(const py::handle &pyContractors);

    vector<Chromosome *> decodeChromosomes(const py::handle &incoming);

    py::object encodeChromosome(Chromosome *incoming);

    py::list encodeChromosomes(vector<Chromosome *> &incoming);

    py::object encodeSchedule(const swork_dict_t &schedule);
}    // namespace PythonDeserializer

#endif    // PYTHON_DESERIALIZER_H
