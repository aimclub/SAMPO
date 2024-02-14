#ifndef PYTHON_DESERIALIZER_H
#define PYTHON_DESERIALIZER_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/contractor.h"

namespace PythonDeserializer {
WorkGraph *workGraph(PyObject *pyWorkGraph);

vector<Contractor *> contractors(PyObject *pyContractors);

vector<Chromosome *> decodeChromosomes(PyObject *incoming);

PyObject *encodeChromosome(Chromosome *incoming);

PyObject *encodeChromosomes(vector<Chromosome *> &incoming);
}    // namespace PythonDeserializer

#endif    // PYTHON_DESERIALIZER_H
