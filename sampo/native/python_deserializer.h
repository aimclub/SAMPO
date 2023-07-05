#ifndef PYTHON_DESERIALIZER_H
#define PYTHON_DESERIALIZER_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "workgraph.h"
#include "contractor.h"

namespace PythonDeserializer {
    WorkGraph* workGraph(PyObject* pyWorkGraph);

    vector<Contractor*> contractors(PyObject* pyContractors);

    vector<Chromosome*> decodeChromosomes(PyObject* incoming);

    PyObject* encodeChromosome(Chromosome* incoming);

    PyObject* encodeChromosomes(vector<Chromosome*>& incoming);
}

#endif //PYTHON_DESERIALIZER_H
