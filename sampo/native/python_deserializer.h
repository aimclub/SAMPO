#include "Python.h"
#include "workgraph.h"
#include "contractor.h"

namespace PythonDeserializer {
    WorkGraph* workGraph(PyObject* pyWorkGraph);

    vector<Contractor*> contractors(PyListObject* pyContractors);
}
