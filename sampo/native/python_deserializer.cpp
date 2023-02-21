#include "pycodec.h"

#include "python_deserializer.h"

namespace PythonDeserializer {

    WorkGraph* workGraph(PyObject* pyWorkGraph) {

    }

    IntervalGaussian decodeIntervalGaussian(PyObject* pyIntervalGaussian) {
        float mean = PyCodec::getAttrFloat(pyIntervalGaussian, "mean");
        float sigma = PyCodec::getAttrFloat(pyIntervalGaussian, "sigma");
        float min_val = PyCodec::getAttrFloat(pyIntervalGaussian, "min_val");
        float max_val = PyCodec::getAttrFloat(pyIntervalGaussian, "max_val");
        return IntervalGaussian(mean, sigma, min_val, max_val);
    }

    Worker* decodeWorker(PyObject* pyWorker) {
        string id = PyCodec::getAttrString(pyWorker, "id");
        string name = PyCodec::getAttrString(pyWorker, "name");
        int count = PyCodec::getAttrInt(pyWorker, "count");
        string contractor_id = PyCodec::getAttrString(pyWorker, "contractor_id");
        IntervalGaussian productivity = decodeIntervalGaussian(PyCodec::getAttr(pyWorker, "productivity"));
        return new Worker(id, name, count, contractor_id, productivity);
    }

    Contractor* decodeContractor(PyObject* pyContractor) {
        PyObject* pyWorkers = PyCodec::getAttr(pyContractor, "workers");
        auto workers = PyCodec::fromList(pyWorkers, decodeWorker);
        return new Contractor(workers);
    }

    vector<Contractor*> contractors(PyObject* pyContractors) {
        return PyCodec::fromList(pyContractors, decodeContractor);
    }
}
