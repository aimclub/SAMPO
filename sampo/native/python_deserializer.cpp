#include <unordered_map>
#include <vector>

#include "pycodec.h"
#include "python_deserializer.h"
#include "utils/use_numpy.h"

namespace PythonDeserializer {

    WorkerReq decodeWorkerReq(PyObject* pyWorkerReq) {
        string kind = PyCodec::getAttrString(pyWorkerReq, "kind");
        Time volume = Time(PyCodec::getAttrInt(PyCodec::getAttr(pyWorkerReq, "volume"), "value"));
        int min_count = PyCodec::getAttrInt(pyWorkerReq, "min_count");
        int max_count = PyCodec::getAttrInt(pyWorkerReq, "max_count");
        return WorkerReq(kind, volume, min_count, max_count);
    }

    WorkUnit* decodeWorkUnit(PyObject* pyWorkUnit) {
        auto worker_reqs = PyCodec::fromList(
                PyCodec::getAttr(pyWorkUnit, "worker_reqs"),
                decodeWorkerReq);
        float volume = PyCodec::getAttrFloat(pyWorkUnit, "volume");
        bool isServiceUnit = PyCodec::getAttrBool(pyWorkUnit, "is_service_unit");
        return new WorkUnit(worker_reqs, volume, isServiceUnit);
    }

    typedef struct {
        WorkUnit* workUnit;
        vector<tuple<string, float, EdgeType>> parents;
    } UnlinkedGraphNode;

    EdgeType decodeEdgeType(PyObject* pyEdgeType) {
        string value = PyCodec::getAttrString(pyEdgeType, "_value_");
        if (value == "IFS") {
            return EdgeType::InseparableFinishStart;
        } else if (value == "FFS") {
            return EdgeType::LagFinishStart;
        } else if (value == "SS") {
            return EdgeType::StartStart;
        } else if (value == "FF") {
            return EdgeType::FinishFinish;
        } else if (value == "FS") {
            return EdgeType::FinishStart;
        } else {
            throw logic_error("Illegal EdgeType: " + value);
        }
    }

    // helper function for next method
    template<typename T>
    T identity(T v) {
        return v;
    }

    UnlinkedGraphNode decodeNodeWorkUnit(PyObject* pyGraphNode) {
        auto workUnit = PyCodec::getAttr(pyGraphNode, "_work_unit", decodeWorkUnit);
        auto pyParents = PyCodec::fromList(PyCodec::getAttr(pyGraphNode, "_parent_edges"), identity<PyObject*>);
        auto parents = vector<tuple<string, float, EdgeType>>();
        // decode with first element replaced GraphNode -> GraphNode#WorkUnit#id
        for (PyObject* pyParent : pyParents) {
            // go deep and get work_unit's id
            string id = PyCodec::getAttrString(
                    PyCodec::getAttr(
                        PyCodec::getAttr(pyParent, "start"),
                        "_work_unit"),
                    "id");
            float lag = PyCodec::getAttrFloat(pyGraphNode, "lag");
            EdgeType type = PyCodec::getAttr(pyGraphNode, "type", decodeEdgeType);
            parents.emplace_back(id, lag, type);
        }
        return { workUnit, parents };
    }

    WorkGraph* workGraph(PyObject* pyWorkGraph) {
        auto unlinked_ordered_nodes = PyCodec::fromList(PyCodec::getAttr(pyWorkGraph, "nodes"), decodeNodeWorkUnit);
        auto nodes = unordered_map<string, GraphNode*>();
        auto ordered_nodes = vector<GraphNode*>();

        // linking
        for (const auto& s_node : unlinked_ordered_nodes) {
            auto linked_parents = vector<tuple<GraphNode*, float, EdgeType>>();
            for (const auto& unlinked_parent : s_node.parents) {
                string id = get<0>(unlinked_parent);
                float lag = get<1>(unlinked_parent);
                EdgeType type = get<2>(unlinked_parent);
                linked_parents.emplace_back(nodes[id], lag, type);
            }

            auto node = new GraphNode(s_node.workUnit, linked_parents);
            ordered_nodes.push_back(node);
            nodes[s_node.workUnit->id] = node;
        }

        return new WorkGraph(ordered_nodes);
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
        IntervalGaussian productivity = PyCodec::getAttr(pyWorker, "productivity", decodeIntervalGaussian);
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

    inline Chromosome* decodeChromosome(PyObject* incoming) {
        PyObject* pyOrder; //= PyList_GetItem(chromosome, 0);
        PyObject* pyResources; //= PyList_GetItem(chromosome, 1);
        PyObject* pyContractors; //= PyList_GetItem(chromosome, 2);

        if (!PyArg_ParseTuple(incoming, "OOO", &pyOrder, &pyResources, &pyContractors)) {
            cerr << "Can't parse chromosome!!!!" << endl;
            return nullptr;
        }

        int worksCount = PyArray_DIM(pyOrder, 0);  // without inseparables
        int resourcesCount = PyArray_DIM(pyResources, 1) - 1;
        int contractorsCount = PyArray_DIM(pyContractors, 0);

        auto* chromosome = new Chromosome(worksCount, resourcesCount, contractorsCount);

//        cout << "------------------" << endl;

        // TODO Find the way to faster copy from NumPy ND array.
        //  !!!Attention!!! You can't just memcpy()! Plain NumPy array is not C-aligned!
        auto& order = chromosome->getOrder();
        for (int i = 0; i < worksCount; i++) {
            int v = *(int *) PyArray_GETPTR1(pyOrder, i);
            *order[i] = v;
//            cout << v << " " << *order[i] << *chromosome->getOrder()[i] << endl;
        }
//        cout << "------------------" << endl;

        auto& resources = chromosome->getResources();
        for (int work = 0; work < worksCount; work++) {
            for (int worker = 0; worker < resourcesCount + 1; worker++) {
                resources[work][worker] = PyCodec::Py_GET2D(pyResources, work, worker);
            }
        }
        auto& contractors = chromosome->getContractors();
        for (int contractor = 0; contractor < contractorsCount; contractor++) {
            for (int worker = 0; worker < resourcesCount; worker++) {
                contractors[contractor][worker] = PyCodec::Py_GET2D(pyContractors, contractor, worker);
            }
        }

        return chromosome;
    }

    vector<Chromosome*> decodeChromosomes(PyObject* incoming) {
        return PyCodec::fromList(incoming, decodeChromosome);
    }

    PyObject* encodeChromosome(Chromosome* incoming) {
        npy_intp dimsOrder[] { incoming->getOrder().size() };
        npy_intp dimsResources[] { incoming->getResources().height(), incoming->getResources().width() };
        npy_intp dimsContractors[] { incoming->getContractors().height(), incoming->getContractors().width() };

        cout << dimsOrder[0] << endl;
        cout << dimsResources[0] << " " << dimsResources[1] << endl;
        cout << dimsContractors[0] << " " << dimsContractors[1] << endl;

        import_array();

        PyObject* pyOrder = PyArray_EMPTY(1, dimsOrder, NPY_INT, 0);
        Py_XINCREF(pyOrder);
        PyObject* pyResources = PyArray_EMPTY(2, dimsResources, NPY_INT, 0);
        Py_XINCREF(pyResources);
        PyObject* pyContractors = PyArray_EMPTY(2, dimsContractors, NPY_INT, 0);
        Py_XINCREF(pyContractors);

//        Py_RETURN_NONE;

        if (pyOrder == nullptr || pyResources == nullptr || pyContractors == nullptr) {
            cout << "Can't allocate chromosome" << endl;
            // we can run out-of-memory, but still just return to Python
            Py_RETURN_NONE;
        }

        for (int i = 0; i < dimsOrder[0]; i++) {
            PyCodec::Py_GET1D(pyOrder, i) = *incoming->getOrder()[i];
        }
        for (int work = 0; work < dimsResources[0]; work++) {
            for (int resource = 0; resource < dimsResources[1]; resource++) {
                PyCodec::Py_GET2D(pyResources, work, resource)
                    = incoming->getResources()[work][resource];
            }
        }
        for (int contractor = 0; contractor < dimsContractors[0]; contractor++) {
            for (int resource = 0; resource < dimsContractors[1]; resource++) {
                PyCodec::Py_GET2D(pyContractors, contractor, resource)
                    = incoming->getContractors()[contractor][resource];
            }
        }

//        Py_RETURN_NONE;

        PyObject* pyChromosome = Py_BuildValue("(OOO)", pyOrder, pyResources, pyContractors);
        Py_XINCREF(pyChromosome);  // we can run out-of-memory, but still just return to Python
        return pyChromosome;
    }

    PyObject* encodeChromosomes(const vector<Chromosome*>& incoming) {
        return PyCodec::toList(incoming, encodeChromosome);
    }
}
