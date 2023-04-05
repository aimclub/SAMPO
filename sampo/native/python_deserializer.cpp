#include <unordered_map>
#include <vector>
#include "pycodec.h"

#include "python_deserializer.h"

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
        } else if (value == "SF") {
            return EdgeType::StartFinish;
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
}
