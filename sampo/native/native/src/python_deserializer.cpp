#include "native/python_deserializer.h"

#include <unordered_map>
#include <vector>

#include "native/pycodec.h"

namespace PythonDeserializer {

WorkerReq decodeWorkerReq(const py::handle &pyWorkerReq) {
    return {
        pyWorkerReq.attr("kind").cast<string>(),
        Time(pyWorkerReq.attr("volume").attr("value").cast<int>()),
        pyWorkerReq.attr("min_count").cast<int>(),
        pyWorkerReq.attr("max_count").cast<int>()
    };
}

WorkUnit *decodeWorkUnit(const py::handle &pyWorkUnit) {
    py::object wr = pyWorkUnit.attr("worker_reqs");
    return new WorkUnit(
                pyWorkUnit.attr("id").cast<string>(),
                pyWorkUnit.attr("name").cast<string>(),
                PyCodec::fromList(wr, decodeWorkerReq),
                pyWorkUnit.attr("volume").cast<float>(),
                pyWorkUnit.attr("is_service_unit").cast<bool>()
    );
}

struct UnlinkedGraphNode {
    WorkUnit *work_unit;
    vector<tuple<string, float, EdgeType>> parents;
};

EdgeType decodeEdgeType(const py::handle &pyEdgeType) {
    auto value = pyEdgeType.attr("_value_").cast<string>();
    if (value == "IFS") return EdgeType::InseparableFinishStart;
    if (value == "FFS") return EdgeType::LagFinishStart;
    if (value == "SS") return EdgeType::StartStart;
    if (value == "FF") return EdgeType::FinishFinish;
    if (value == "FS") return EdgeType::FinishStart;

    throw logic_error("Illegal EdgeType: " + value);
}

UnlinkedGraphNode decodeNodeWorkUnit(const py::handle &pyGraphNode) {
    auto* work_unit = decodeWorkUnit(pyGraphNode.attr("_work_unit"));
    auto py_parents = pyGraphNode.attr("_parent_edges").cast<py::list>();
    auto parents = vector<tuple<string, float, EdgeType>>();
    // decode with first element replaced GraphNode -> GraphNode#WorkUnit#id
    for (const auto &py_parent : py_parents) {
        // go deep and get work_unit's id
        auto id = py_parent.attr("start").attr("work_unit").attr("id").cast<string>();
        auto lag = py_parent.attr("lag").cast<float>();
        EdgeType type = decodeEdgeType(py_parent.attr("type"));
        parents.emplace_back(id, lag, type);
    }

    return { work_unit, parents };
}

WorkGraph *workGraph(const py::handle &pyWorkGraph) {
    auto unlinked_ordered_nodes = PyCodec::fromList(pyWorkGraph.attr("nodes"), decodeNodeWorkUnit);
    auto nodes         = unordered_map<string, GraphNode *>();
    auto ordered_nodes = vector<GraphNode *>();

    // linking
    for (const auto &s_node : unlinked_ordered_nodes) {
        auto linked_parents = vector<tuple<GraphNode *, float, EdgeType>>();
        auto* node = new GraphNode(s_node.work_unit, linked_parents);
        ordered_nodes.push_back(node);
        nodes[s_node.work_unit->id] = node;
    }

    for (const auto &s_node : unlinked_ordered_nodes) {
        vector<tuple<GraphNode *, float, EdgeType>> linked_parents;
        for (const auto &unlinked_parent : s_node.parents) {
            string id     = get<0>(unlinked_parent);
            float lag     = get<1>(unlinked_parent);
            EdgeType type = get<2>(unlinked_parent);
            linked_parents.emplace_back(nodes[id], lag, type);
        }

        nodes[s_node.work_unit->id]->add_parents(linked_parents);
    }

    return new WorkGraph(ordered_nodes);
}

IntervalGaussian decodeIntervalGaussian(const py::handle &pyIntervalGaussian) {
    return IntervalGaussian(
        pyIntervalGaussian.attr("mean").cast<float>(),
        pyIntervalGaussian.attr("sigma").cast<float>(),
        pyIntervalGaussian.attr("min_val").cast<float>(),
        pyIntervalGaussian.attr("max_val").cast<float>()
    );
}

string get_string_attr(const py::handle &obj, const char *name) {
    return obj.attr(name).cast<std::string>();
}

Worker decodeWorker(const py::handle &pyWorker) {
    auto name = get_string_attr(pyWorker, "name"); // pyWorker.attr("name").cast<std::string>();
    auto id = get_string_attr(pyWorker, "id"); // pyWorker.attr("name").cast<std::string>();
    return Worker(
            id,
            name,
            pyWorker.attr("count").cast<int>(),
            pyWorker.attr("cost_one_unit").cast<int>(),
            pyWorker.attr("contractor_id").cast<string>(),
            decodeIntervalGaussian(pyWorker.attr("productivity"))
    );
}

Contractor *decodeContractor(const py::handle &pyContractor) {
    auto name = pyContractor.attr("name").cast<string>();
    return new Contractor(
            pyContractor.attr("id").cast<string>(),
            name,
            PyCodec::fromList(pyContractor.attr("worker_list"), decodeWorker)
    );
}

vector<Contractor *> contractors(const py::handle &pyContractors) {
    return PyCodec::fromList(pyContractors, decodeContractor);
}

Chromosome *decodeChromosome(const py::handle &py_incoming) {
    auto py_tuple = py_incoming.cast<py::tuple>();
    auto py_order = py_tuple[0].cast<py::array_t<int>>();
    auto py_resources = py_tuple[1].cast<py::array_t<int>>();
    auto py_contractors = py_tuple[2].cast<py::array_t<int>>();

    size_t works_count       = py_order.size();    // without inseparables
    size_t resources_count   = py_resources.shape(1) - 1;
    size_t contractors_count = py_contractors.shape(0);

    auto *chromosome = new Chromosome(works_count, resources_count, contractors_count);

    // TODO Find the way to faster copy from NumPy ND array.
    //  !!!Attention!!! You can't just memcpy()! Plain NumPy array is not
    //  C-aligned!
    auto &order = chromosome->getOrder();
    const auto* py_order_ptr = py_order.data();
    for (int i = 0; i < works_count; i++) {
        *order[i] = py_order_ptr[i];
    }

    auto &resources = chromosome->getResources();
    const auto* py_resources_ptr = py_resources.data();
    for (size_t work = 0; work < works_count; work++) {
        for (size_t worker = 0; worker < resources_count + 1; worker++) {
            resources[work][worker] = py_resources_ptr[work * (resources_count + 1) + worker];
        }
    }

    auto &contractors = chromosome->getContractors();
    const auto* py_contractors_ptr = py_contractors.data();
    for (int contractor = 0; contractor < contractors_count; contractor++) {
        for (int worker = 0; worker < resources_count; worker++) {
            contractors[contractor][worker] = py_contractors_ptr[contractor * resources_count + worker];
        }
    }

    return chromosome;
}

vector<Chromosome *> decodeChromosomes(const py::handle &incoming) {
    return PyCodec::fromList(incoming, decodeChromosome);
}

py::object encodeChromosome(Chromosome *incoming) {
    int dims_order[] { incoming->getOrder().size() };
    int dims_resources[] { incoming->getResources().height(),
                               incoming->getResources().width() };
    int dims_contractors[] { incoming->getContractors().height(),
                                 incoming->getContractors().width() };

    cout << dims_order[0] << endl;
    cout << dims_resources[0] << " " << dims_resources[1] << endl;
    cout << dims_contractors[0] << " " << dims_contractors[1] << endl;

    auto py_order = py::array_t<int>(dims_order);
    auto py_resources = py::array_t<int>(dims_resources);
    auto py_contractors = py::array_t<int>(dims_contractors);

    auto py_order_ptr = py_order.mutable_unchecked();
    for (int i = 0; i < dims_order[0]; i++) {
        py_order_ptr[i] = *incoming->getOrder()[i];
    }

    auto py_resources_ptr = py_resources.mutable_unchecked();
    for (int work = 0; work < dims_resources[0]; work++) {
        for (int resource = 0; resource < dims_resources[1]; resource++) {
            py_resources_ptr[work * dims_resources[1] + resource] = incoming->getResources()[work][resource];
        }
    }

    auto py_contractors_ptr = py_contractors.mutable_unchecked();
    for (int contractor = 0; contractor < dims_contractors[0]; contractor++) {
        for (int resource = 0; resource < dims_contractors[1]; resource++) {
            py_contractors_ptr[contractor * dims_contractors[1] + resource] = incoming->getContractors()[contractor][resource];
        }
    }

    return py::make_tuple(py_order, py_resources, py_contractors).cast<py::object>();
}

py::list encodeChromosomes(const vector<Chromosome *> &incoming) {
    return PyCodec::toList(incoming, encodeChromosome);
}
}    // namespace PythonDeserializer
