#ifndef WORKGRAPH_H
#define WORKGRAPH_H

#include <tuple>
#include <vector>

#include "basic_types.h"
#include "dtime.h"
#include "evaluator_types.h"

using namespace std;

// Attention! Here we have an idiom of fully immutable data types, so
// it's normal situation that you can't directly map C++ code to the
// corresponding Python code. In addition, there can be some non-trivial
// optimizations, so come on.

enum EdgeType {
    InseparableFinishStart,
    LagFinishStart,
    StartStart,
    FinishFinish,
    FinishStart,
    None
};

class WorkerReq {
public:
    string kind;
    Time volume;
    int min_count;
    int max_count;

    explicit WorkerReq(string kind, Time volume, int min_count, int max_count)
        : kind(std::move(kind)),
          volume(volume),
          min_count(min_count),
          max_count(max_count) { }
};

class WorkUnit : public Identifiable {
public:
    std::vector<WorkerReq> worker_reqs;
    string name;
    float volume;
    bool isServiceUnit;

    explicit WorkUnit(
        string name = "",
        const std::vector<WorkerReq> &worker_reqs = std::vector<WorkerReq>(),
        float volume                              = 1,
        bool isServiceUnit                        = false
    )
        : worker_reqs(worker_reqs),
          name(std::move(name)),
          volume(volume),
          isServiceUnit(isServiceUnit) { }
};

class GraphNode;

class GraphEdge {
public:
    GraphNode *start;
    GraphNode *finish;
    float lag;
    EdgeType type;

    explicit GraphEdge(
        GraphNode *start,
        GraphNode *finish,
        float lag     = 0,
        EdgeType type = EdgeType::None
    )
        : start(start), finish(finish), lag(lag), type(type) { }
};

class GraphNode {
private:
    WorkUnit *workUnit;
    std::vector<GraphEdge> parentEdges   = std::vector<GraphEdge>();
    std::vector<GraphEdge> childrenEdges = std::vector<GraphEdge>();

public:
    explicit GraphNode(WorkUnit *workUnit) : workUnit(workUnit) {};

    // Create plain node with all edge types = FS
    GraphNode(WorkUnit *workUnit, std::vector<GraphNode *> &parents)
        : GraphNode(workUnit) {
        for (auto p : parents) {
            auto edge = GraphEdge(p, this, -1, EdgeType::FinishStart);
            parentEdges.emplace_back(edge);
            p->childrenEdges.emplace_back(edge);
        }
    }

    // Create node with custom edge types
    GraphNode(
        WorkUnit *workUnit,
        std::vector<tuple<GraphNode *, float, EdgeType>> &parents
    )
        : GraphNode(workUnit) {
        for (auto &tuple : parents) {
            auto p    = get<0>(tuple);
            auto edge = GraphEdge(p, this, get<1>(tuple), get<2>(tuple));
            parentEdges.emplace_back(edge);
            p->childrenEdges.emplace_back(edge);
        }
    }

    GraphNode *inseparableSon() {
        for (GraphEdge &son : childrenEdges) {
            if (son.type == EdgeType::InseparableFinishStart) {
                return son.finish;
            }
        }
        return nullptr;
    }

    GraphNode *inseparableParent() {
        for (GraphEdge &parent : parentEdges) {
            if (parent.type == EdgeType::InseparableFinishStart) {
                return parent.start;
            }
        }
        return nullptr;
    }

    std::vector<GraphNode *> parents() {
        auto parents = std::vector<GraphNode *>();
        for (GraphEdge &parent : parentEdges) {
            parents.push_back(parent.start);
        }
        return parents;
    }

    std::vector<GraphNode *> children() {
        auto children = std::vector<GraphNode *>();
        for (GraphEdge &child : childrenEdges) {
            children.push_back(child.start);
        }
        return children;
    }

    std::vector<GraphEdge> edgesTo() {
        return parentEdges;
    }

    std::vector<GraphEdge> edgesFrom() {
        return childrenEdges;
    }

    WorkUnit *getWorkUnit() {
        return workUnit;
    }

    string id() {
        return getWorkUnit()->id;
    }

    std::vector<GraphNode *> getInseparableChainWithSelf() {
        auto chain = std::vector<GraphNode *>();
        chain.push_back(this);
        auto child = inseparableSon();
        if (child) {
            auto subChain = child->getInseparableChainWithSelf();
            chain.insert(chain.end(), subChain.begin(), subChain.end());
        }
        return chain;
    }

    Time min_start_time(swork_dict_t &node2swork) {
        Time time;
        for (auto& edge : this->parentEdges) {
            auto it = node2swork.find(edge.start->id());
            if (it == node2swork.end()) {
                return Time.is_inf();
            }
            time = max(time, it->second.start_time());
        }
        return time;
    }
};

class WorkGraph {
public:
    GraphNode *start;
    GraphNode *finish;
    std::vector<GraphNode *> nodes;

    // `nodes` param MUST be a std::vector with topologically-ordered nodes
    explicit WorkGraph(const std::vector<GraphNode *> &nodes)
        : start(nodes[0]), finish(nodes[nodes.size() - 1]), nodes(nodes) { }
};

#endif    // WORKGRAPH_H
