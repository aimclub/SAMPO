#pragma once

#include <tuple>
#include <vector>

#include "works.h"
#include "dtime.h"
#include "scheduled_work.h"

using namespace std;

using swork_dict_t = unordered_map<string, ScheduledWork>;

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
        const EdgeType &type = EdgeType::None
    )
        : start(start), finish(finish), lag(lag), type(type) { }
};

class GraphNode {
private:
    WorkUnit *work_unit;
    std::vector<GraphEdge> parent_edges;
    std::vector<GraphEdge> children_edges;
public:
    // Create plain node with all edge types = FS
    explicit GraphNode(WorkUnit *work_unit, const std::vector<GraphNode *> &parents = {}) : work_unit(work_unit) {
        for (auto* p : parents) {
            GraphEdge edge(p, this, 0, EdgeType::FinishStart);
            parent_edges.push_back(edge);
            p->children_edges.push_back(edge);
        }
    }

    // Create node with custom edge types
    GraphNode(WorkUnit *workUnit, const std::vector<tuple<GraphNode *, float, EdgeType>> &parents)
        : GraphNode(workUnit) {
        add_parents(parents);
    }

    void add_parents(const std::vector<tuple<GraphNode *, float, EdgeType>> &parents) {
        for (const auto&[p, lag, type] : parents) {
            auto edge = GraphEdge(p, this, lag, type);
            parent_edges.emplace_back(edge);
            p->children_edges.emplace_back(edge);
        }
    }

    GraphNode* inseparable_son() const {
        for (const GraphEdge &son : children_edges) {
            if (son.type == EdgeType::InseparableFinishStart) {
                return son.finish;
            }
        }
        return nullptr;
    }

    GraphNode* inseparable_parent() const {
        for (const GraphEdge &parent : parent_edges) {
            if (parent.type == EdgeType::InseparableFinishStart) {
                return parent.start;
            }
        }
        return nullptr;
    }

    std::vector<GraphNode *> parents() const {
        auto parents = std::vector<GraphNode *>();
        for (const GraphEdge &parent : parent_edges) {
            parents.push_back(parent.start);
        }
        return parents;
    }

    std::vector<GraphNode *> children() const {
        auto children = std::vector<GraphNode *>();
        for (const GraphEdge &child : children_edges) {
            children.push_back(child.start);
        }
        return children;
    }

    std::vector<GraphEdge> edgesTo() const {
        return parent_edges;
    }

    std::vector<GraphEdge> edgesFrom() const {
        return children_edges;
    }

    WorkUnit *getWorkUnit() const {
        return work_unit;
    }

    const string& id() const {
        return getWorkUnit()->id;
    }

    inline bool is_inseparable_parent() {
        return inseparable_son() != nullptr;
    }

    inline bool is_inseparable_son() {
        return inseparable_parent() != nullptr;
    }

    std::vector<const GraphNode *> getInseparableChainWithSelf() const {
        std::vector<const GraphNode *> chain;
        chain.push_back(this);
        auto* child = inseparable_son();
        if (child) {
            auto sub_chain = child->getInseparableChainWithSelf();
            chain.insert(chain.end(), sub_chain.begin(), sub_chain.end());
        }
        return chain;
    }

    Time min_start_time(const swork_dict_t &node2swork) const {
        Time time;
        for (const auto& edge : this->parent_edges) {
            auto it = node2swork.find(edge.start->id());
            if (it == node2swork.end()) {
                return Time::inf();
            }
            time = max(time, it->second.finish_time() + (int) edge.lag);
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
    explicit WorkGraph(const std::vector<GraphNode *> &nodes) : nodes(nodes) {
        if (!nodes.empty()) {
            start = nodes[0];
            finish = nodes[nodes.size() - 1];
        }
    }
};
