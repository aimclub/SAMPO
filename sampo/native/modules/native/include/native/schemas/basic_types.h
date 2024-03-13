#pragma once

#include <string>

class Identifiable {
public:
    std::string id;

    explicit Identifiable(std::string id = "") : id(std::move(id)) {}
};
