//
// Created by Quarter on 23.06.2023.
//

#ifndef NATIVE_EXTERNAL_H
#define NATIVE_EXTERNAL_H

#include <string>

namespace External {
#ifdef WIN32
    static const std::string timeEstimatorLibPath = "wte.dll";
#endif
#ifdef __linux__
    static const std::string timeEstimatorLibPath = "./wte.so";
#endif
#ifdef __APPLE__
    static const std::string timeEstimatorLibPath = "./wte.dylib";
#endif
}

#endif //NATIVE_EXTERNAL_H
