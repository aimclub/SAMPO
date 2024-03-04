#ifndef SAMPO_GENERAL_TIMELINE_H
#define SAMPO_GENERAL_TIMELINE_H

#include "native/schemas/dtime.h"

template <typename T>
class GeneralTimeline {
public:
    void update_timeline(Time start_time, Time exec_time, T obj);

    T operator[](size_t i);

    size_t size();
};

#endif //SAMPO_GENERAL_TIMELINE_H
