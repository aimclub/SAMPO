#ifndef SAMPO_GENERAL_TIMELINE_H
#define SAMPO_GENERAL_TIMELINE_H

#include "native/schemas/dtime.h"
#include "native/schemas/sorted_list.h"

template <typename T>
class GeneralTimeline {
private:
    EventSortedList<T> timeline;
    size_t next_idx;
public:
    void update_timeline(Time start_time, Time exec_time, T* obj);

    std::tuple<Time, Time, T> operator[](size_t i);

    size_t size() const;
};

#endif //SAMPO_GENERAL_TIMELINE_H
