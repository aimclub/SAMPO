#ifndef SAMPO_SORTED_LIST_H
#define SAMPO_SORTED_LIST_H

#include <set>

#include "native/schemas/dtime.h"

enum EventType {
    START,
    END
};

template <typename T>
struct ScheduleEvent {
    EventType type;
    Time time;
    int event_idx;
    T obj;
};

template <typename T>
class EventSortedList {
private:
    std::set<ScheduleEvent<T>> data;
public:
    size_t bisect_right(const ScheduleEvent<T>& value);

    size_t bisect_left(const ScheduleEvent<T>& value);

    ScheduleEvent<T>& operator[](size_t i);
};

#endif //SAMPO_SORTED_LIST_H
