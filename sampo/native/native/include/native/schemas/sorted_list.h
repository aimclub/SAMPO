#ifndef SAMPO_SORTED_LIST_H
#define SAMPO_SORTED_LIST_H

#include <set>

#include "native/schemas/dtime.h"

enum EventType {
    START = 0,
    END = 1
};

template <typename T>
class ScheduleEvent {
private:
    EventType type;
    Time time;
    int event_idx;
    T* obj;
public:
    ScheduleEvent(EventType type, Time time, int event_idx, T* obj = nullptr)
        : type(type), time(time), event_idx(event_idx), obj(obj) {}
};

template <typename T>
bool event_cmp(const ScheduleEvent<T> &a, const ScheduleEvent<T> &b) {
    if (a.time != b.time) {
        return a.time > b.time;
    }
    if (a.type != b.type) {
        return a.type > b.type;
    }
    return a.event_idx > b.event_idx;
}

template <typename T>
class EventSortedList {
private:
    std::set<ScheduleEvent<T>, decltype(event_cmp<T>)*> data;
public:
    size_t bisect_right(const ScheduleEvent<T>& value) const;

    size_t bisect_right(const Time& timestamp) const;

    size_t bisect_left(const ScheduleEvent<T>& value) const;

    size_t bisect_left(const Time& timestamp) const;

    ScheduleEvent<T>& operator[](size_t i);

    void add(const ScheduleEvent<T>& value);

    size_t size() const;
};

#endif //SAMPO_SORTED_LIST_H
