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
public:
    EventType type;
    Time time;
    int event_idx;
    T* obj;

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
    std::set<ScheduleEvent<T>>::const_iterator begin() const {
        return data.cbegin();
    }

    std::set<ScheduleEvent<T>>::const_iterator end() const {
        return data.cend();
    }

    std::set<ScheduleEvent<T>>::const_iterator bisect_right(const ScheduleEvent<T>& value) const {
        // TODO Test it is equals to Python variant
        std::set<ScheduleEvent<T>>::const_iterator bound = data.upper_bound(value);
        if (bound != data.cbegin()) {
            bound++;
            if (bound == value) {
                return bound - 1;
            }
        }
        return bound;
    }

    std::set<ScheduleEvent<T>>::const_iterator bisect_right(const Time& timestamp) const {
        return bisect_right(ScheduleEvent<T>(EventType::END, timestamp, TIME_INF));
    }

    std::set<ScheduleEvent<T>>::const_iterator bisect_left(const ScheduleEvent<T>& value) const {
        // TODO Test it is equals to Python variant
        return data.lower_bound(value);
    }

    std::set<ScheduleEvent<T>>::const_iterator bisect_left(const Time& timestamp) const {
        return bisect_left(ScheduleEvent<T>(EventType::END, timestamp, TIME_INF));
    }

    void add(const ScheduleEvent<T>& value) {
        data.insert(value);
    }

    size_t size() const;
};

#endif //SAMPO_SORTED_LIST_H
