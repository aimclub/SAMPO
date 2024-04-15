#include "native/schemas/sorted_list.h"

using namespace std;

template <typename T>
size_t EventSortedList<T>::bisect_right(const ScheduleEvent<T>& value) const {
    // TODO Test it is equals to Python variant
    size_t bound = data.upper_bound(value);
    if (bound > 0 && data[bound - 1] == value) {
        return bound - 1;
    }
    return bound;
}

template <typename T>
size_t EventSortedList<T>::bisect_left(const ScheduleEvent<T>& value) const {
    // TODO Test it is equals to Python variant
    return data.lower_bound(value);
}

template <typename T>
ScheduleEvent<T>& EventSortedList<T>::operator[](size_t i) {
    return data[i];
}

template <typename T>
size_t EventSortedList<T>::bisect_left(const Time& timestamp) const {
    return bisect_left(ScheduleEvent<T>(EventType::END, timestamp, TIME_INF));
}

template <typename T>
size_t EventSortedList<T>::bisect_right(const Time& timestamp) const {
    return bisect_right(ScheduleEvent<T>(EventType::END, timestamp, TIME_INF));
}

template <typename T>
void EventSortedList<T>::add(const ScheduleEvent<T>& value) {
    data.insert(value);
}
