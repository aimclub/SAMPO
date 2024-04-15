#include "native/scheduler/timeline/general_timeline.h"

template <typename T>
void GeneralTimeline<T>::update_timeline(Time start_time, Time exec_time, T* obj) {
    timeline.add(ScheduleEvent(EventType::START, start_time, next_idx, obj));
    timeline.add(ScheduleEvent(EventType::END, start_time + exec_time, next_idx, obj));
    next_idx++;
}

template <typename T>
std::tuple<Time, Time, T> GeneralTimeline<T>::operator[](size_t i) {
    return timeline[i];
}

template <typename T>
size_t GeneralTimeline<T>::size() const {
    return timeline.size();
}
