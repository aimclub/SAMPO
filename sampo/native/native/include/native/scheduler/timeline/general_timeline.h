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
    void update_timeline(Time start_time, Time exec_time, T* obj) {
        timeline.add(ScheduleEvent(EventType::START, start_time, next_idx, obj));
        timeline.add(ScheduleEvent(EventType::END, start_time + exec_time, next_idx, obj));
        next_idx++;
    }

    std::set<ScheduleEvent<T>>::const_iterator iterator() {
        return timeline.begin();
    }

    const Time& get_time_at(size_t i) const {
        return timeline[i].time;
    }

    size_t size() const {
        return timeline.size();
    }

    bool is_end(std::set<ScheduleEvent<T>>::const_iterator it) {
        return it == timeline.end();
    }
};

#endif //SAMPO_GENERAL_TIMELINE_H
