#pragma once

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include "native/schemas/dtime.h"
#include "native/schemas/array2d.h"

using namespace std;

struct Zone {
    string name;
    int status;

	Zone(string name, int status) : name(name), status(status) {}
};

class ZoneStatuses {
	virtual int statuses_available() {
        return 2;
    }

	virtual bool match_status(int status_to_check, int required_status) {
        return status_to_check == required_status;
    }
};

class DefaultNotStatedZoneStatuses : ZoneStatuses {
public:
	int statuses_available() override {
		return 3;
	}

	bool match_status(int status_to_check, int required_status) override {
		return required_status == 0 || status_to_check == required_status;
	}
};

struct ZoneConfiguration {
	unordered_map<string, int> start_statuses;
	Array2D<int> time_costs;
	ZoneStatuses statuses;

    ZoneConfiguration(unordered_map<string, int> start_statuses,
                      Array2D<int> time_costs,
                      ZoneStatuses statuses) : start_statuses(start_statuses),
                                               time_costs(time_costs),
                                               statuses(statuses) {

    }

	int change_cost(int from_status, int to_status) {
		return time_costs[from_status][to_status];
	}
};


struct ZoneTransition {
    string name;
    int from_status;
    int	to_status;
    Time& start_time;
    Time& end_time;

	ZoneTransition(string name,
                   int from_status,
                   int to_status,
                   Time& start_time,
                   Time& end_time) : name(name),
                                     from_status(from_status),
                                     to_status(to_status),
                                     start_time(start_time),
                                     end_time(end_time) {}
};
