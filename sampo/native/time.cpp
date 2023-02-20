#include "time.h"

#define TIME_INF 2000000000

class Time {
private:
    int value;
public:
    explicit Time(int value) {
        if (value > TIME_INF) {
            value = TIME_INF;
        } else if (value < 0) {
            value = 0;
        }
        this->value = value;
    }

    static Time inf() {
        static Time inf(TIME_INF); // caching the inf instance
        return inf;
    }

    bool is_inf() const {
        return this->value == TIME_INF;
    }

    // Copy-paste body between functions with different arg type
    // is here to avoid functional call overhead. Time instances
    // should be as fast as possible.

    Time operator+(Time& other) const {
        return Time(this->value + other.value);
    }

    Time operator+(int other) const {
        return Time(this->value + other);
    }

    Time operator-(Time& other) const {
        return Time(this->value - other.value);
    }

    Time operator-(int other) const {
        return Time(this->value - other);
    }

    Time operator*(Time& other) const {
        return Time(this->value * other.value);
    }

    Time operator*(int other) const {
        return Time(this->value * other);
    }

    Time operator/(Time& other) const {
        return Time(this->value / other.value);
    }

    Time operator/(int other) const {
        return Time(this->value / other);
    }

    bool operator<(Time& other) const {
        return this->value < other.value;
    }

    bool operator<(int other) const {
        return this->value < other;
    }

    bool operator>(Time& other) const {
        return this->value > other.value;
    }

    bool operator>(int other) const {
        return this->value > other;
    }

    bool operator<=(Time& other) const {
        return this->value <= other.value;
    }

    bool operator<=(int other) const {
        return this->value <= other;
    }

    bool operator>=(Time& other) const {
        return this->value >= other.value;
    }

    bool operator>=(int other) const {
        return this->value >= other;
    }

    bool operator==(Time& other) const {
        return this->value == other.value;
    }

    bool operator==(int other) const {
        return this->value == other;
    }
};
