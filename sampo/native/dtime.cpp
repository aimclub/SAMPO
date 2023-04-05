#include "dtime.h"

#define TIME_INF 2000000000

Time::Time(int value) {
    if (value > TIME_INF) {
        value = TIME_INF;
    } else if (value < 0) {
        value = 0;
    }
    this->value = value;
}

Time Time::inf() {
    static Time inf(TIME_INF); // caching the inf instance
    return inf;
}

bool Time::is_inf() const {
    return this->value == TIME_INF;
}

// Copy-paste body between functions with different arg type
// is here to avoid functional call overhead. Time instances
// should be as fast as possible.

Time Time::operator+(Time& other) const {
    return Time(this->value + other.value);
}

Time Time::operator+(int other) const {
    return Time(this->value + other);
}

Time Time::operator-(Time& other) const {
    return Time(this->value - other.value);
}

Time Time::operator-(int other) const {
    return Time(this->value - other);
}

Time Time::operator*(Time& other) const {
    return Time(this->value * other.value);
}

Time Time::operator*(int other) const {
    return Time(this->value * other);
}

Time Time::operator/(Time& other) const {
    return Time(this->value / other.value);
}

Time Time::operator/(int other) const {
    return Time(this->value / other);
}

bool Time::operator<(Time& other) const {
    return this->value < other.value;
}

bool Time::operator<(int other) const {
    return this->value < other;
}

bool Time::operator>(Time& other) const {
    return this->value > other.value;
}

bool Time::operator>(int other) const {
    return this->value > other;
}

bool Time::operator<=(Time& other) const {
    return this->value <= other.value;
}

bool Time::operator<=(int other) const {
    return this->value <= other;
}

bool Time::operator>=(Time& other) const {
    return this->value >= other.value;
}

bool Time::operator>=(int other) const {
    return this->value >= other;
}

bool Time::operator==(Time& other) const {
    return this->value == other.value;
}

bool Time::operator==(int other) const {
    return this->value == other;
}
