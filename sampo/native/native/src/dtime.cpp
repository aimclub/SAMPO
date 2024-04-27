#include <algorithm>

#include "native/schemas/dtime.h"

#define TIME_INF 2000000000

Time::Time(int value) {
    if (value > TIME_INF) {
        value = TIME_INF;
    }
    else if (value < -1) {
        value = -1;
    }
    this->value = value;
}

Time::Time(const Time &other) = default;

int Time::val() const {
    return this->value;
}

Time Time::inf() {
    static Time inf(TIME_INF);    // caching the inf instance
    return inf;
}

Time Time::unassigned() {
    static Time inf(-1);    // caching the inf instance
    inf.value = -1;
    return inf;
}

bool Time::is_unassigned() const {
    return this->value == -1;
}

bool Time::is_inf() const {
    return this->value == TIME_INF;
}

// Copy-paste body between functions with different arg type
// is here to avoid functional call overhead. Time instances
// should be as fast as possible.

Time Time::operator+(const Time &other) const {
    return Time(this->value + other.value);
}

Time Time::operator+(int other) const {
    return Time(this->value + other);
}

Time Time::operator-(const Time &other) const {
    return Time(this->value - other.value);
}

Time Time::operator-(int other) const {
    return Time(this->value - other);
}

Time Time::operator*(const Time &other) const {
    return Time(this->value * other.value);
}

Time Time::operator*(int other) const {
    return Time(this->value * other);
}

Time Time::operator/(const Time &other) const {
    return Time(this->value / other.value);
}

Time Time::operator/(int other) const {
    return Time(this->value / other);
}

bool Time::operator<(const Time &other) const {
    return this->value < other.value;
}

bool Time::operator<(int other) const {
    return this->value < other;
}

bool Time::operator>(const Time &other) const {
    return this->value > other.value;
}

bool Time::operator>(int other) const {
    return this->value > other;
}

bool Time::operator<=(const Time &other) const {
    return this->value <= other.value;
}

bool Time::operator<=(int other) const {
    return this->value <= other;
}

bool Time::operator>=(const Time &other) const {
    return this->value >= other.value;
}

bool Time::operator>=(int other) const {
    return this->value >= other;
}

bool Time::operator==(const Time &other) const {
    return this->value == other.value;
}

bool Time::operator==(int other) const {
    return this->value == other;
}

Time& Time::operator++(int n) {
    if (n == 0)
        n = 1;
    return this->operator+=(n);
}

Time& Time::operator--(int n) {
    if (n == 0)
        n = 1;
    return this->operator-=(n);
}

Time& Time::operator+=(int other) {
    this->value += other;
    return *this;
}

Time& Time::operator+=(const Time &other) {
    return this->operator+=(other.value);
}

Time& Time::operator-=(int other) {
    return this->operator+=(-other);
}

Time& Time::operator-=(const Time &other) {
    return this->operator-=(other.value);
}
