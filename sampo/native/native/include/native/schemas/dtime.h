#ifndef NATIVE_TIME_H
#define NATIVE_TIME_H

#define TIME_INF 2000000000

class Time {
private:
    int value;

public:
    Time(int value = 0);

    Time(Time const &other);

    static Time inf();

    static Time unassigned();

    bool is_unassigned() const;

    bool is_inf() const;

    int val() const;

    Time operator+(const Time &other) const;

    Time operator+(int other) const;

    Time operator-(const Time &other) const;

    Time operator-(int other) const;

    Time operator*(const Time &other) const;

    Time operator*(int other) const;

    Time operator/(const Time &other) const;

    Time operator/(int other) const;

    bool operator<(const Time &other) const;

    bool operator<(int other) const;

    bool operator>(const Time &other) const;

    bool operator>(int other) const;

    bool operator<=(const Time &other) const;

    bool operator<=(int other) const;

    bool operator>=(const Time &other) const;

    bool operator>=(int other) const;

    bool operator==(const Time &other) const;

    bool operator==(int other) const;

    Time& operator++(int n);

    Time& operator--(int n);

    Time& operator+=(int other);

    Time& operator+=(const Time &other);

    Time& operator-=(int other);

    Time& operator-=(const Time &other);
};

#endif    // NATIVE_TIME_H
