#ifndef NATIVE_TIME_H
#define NATIVE_TIME_H

class Time {
private:
    int value;
public:
    explicit Time(int value);

    static Time inf();

    bool is_inf() const;

    Time operator+(Time &other) const;

    Time operator+(int other) const;

    Time operator-(Time &other) const;

    Time operator-(int other) const;

    Time operator*(Time &other) const;

    Time operator*(int other) const;

    Time operator/(Time &other) const;

    Time operator/(int other) const;

    bool operator<(Time &other) const;

    bool operator<(int other) const;

    bool operator>(Time &other) const;

    bool operator>(int other) const;

    bool operator<=(Time &other) const;

    bool operator<=(int other) const;

    bool operator>=(Time &other) const;

    bool operator>=(int other) const;

    bool operator==(Time &other) const;

    bool operator==(int other) const;
};

#endif //NATIVE_TIME_H
