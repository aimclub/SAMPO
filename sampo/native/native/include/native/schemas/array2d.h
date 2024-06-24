#ifndef SAMPO_ARRAY2D_H
#define SAMPO_ARRAY2D_H

#include <iostream>

using namespace std;

template <typename T>
class Array2D {
private:
    size_t length = 0;
    size_t stride = 0;
    T *data       = nullptr;
    bool shallow  = true;

public:
    Array2D() = default;

    Array2D(size_t length, size_t stride, T *data)
            : length(length), stride(stride), data(data), shallow(true) { }

    //    Array2D(size_t length, size_t stride)
    //        : Array2D(length, stride, (T*) malloc(length * sizeof(T))),
    //        shallow(true) {}

    explicit Array2D(const Array2D &other)
            : Array2D(other.length, other.stride, other.shallow) {
        //        memcpy(this->data, other.data, length * sizeof(T));
        cout << "Copy" << endl;
    }

    ~Array2D() {
        if (!shallow) {
            cout << "Free array" << endl;
            //        free(this->data);
        }
    }

    // int* to use this operator as 2D array. To use as 1D array, follow this
    // call with '*'.
    int *operator[](size_t i) {
        return this->data + i * stride;
    }

    // shallow copy
    Array2D<T> &operator=(const Array2D &other) {
        this->length = other.length;
        this->stride = other.stride;
        this->data   = other.data;
        return *this;
    }

    int width() {
        return stride;
    }

    int height() {
        return length / stride;
    }

    int size() {
        return length;
    }
};

#endif //SAMPO_ARRAY2D_H
