#ifndef ARRAYT_HPP
#define ARRAYT_HPP


#include <cstddef>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

template <typename T>
class ArrayT {
public:
    // Конструкторы
    explicit ArrayT(const int s = 0);
    ArrayT(const ArrayT<T>& other);
    ArrayT& operator=(const ArrayT<T>& other);
    // Деструктор
    ~ArrayT() = default;
    // Оператор итерирования
    const T& operator[](const int& index) const;
    T& operator[](const int& index);
    // Взятие размера
    [[nodiscard]] int32_t ssize() const;
    // Изменение размера
    void resize(const int& size);
    void insert(const int& i, const T& elem);
    void remove(const int& i);
    // Ввод/вывод
    std::istream& ReadFrom(std::istream& istrm);
    std::ostream& WriteTo(std::ostream& ostrm); //const

private:
    T* data = nullptr;
    std::ptrdiff_t ssize_ = 0;
    std::ptrdiff_t capacity_ = 0;
};
template <typename T>
std::ostream& operator<<(std::ostream& ostrm, ArrayT<T>& array);
template <typename T>
std::istream& operator>>(std::istream& istrm, ArrayT<T>& array);

template <typename T>
ArrayT<T>::ArrayT(const int s) : ssize_(s) {
    if (s < 0) { throw std::invalid_argument("invalid size"); }
    capacity_ = ssize_ * 2;
    data = new T[capacity_]{};
}

template <typename T>
ArrayT<T>::ArrayT(const ArrayT<T>& other) : ssize_(other.ssize_) {
    capacity_ = ssize_ * 2;
    data = new T[capacity_];
}

template <typename T>
ArrayT<T>& ArrayT<T>::operator=(const ArrayT<T>& other) {
    if (data != other.data) {
        delete[] data;
        ssize_ = other.ssize();
        capacity_ = ssize_ * 2;
        data = new T[capacity_];
        for (int i = 0; i < ssize_; ++i)
            data[i] = other.data[i];
    }
    return *this;
}

template <typename T>
T& ArrayT<T>::operator[](const int& index) {
    if (index < 0 || index >= ssize_) {
        throw std::out_of_range("invalid index");
    }
    return data[index];
}

template <typename T>
const T& ArrayT<T>::operator[](const int& index) const {
    if (index < 0 || index >= ssize_) {
        throw std::out_of_range("invalid index");
    }
    return data[index];
}

template <typename T>
int32_t ArrayT<T>::ssize() const { return ssize_; }

template <typename T>
void ArrayT<T>::resize(const int& new_size) {
    if (new_size <= 0) { throw std::invalid_argument("invalid size"); }
    if (capacity_ < new_size) {
        T* old = data;
        capacity_ = new_size * 2;
        data = new T[capacity_];
        for (int i = 0; i < new_size; i++) {
            data[i] = old[i];
        }
        delete[] old;
    }
    ssize_ = new_size;
}

template <typename T>
void ArrayT<T>::insert(const int& i, const T& elem) {
    if (i < 0 || i > ssize_) { throw std::out_of_range("invalid index"); }
    (*this).resize(ssize_ + 1);
    for (int j = ssize_ - 1; j > i; --j) {
        data[j] = data[j - 1];
    }
    data[i] = elem;
}

template <typename T>
void ArrayT<T>::remove(const int& i) {
    if (i < 0 || i >= ssize_) { throw std::out_of_range("invalid index"); }
    std::rotate(data + i, data + i + 1, data + ssize_);
    (*this).resize(ssize_ - 1);
}

template <typename T>
std::istream& ArrayT<T>::ReadFrom(std::istream& istrm) {
    if (ssize_ > 0) {
        int i = 0;
        T xi = 0;
        while (!istrm.eof() && istrm.good() && i < ssize_) {
            istrm >> xi;
            data[i] = xi;
            i++;
        }
    }
    else {
        istrm.setstate(std::ios_base::failbit);
    }
    return istrm;
}

template <typename T>
std::ostream& ArrayT<T>::WriteTo(std::ostream& ostrm) {
    if (ssize_ > 0) {
        int i = 0;
        while (ostrm.good() && i < ssize_) {
            ostrm << data[i] << " ";
            i++;
        }
    }
    return ostrm;
}

template <typename T>
std::ostream& operator<<(std::ostream& ostrm, ArrayT<T>& array) {
    return array.WriteTo(ostrm);
}

template <typename T>
std::istream& operator>>(std::istream& istrm, ArrayT<T>& array) {
    return array.ReadFrom(istrm);
}
#endif