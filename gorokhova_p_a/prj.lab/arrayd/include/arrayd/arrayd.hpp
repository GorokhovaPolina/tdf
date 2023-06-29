#ifndef ARRAYD_HPP
#define ARRAYD_HPP

#include <cstdint>
#include <iosfwd>

class ArrayD {
public:
    explicit ArrayD(int s = 0);
    ArrayD(const ArrayD& other);
    ArrayD& operator=(const ArrayD& other);
    
    ~ArrayD() = default;
    
    const double& operator[](const int& index) const;
    double& operator[](const int& index);
    
    [[nodiscard]] int32_t ssize() const;

    void resize(const int& size);
    void insert(const int& i, const double& elem);
    void remove(const int& i);
 
    std::istream& ReadFrom(std::istream& istrm);
    std::ostream& WriteTo(std::ostream& ostrm); 

private:
    double* data = nullptr;
    std::ptrdiff_t ssize_ = 0;
    std::ptrdiff_t capacity_ = 0;
};

std::ostream& operator<<(std::ostream& ostrm, ArrayD& array);
std::istream& operator>>(std::istream& istrm, ArrayD& array);

#endif