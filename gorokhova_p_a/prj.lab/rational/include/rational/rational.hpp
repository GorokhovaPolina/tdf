#ifndef RATIONAL_HPP
#define RATIONAL_HPP

#include <cstdint>
#include <iosfwd>

// Greatest common division
int32_t GCD(int32_t a, int32_t b);
class Rational {
public:
    // Constructs
    Rational() = default;
    Rational(const Rational&) = default;
    Rational(const int32_t p, const int32_t q);
    explicit Rational(const int32_t p) noexcept;
    ~Rational() = default;
    // Algebra declaration
    Rational operator-();
    Rational& operator+=(const Rational& rhs);
    Rational& operator-=(const Rational& rhs);
    Rational& operator/=(const Rational& rhs);
    Rational& operator*=(const Rational& rhs);
    Rational& operator+=(const int32_t& rhs);
    Rational& operator-=(const int32_t& rhs);
    Rational& operator/=(const int32_t& rhs);
    Rational& operator*=(const int32_t& rhs);
    Rational& operator++();
    Rational operator++(int);
    Rational& operator--();
    Rational operator--(int);
    // Compare
    bool operator==(const Rational& rhs) const noexcept;
    bool operator>(const Rational& rhs) const noexcept;
    bool operator!=(const Rational& rhs) const noexcept;
    bool operator<(const Rational& rhs) const noexcept;
    bool operator<=(const Rational& rhs) const noexcept;
    bool operator>=(const Rational& rhs) const noexcept;
    // Input/ouput
    std::ostream& writeTo(std::ostream& ostrm) const noexcept;
    std::istream& readFrom(std::istream& istrm) noexcept;

private:
    int32_t num_{ 0 };
    int32_t denum_{ 1 };
    static const char separator{ '/' };
};

// Input/output
std::ostream& operator<<(std::ostream& ostrm, const Rational& rhs);
std::istream& operator>>(std::istream& istrm, Rational& rhs);
// Extra algebra
Rational operator+(const Rational& lhs, const Rational& rhs);
Rational operator-(const Rational& lhs, const Rational& rhs);
Rational operator/(const Rational& lhs, const Rational& rhs);
Rational operator*(const Rational& lhs, const Rational& rhs);
// Compare with int
bool operator==(const Rational& lhs, const int32_t& rhs);
bool operator!=(const Rational& lhs, const int32_t& rhs);
bool operator>(const Rational& lhs, const int32_t& rhs);
bool operator<(const Rational& lhs, const int32_t& rhs);
bool operator<=(const Rational& lhs, const int32_t& rhs);
bool operator>=(const Rational& lhs, const int32_t& rhs);

#endif