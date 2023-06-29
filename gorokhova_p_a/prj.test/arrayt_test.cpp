#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "arrayt/arrayt.hpp"
#include "doctest/doctest.h"


TEST_CASE("Test construct and copy") {
    SUBCASE("Test with normal size") {
        CHECK_NOTHROW(ArrayT<int>(3));
    }
    SUBCASE("Test with size = 0") {
        CHECK_NOTHROW(ArrayT<int>(0));
    }
    SUBCASE("Test with negative size") {
        CHECK_THROWS(ArrayT<int>(-1));
    }
    SUBCASE("Test copy") {
        ArrayT<int> x(1);
        x[0] = 1;
        ArrayT<int> y = x;
        y[0] = 2;
        CHECK((x[0] != y[0]));
    }
    SUBCASE("Test copy yourself") {
        ArrayT<int> x(2);
        x[0] = 404;
        x[1] = 404;
        x = x;
        CHECK((x[0] == 404 && x[1] == 404));
    }
}
TEST_CASE("Test with operator[]") {
    SUBCASE("Test with middle index") {
        ArrayT<int> x(3);
        CHECK_NOTHROW(x[1] = 404);
        CHECK((x[1] == 404));
    }
    SUBCASE("Test with begin index") {
        ArrayT<int> x(3);
        CHECK_NOTHROW(x[0] = 404);
        CHECK((x[0] == 404));
    }
    SUBCASE("Test with last index") {
        ArrayT<int> x(3);
        CHECK_NOTHROW(x[2] = 404);
        CHECK((x[2] == 404));
    }
    SUBCASE("Test with out of range index > size") {
        ArrayT<int> x(3);
        CHECK_THROWS(x[4]);
    }
    SUBCASE("Test with out of range index == size") {
        ArrayT<int> x(3);
        CHECK_THROWS(x[3]);
    }
    SUBCASE("Test with out of range index < 0") {
        ArrayT<int> x(3);
        CHECK_THROWS(x[-1]);
    }
}

TEST_CASE("Test of ssize, resize, insert, remove function") {
    SUBCASE("size == size") {
        CHECK((ArrayT<int>(5).ssize() == 5));
    }
    SUBCASE("Resize to size+1") {
        ArrayT<int> x(1);
        x.resize(x.ssize() + 1);
        CHECK_NOTHROW(x[1]);
    }
    SUBCASE("Resize to 0") {
        ArrayT<int> x(1);
        CHECK_THROWS(x.resize(0));
    }
    SUBCASE("Resize to size < 0") {
        ArrayT<int> x(1);
        CHECK_THROWS(x.resize(-1));
    }
    ArrayT<int> test(3);
    ArrayT<int> test_i(0);
    int size = test.ssize();
    for (int i = 0; i < size; i++) {
        test[i] = i + 1;
    }
    SUBCASE("insert to middle") {
        test_i = test;
        test_i.insert(1, 4);
        CHECK((test_i[0] == test[0]));
        CHECK((test_i[1] == 4));
        CHECK((test_i[2] == test[1]));
        CHECK((test_i[3] == test[2]));
        CHECK((test_i.ssize() == size + 1));
    }
    SUBCASE("insert to begin") {
        test_i = test;
        test_i.insert(0, 4);
        CHECK((test_i[0] == 4));
        CHECK((test_i[1] == test[0]));
        CHECK((test_i[2] == test[1]));
        CHECK((test_i[3] == test[2]));
        CHECK((test_i.ssize() == size + 1));
    }
    SUBCASE("insert to end") {
        test_i = test;
        test_i.insert(3, 4);
        CHECK((test_i[0] == test[0]));
        CHECK((test_i[1] == test[1]));
        CHECK((test_i[2] == test[2]));
        CHECK((test_i[3] == 4));
        CHECK((test_i.ssize() == size + 1));
    }
    SUBCASE("insert to invalid position < 0") {
        test_i = test;
        CHECK_THROWS(test_i.insert(-1, 4));
    }
    SUBCASE("insert to invalid position > size") {
        test_i = test;
        CHECK_THROWS(test_i.insert(4, 4));
    }

    SUBCASE("remove from middle") {
        test_i = test;
        test_i.remove(1);
        CHECK((test_i[0] == test[0]));
        CHECK((test_i[1] == test[2]));
        CHECK((test_i.ssize() == size - 1));
    }
    SUBCASE("remove from begin") {
        test_i = test;
        test_i.remove(0);
        CHECK((test_i[0] == test[1]));
        CHECK((test_i[1] == test[2]));
        CHECK((test_i.ssize() == size - 1));
    }
    SUBCASE("remove from end") {
        test_i = test;
        test_i.remove(2);
        CHECK((test_i[0] == test[0]));
        CHECK((test_i[1] == test[1]));
        CHECK((test_i.ssize() == size - 1));
    }
    SUBCASE("remove from invalid position < 0") {
        test_i = test;
        CHECK_THROWS(test_i.remove(-1));
    }
    SUBCASE("remove from invalid position == size") {
        test_i = test;
        CHECK_THROWS(test_i.remove(3));
    }
    SUBCASE("remove from invalid position > size") {
        test_i = test;
        CHECK_THROWS(test_i.remove(4));
    }
}
TEST_CASE("My tests") {
    ArrayT<int> a(4);

    SUBCASE("Test: inserting value") {
        for (int i = 0; i < 4; i++) {
            a[i] = double(i + 1);
        }
        a.insert(4, 5);

        for (int i = 0; i < 5; i++) {
            CHECK(a[i] == double(i + 1));
        }
        CHECK(a.ssize() == 5);

        a.insert(5, 5);
        CHECK(a.ssize() == 6);
    }

    SUBCASE("Test: removing values") {
        ArrayT<int> b(4);
        for (int i = 0; i < 4; i++) {
            b[i] = double(i + 1);
        }
        b.insert(4, 5);

        b.insert(5, 5);
        CHECK(b.ssize() == 6);

        b.remove(2);
        CHECK(b[0] == 1);
        CHECK(b[1] == 2);
        CHECK(b[2] == 4);
        CHECK(b[3] == 5);
        CHECK(b.ssize() == 5);
    }
}