#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "matrixs/matrixs.hpp"
#include "doctest/doctest.h"

TEST_CASE("Test of constructors") {
    SUBCASE("Test construct with <tuple> and relevant size") {
        CHECK_NOTHROW(MatrixS(MatrixS::SizeType{ 2,2 }));
    }
    SUBCASE("Test construct with <tuple> and nonrelevant size") {
        CHECK_THROWS(MatrixS(MatrixS::SizeType{ -1,0 }));
    }
    SUBCASE("Test construct with two term and relevant size") {
        CHECK_NOTHROW(MatrixS(1, 1));
    }
    SUBCASE("Test construct with two term and nonrelevant size") {
        CHECK_THROWS(MatrixS(-1, -1));
    }
    MatrixS test(2, 2);
    SUBCASE("Test construct with another MatrixS with ()") {
        test.at(0, 0) = 1;
        MatrixS another(test);
        another.at(0, 0) = 2;
        CHECK((test.at(0, 0) != another.at(0, 0)));
    }
    SUBCASE("Test construct with another MatrixS with operator=") {
        test.at(0, 0) = 1;
        MatrixS another = test;
        another.at(0, 0) = 2;
        CHECK((test.at(0, 0) != another.at(0, 0)));
    }
    SUBCASE("Test construct with another MatrixS with operator= and same MatrixS") {
        MatrixS x(1, 2);
        x.at(0, 0) = 404;
        x.at(0, 1) = 404;
        x = x;
        CHECK((x.at(0, 0) == 404 && x.at(0, 1) == 404));
    }
}

TEST_CASE("Test of .at() function") {
    MatrixS test(3, 3);
    SUBCASE("middle index") {
        CHECK_NOTHROW(test.at(1, 1) = 404);
        CHECK((test.at(1, 1) == 404));
    }
    SUBCASE("middle index (but index is SizeType)") {
        CHECK_NOTHROW(test.at(MatrixS::SizeType{ 1,1 }) = 404);
        CHECK((test.at(MatrixS::SizeType{ 1,1 }) == 404));
    }
    SUBCASE("begin index") {
        CHECK_NOTHROW(test.at(0, 0) = 404);
        CHECK((test.at(0, 0) == 404));
    }
    SUBCASE("begin index (but index is SizeType)") {
        CHECK_NOTHROW(test.at(MatrixS::SizeType{ 0,0 }) = 404);
        CHECK((test.at(MatrixS::SizeType{ 0,0 }) == 404));
    }
    SUBCASE("end index") {
        CHECK_NOTHROW(test.at(2, 2) = 404);
        CHECK((test.at(2, 2) == 404));
    }
    SUBCASE("end index (but index is SizeType)") {
        CHECK_NOTHROW(test.at(MatrixS::SizeType{ 2,2 }) = 404);
        CHECK((test.at(MatrixS::SizeType{ 2,2 }) == 404));
    }
    SUBCASE("out of range index == size") {
        CHECK_THROWS(test.at(3, 0));
        CHECK_THROWS(test.at(0, 3));
        CHECK_THROWS(test.at(3, 3));
    }
    SUBCASE("out of range index == size (but index is SizeType)") {
        CHECK_THROWS(test.at(MatrixS::SizeType{ 3,0 }));
        CHECK_THROWS(test.at(MatrixS::SizeType{ 0,3 }));
        CHECK_THROWS(test.at(MatrixS::SizeType{ 3,3 }));
    }
    SUBCASE("out of range index > size") {
        CHECK_THROWS(test.at(4, 0));
        CHECK_THROWS(test.at(0, 4));
        CHECK_THROWS(test.at(4, 4));
    }
    SUBCASE("out of range index > size (but index is SizeType)") {
        CHECK_THROWS(test.at(MatrixS::SizeType{ 4,0 }));
        CHECK_THROWS(test.at(MatrixS::SizeType{ 0,4 }));
        CHECK_THROWS(test.at(MatrixS::SizeType{ 4,4 }));
    }
    SUBCASE("out of range index < 0") {
        CHECK_THROWS(test.at(-1, 0));
        CHECK_THROWS(test.at(0, -1));
        CHECK_THROWS(test.at(-1, -1));
    }
    SUBCASE("out of range index < 0 (but index is SizeType)") {
        CHECK_THROWS(test.at(MatrixS::SizeType{ -1,1 }));
        CHECK_THROWS(test.at(MatrixS::SizeType{ 1,-1 }));
        CHECK_THROWS(test.at(MatrixS::SizeType{ -1,-1 }));
    }
}

TEST_CASE("Return shape funcions") {
    MatrixS test(3, 4);
    SUBCASE("test ssize") {
        CHECK((test.ssize() == MatrixS::SizeType{ 3,4 }));
    }
    SUBCASE("test nRows") {
        CHECK((test.nRows() == 3));
    }
    SUBCASE("test nCols") {
        CHECK((test.nCols() == 4));
    }
}

TEST_CASE("Test resize fuction") {
    SUBCASE("resize from n,m with relevant terms") {
        MatrixS test(2, 1);
        CHECK_NOTHROW(test.resize(1, 3));
        CHECK((test.ssize() == MatrixS::SizeType{ 1,3 }));
    }
    SUBCASE("resize from n,m with nonrelevant terms of rows") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(-1, 3));
    }
    SUBCASE("resize from n,m with nonrelevant terms of cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(1, -3));
    }
    SUBCASE("resize from n,m with nonrelevant terms of rows and cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(-1, -3));
    }
    SUBCASE("resize from n,m with zero terms of rows") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(0, 3));
    }
    SUBCASE("resize from n,m with zero terms of cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(1, 0));
    }
    SUBCASE("resize from n,m with zero terms of rows and cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(0, 0));
    }
    SUBCASE("resize from SizeType with relevant terms") {
        MatrixS test(2, 1);
        CHECK_NOTHROW(test.resize(MatrixS::SizeType{ 1,3 }));
        CHECK((test.ssize() == MatrixS::SizeType{ 1,3 }));
    }
    SUBCASE("resize from SizeType with nonrelevant terms of rows") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(MatrixS::SizeType{ -1,3 }));
    }
    SUBCASE("resize from SizeType with nonrelevant terms of cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(MatrixS::SizeType{ 1,-3 }));
    }
    SUBCASE("resize from SizeType with nonrelevant terms of rows and cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(MatrixS::SizeType{ -1,-3 }));
    }
    SUBCASE("resize from SizeType with zero terms of rows") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(MatrixS::SizeType{ 0,3 }));
    }
    SUBCASE("resize from SizeType with zero terms of cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(MatrixS::SizeType{ 1,0 }));
    }
    SUBCASE("resize from SizeType with zero terms of rows and cols") {
        MatrixS test(2, 1);
        CHECK_THROWS(test.resize(MatrixS::SizeType{ 0,0 }));
    }
}