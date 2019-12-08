#!/usr/bin/evn python
# -*- coding:utf-8 -*-
from .test_base import TestBase
from yi.mathmatics.modeling import Random, LeastSquare


class TestModeling(TestBase):
    def test_should_run_midsqure_random(self):
        rands = Random.midsquare(2041, 1000)
        self.logger.debug(rands)
        assert len(rands) == 1001

    def test_should_run_least_square(self):
        a = [[1, 1], [1, -1], [1, 1]]
        b = [2, 1, 3]
        x = LeastSquare.run(a, b)
        self.logger.debug(x)
        assert len(x) == 2

    def test_should_run_gauss_elimination(self):
        A = [[3.0, 1.0], [1.0, 3.0]]
        b = [6.0, 4.0]
        x = LeastSquare.gauss_elimination(A, b)
        self.logger.debug(x)
        assert len(x) == len(b)
