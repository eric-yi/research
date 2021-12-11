#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Random(object):
    @staticmethod
    def num_size(x):
        x = abs(x)
        c = 0
        while x >= 10 ** c:
            c += 1
        return c

    @staticmethod
    def midsquare(seed, count):
        mid = Random.num_size(seed)
        logger.debug(mid)
        size = 2 * mid
        ps = int(mid - mid / 2)
        pe = int(mid + mid / 2)
        rand = seed
        rands = [rand]
        for n in range(0, count):
            rand_str = str(rand ** 2)
            rand_str = (size - len(rand_str)) * '0' + rand_str
            rand = int(rand_str[ps:pe])
            rands.append(rand)
        return rands


class LeastSquare(object):
    @staticmethod
    def gauss_elimination(A, B):
        rows, cols = len(A), len(A[0])
        for m in range(0, cols):
            if A[m][m] == 0:
                logger.error('error, gauss failed, no solution')
                return
            for n in range(0, rows):
                if m != n:
                    factor = A[n][m] / A[m][m]
                    for n_m in range(0, cols):
                        A[n][n_m] -= factor * A[m][n_m]
        x = []
        for i in range(0, len(B)):
            x.append(B[i] / A[i][i])
        return x

    @staticmethod
    def run(a, b):
        a = np.array(a)
        a_t = np.transpose(a)
        A = np.dot(a_t, a)
        b = np.array(b)
        B = np.dot(a_t, b)
        return LeastSquare.gauss_elimination(A, B)
