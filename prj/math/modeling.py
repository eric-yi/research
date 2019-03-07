#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def num_size(x):
    x = abs(x)
    c = 0
    while x >= 10 ** c:
        c += 1
    return c


class Analog(object):
    def __init__(self):
        pass

    @staticmethod
    def midsquare_rand(seed, count):
        mid = num_size(seed)
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


rands = Analog.midsquare_rand(2041, 1000)
logger.debug(rands)
