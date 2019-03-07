#!/usr/bin/evn python
# -*- coding:utf-8 -*-

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Analog(object):
    def __init__(self):
        pass

    @staticmethod
    def midsquare_rand(seed, count):
        mid = len(str(seed))
        size = 2 * mid
        ps = int(mid - mid / 2)
        pe = int(mid + mid / 2)
        rand = seed
        rands = [rand]
        for n in range(0, count):
            rand_str = str(rand * rand)
            rand_str = (size - len(rand_str)) * '0' + rand_str
            rand = int(rand_str[ps:pe])
            rands.append(rand)
        return rands


rands = Analog.midsquare_rand(204178991, 1000)
logger.debug(rands)
