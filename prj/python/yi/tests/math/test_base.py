#!/usr/bin/evn python
# -*- coding:utf-8 -*-
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


class TestBase(object):
    def setup_class(self):
        self.logger = logging.getLogger(__name__)

    def teardown_class(self):
        pass