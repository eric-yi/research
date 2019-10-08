#!/usr/bin/evn python
# -*- coding:utf-8 -*-
import yi.keras.practice as practice


class TestSimple:
    def test_do_lab_with_softmax(self):
        practice.do_lab_with_softmax()
        assert True

    def test_do_lab_with_mlp(self):
        practice.do_lab_with_mlp()
        assert True
