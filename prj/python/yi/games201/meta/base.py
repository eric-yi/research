#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import cv2 as cv
from datetime import datetime
from types import DynamicClassAttribute


class TimeUtil:
    @staticmethod
    def now():
        return datetime.now()

    @staticmethod
    def millis(start_time):
        dt = datetime.now() - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * \
            1000 + dt.microseconds / 1000.0
        return ms

    @staticmethod
    def seconds(start_time):
        return TimeUtil.millis(start_time) / 1000.0


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @DynamicClassAttribute
    def position(self):
        return (int(self.x * 100), int(self.y * 100))

    def __add__(self, v):
        if isinstance(v, Vec2):
            return Vec2(self.x + v.x, self.y + v.y)
        return Vec2(self.x + v, self.y + v)

    def __mul__(self, v):
        return Vec2(self.x * v, self.y * v)

    def __sub__(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self):
        return Vec2(self.x / self.norm(), self.y / self.norm())

class Window:
    class FrameText:
        def __init__(self, frame, position):
            self.font = cv.FONT_HERSHEY_PLAIN
            self.font_scale = 0.8
            self.font_color = (255, 255, 255)
            self.fps = 0.0
            self.avg = 0.0

        def update(self, fps, avg):
            self.fps = fps
            self.avg = avg

        def write(self, frame , position):
            cv.putText(frame, '{:.2f} fps {:.2f} ms'.format(self.fps, self.avg), position, self.font,
                       self.font_scale, self.font_color)

    def __init__(self, width, height, title='window'):
        self.width = width
        self.height = height
        self.title = title
        self.reset()
        self.frame_text = Window.FrameText(self.frame, (300, 20))

    def reset(self):
        self.frame = np.zeros((400, 400, 3), np.uint8) * 255

    def restart(self):
        self.start_time = TimeUtil.now()
        self.fps = 0.0
        self.avg = 0.0

    def show(self, update):
        self.restart()
        last_time = TimeUtil.now()
        while True:
            print('update...')
            self.reset()
            if TimeUtil.millis(self.start_time) >= 1000:
                self.frame_text.update(self.fps, 1000.0 / self.fps)
                self.restart()
            self.frame_text.frame = self.frame
            self.frame_text.write(self.frame, (260, 20))
            update(self.frame, self.frame_text.avg / 1000.0)
            cv.imshow(self.title, self.frame)
            cv.moveWindow(self.title, 3000, 400)
            self.fps += 1
            cv.waitKey(1)
            last_time = TimeUtil.now()
            

