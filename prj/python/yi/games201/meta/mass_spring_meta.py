#!/usr/bin/env python
# -*- coding:utf-8 -*-

from base import *
import cv2 as cv
import math

class Spring:
    class Linker:
        def __init__(self, knot, length=0.1):
            self.knot = knot
            self.length = length

    class Knot:
        def __init__(self, position, spring):
            self.position = position
            self.spring = spring
            self.mass = 1.0
            self.velocity = Vec2(0.0, 0.0)
            self.linkers = []

        def add_linker(self, knot):
            print('add linker...')
            self.linkers.append(Spring.Linker(knot))

        def update(self, dt):
            self.velocity *= math.exp(-dt * self.spring.damping)
            force = self.spring.gravity * self.mass

            for linker in self.linkers:
                d = self.position - linker.knot.position
                force += d.normalized() * (-self.spring.stiffness * (d.norm() - linker.length))

            self.velocity = force * dt * (1.0 / self.mass)

            if self.position.y >= (self.spring.window.height - 2):
                self.position.y = self.spring.window.height - 2
                self.velocity.y = 0

            self.position += self.velocity * dt

    def __init__(self):
        self.window = Window(400, 400)
        self.stiffness = 10000
        self.damping = 20
        self.gravity = Vec2(0.0, 9.8)
        self.knots  = []

    def add(self, position):
        new_knot = Spring.Knot(position, self)
        self.knots.append(new_knot)
        # if len(self.knots) > 1:
        #     print('add knots linkers...')
        #     last_knot = self.knots[len(self.knots) - 2]
        #     knot.add_linker(last_knot)
        #     last_knot.add_linker(knot)

    def link(self):
        i = 0
        while i < len(self.knots):
            k1 = self.knots[i]
            j = i + 1
            while j < len(self.knots):
                k2 = self.knots[j]
                k1.add_linker(k2)
                k2.add_linker(k1)
                j += 1
            i += 1
            

    def update(self, frame, dt):
        print('update frame...')
        for i, knot in enumerate(self.knots):
            knot.update(dt)
            cv.circle(frame, knot.position.position, 1, (0, 0, 255), 2)
            if i == 0 and len(self.knots) > 0:
                cv.line(frame, self.knots[len(self.knots) - 1].position.position, knot.position.position, (0, 255, 0))
            if i > 0:
                cv.line(frame, self.knots[i-1].position.position, knot.position.position, (0, 255, 0))

    def run(self):
        self.link()
        self.window.show(self.update)


spring = Spring()
spring.add(Vec2(0.3, 0.3))
spring.add(Vec2(0.2, 0.4))
spring.add(Vec2(0.4, 0.4))
spring.run()

