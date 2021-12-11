#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import math

image = np.ones((600, 600, 3), np.uint8) * 255

# pt1 = (150, 100)
# pt2 = (100, 200)
# pt3 = (200, 200)

# cv2.circle(image, pt1, 2, (0, 0, 255), -1)
# cv2.circle(image, pt2, 2, (0, 0, 255), -1)
# cv2.circle(image, pt3, 2, (0, 0, 255), -1)

# triangle_cnt = np.array([pt1, pt2, pt3])

# cv2.drawContours(image, [triangle_cnt], 0, (0, 255, 0), -1)

# cv2.imshow("image", image)
# cv2.waitKey()


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def toArr(self):
        return (int(self.x), int(self.y))

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def sub(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def norm(self):
        return math.sqrt(self.x*self.x + self.y * self.y)

    def normalize(self):
        return Vec2(self.x / self.norm(), self.y / self.norm())

    def mul(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)


class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.v = Vec2(0.0, 0.0)


triangles = []


def create(a_point, edge_length):
    x_det = int(edge_length * math.cos(math.pi / 3))
    y_det = int(edge_length * math.cos(math.pi / 6))
    b_point = Vec2(a_point.x - x_det, a_point.y - y_det)
    c_point = Vec2(a_point.x + x_det, a_point.y - y_det)
    return Triangle(a_point, b_point, c_point)


def draw(triangle):
    # cv2.circle(image, (triangle.a.x, triangle.a.y), 2, (0, 0, 255), -1)
    # cv2.circle(image, (triangle.b.x, triangle.b.y), 2, (0, 0, 255), -1)
    # cv2.circle(image, (triangle.c.x, triangle.c.y), 2, (0, 0, 255), -1)
    cv2.line(image, triangle.a.toArr(), triangle.b.toArr(), (0, 0, 0), 1)
    cv2.line(image, triangle.b.toArr(), triangle.c.toArr(), (0, 0, 0), 1)
    cv2.line(image, triangle.c.toArr(), triangle.a.toArr(), (0, 0, 0), 1)


# triangles.append(Triangle(Vec2(150, 100), Vec2(100, 200), Vec2(200, 200)))


for r in range(5):
    a_point = Vec2(50 + r*20, 100 + r*20)
    triangle = create(a_point, 20)
    triangles.append(triangle)

dt = 1e-3
damping = 20.0
particle_mass = 1
gravity = Vec2(0, -9.8)
spring_stiffness = 10000


def update():
    for t in triangles:
        t.v.x *= math.exp(-dt * damping)
        t.v.y *= math.exp(-dt * damping)
        force = Vec2(gravity.x * particle_mass, gravity.y * particle_mass)
        for o in triangles:
            if o is not t:
                det = t.a.sub(o.a)
                v_det = det.normalize().mul(-spring_stiffness * (det.norm() - 0.1))
                force.x += v_det.x
                force.y += v_det.y
        t.v.x += dt * force.x / particle_mass
        t.v.y += dt * force.y / particle_mass

    for t in triangles:
        t.a.x += t.v.x * dt
        t.a.y += t.v.y * dt
        x_det = int(20 * math.cos(math.pi / 3))
        y_det = int(20 * math.cos(math.pi / 6))
        t.b = Vec2(a_point.x - x_det, a_point.y - y_det)
        t.c = Vec2(a_point.x + x_det, a_point.y - y_det)


while True:
    image = np.ones((600, 600, 3), np.uint8) * 255
    update()
    for triangle in triangles:
        draw(triangle)
    cv2.imshow("image", image)
    if cv2.waitKey(25) & 0xFF == 27:
        break
