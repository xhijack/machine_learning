# -*- coding: utf-8 -*-
#
# Created by: Bunyamin, Ramdani, Selly, Redy
# Advisor: Arief Fatchul Huda
# Created Date :
# following Book of Lawrence "Fundamental Neural Network"
#

from numpy import array, dot, random

training_data = [
    (array([1,1,1]), 1),
    (array([1,1,0]), -1),
    (array([1,0,1]), -1),
    (array([0,1,1]), -1),
]


def f_net(n):
    if n > eta:
        return 1
    elif -eta <= n <= eta:
        return 0
    elif n < -eta:
        return -1


eta, b, stop, i = 0.2, 0, False, 1
w = [0, 0, 0]

while not stop:
    stop = True
    for x, expected in training_data:
        net = b + dot(w, x)
        out = f_net(net)
        if out != expected:
            w += x * expected
            b += expected
            stop = False

        print("Counter:", i,  x, net, out, expected, x * expected, w)
    i += 1



