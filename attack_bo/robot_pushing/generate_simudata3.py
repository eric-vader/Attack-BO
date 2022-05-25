#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
from .push_world import *
import sys

# Returns the distance to goal of an object pushed by a pusher.
# The goal is to minimize f.
# You can define the function to be maximized as following
# gpos = 10 .* rand(1, 2) - 5;
# f = @(x) 5 - robot_pushing_3(x(1:2), x(3), gpos);
# tuning range: 
# xmin = [-5; -5; 1];
# xmax = [5; 5; 30];    
def robot_push_3d(rx, ry, steps, gx, gy):
    simu_steps = int(float(steps) * 10)
    
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
    thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))

    # TODO fixed div by 0 error
    if rx == 0:
        rx = 0.000000000000000000001
    if ry == 0:
        ry = 0.000000000000000000001
    init_angle = np.arctan(ry/rx)
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
    ret = simu_push(world, thing, robot, base, simu_steps)
    ret = np.linalg.norm(np.array([gx, gy]) - ret)
    return ret

if __name__ == '__main__':
    rx = float(sys.argv[1])
    ry = float(sys.argv[2])
    gx = float(sys.argv[4])
    gy = float(sys.argv[5])
    ret = robot_push_3d(rx, ry, sys.argv[3], gx, gy)
    sys.stdout.write(str(ret))

