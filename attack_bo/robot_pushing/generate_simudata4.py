#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
from .push_world import *
import sys

# Returns the distance to goal of an object pushed by a pusher.
# The goal is to minimize f.
# You can define the function to be maximized as following
# gpos = 10 .* rand(1, 2) - 5;
# f = @(x) 5 - robot_pushing_4(x(1:2), x(4), x(3), gpos);
# tuning range of x:
# xmin = [-5; -5; 1; 0];
# xmax = [5; 5; 30; 2*pi];

def robot_push_4d(rx, ry, steps, gx, gy, init_angle):
    simu_steps = int(float(steps) * 10)
    # Set the parameter to True if need gui
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
    thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))

    # TODO fixed div by 0 error
    if rx == 0:
        rx = 0.000000000000000000001
    if ry == 0:
        ry = 0.000000000000000000001

    xvel = -rx;
    yvel = -ry;
    regu = np.linalg.norm([xvel,yvel])
    xvel = xvel / regu * 10;
    yvel = yvel / regu * 10;
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
    ret = simu_push2(world, thing, robot, base, xvel, yvel, simu_steps)
    ret = np.linalg.norm(np.array([gx, gy]) - ret)
    return ret

# difference to generate_simudata is an input that control angle of push
if __name__ == '__main__':
    rx = float(sys.argv[1])
    ry = float(sys.argv[2])
    gx = float(sys.argv[4])
    gy = float(sys.argv[5])
    init_angle = float(sys.argv[6])
    ret = robot_push_4d(rx, ry, sys.argv[3], gx, gy, init_angle)
    sys.stdout.write(str(ret))