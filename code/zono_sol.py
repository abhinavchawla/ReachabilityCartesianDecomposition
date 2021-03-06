'''
Zonotope reach
'''

import math
from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import expm


class Zonotope:
    'zonotope class'

    def __init__(self, box, a_mat=None, b_vec=None):

        self.box = np.array(box, dtype=float)
        self.a_mat = a_mat if a_mat is not None else np.identity(self.box.shape[0])

        self.dims = self.a_mat.shape[0]
        self.gens = self.a_mat.shape[1]

        self.b_vec = b_vec if b_vec is not None else np.zeros((self.dims, 1))

    def print(self):
        print("Zonotope Parameters")
        print("Dimensions", self.dims)
        print("Generators", self.gens)
        print("A_Mat", self.a_mat.shape, self.a_mat)
        print("B_Vec", self.b_vec.shape, self.b_vec)
        print("BOX", self.box.shape, self.box)

    def verts(self, xdim=0, ydim=1):
        'get verticies for plotting 2d projections'

        verts = []

        for angle in np.linspace(0, 2 * math.pi, 32):

            direction = np.zeros((self.dims,))
            if xdim!=-1:
                direction[xdim] = math.cos(angle)
            if ydim!=-1:
                direction[ydim] = math.sin(angle)

            pt = self.max(direction)
            if ydim!=-1:
                y_pt = pt[ydim][0]
            else:
                y_pt = 0

            if xdim!=-1:
                x_pt = pt[xdim][0]
            else:
                x_pt = 0

            xy_pt = (x_pt,y_pt)
            if verts and np.allclose(xy_pt, verts[-1]):
                continue

            verts.append(xy_pt)
        # print("Vertices: ",verts)
        return verts

    def plot(self, color='k-o', xdim=0, ydim=1, zorder=1):
        'plot 2d projections'
        if xdim ==-1:
            self.plotIn1D(ydim, "y", color, zorder)
            return
        if ydim ==-1:
            self.plotIn1D(xdim, "x", color, zorder)
            return
        v_list = self.verts(xdim=xdim, ydim=ydim)
        xs = [v[xdim] for v in v_list]
        xs.append(v_list[0][xdim])
        ys = [v[ydim] for v in v_list]
        ys.append(v_list[0][ydim])
        plt.plot(xs, ys, color, zorder=zorder)

    def plotIn1D(self, dim, axis, color, zorder):
        if axis == "x":
            v_list = self.verts(xdim=dim, ydim=-1)
            xs = [v[0] for v in v_list]
            xs.append(v_list[0][0])
            ys = [0 for v in v_list]
            ys.append(0)
            plt.plot(xs, ys, color, zorder=zorder)
        if axis == "y":
            v_list = self.verts(xdim=-1, ydim=dim)
            xs = [0 for v in v_list]
            xs.append(0)
            ys = [v[1] for v in v_list]
            ys.append(v_list[0][1])
            plt.plot(xs, ys, color, zorder=zorder)

    def max(self, direction):
        '''returns the point in the box that is the maximum in the passed in direction

        if x is the point and c is the direction, this should be the maximum dot of x and c
        '''

        direction = self.a_mat.transpose().dot(direction)
        # print(direction)

        # box has two columns and n rows
        # direction is a vector (one column and n rows)

        # returns a point (one column with n rows)

        box = self.box
        rv = []

        for dim, (lb, ub) in enumerate(box):
            if direction[dim] > 0:
                rv.append([ub])
            else:
                rv.append([lb])

        pt = np.array(rv)
        # print(pt)

        return self.a_mat.dot(pt) + self.b_vec


def init_plot():
    'initialize plotting style'

    try:
        matplotlib.use('TkAgg')  # set backend
    except:
        pass

    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

    plt.axis('equal')


def is_inside_invariant(zono, mode, boundary):
    'is the zonotope inside the invariant for the mode?'
    init_plot
    rv = True

    if mode == 1:
        min_x_pt = zono.max([-1, 0])
        min_x_val = min_x_pt[0]

        tol = 1e-6

        if min_x_val + tol >= boundary:  # left the invariant
            rv = False

    return rv


def main():
    'main entry point'

    # init_box = [[-1.0, 1.0], [-1, 1.0],[-1.0,1.0]]
    # gens = [[1, 0, 0], [0, 0.5, 0.5]]
    # init_zono = Zonotope(init_box, np.array(gens))
    # init_zono.plot()
    #
    # init_box = [[-1.0, 1.0], [-1, 1.0]]
    # gens = [[1,0], [0, 1]]
    # init_zono = Zonotope(init_box, np.array(gens))
    # init_zono.plot()

    init_box = [[-1.0, 1.0], [-1.0, 1.0]]

    dynamics_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)  # mode 1: x' = y, y' = -x
    dx_mode_2 = 2.0  # mode 2: x' = 2, y' = 0
    time_step = 0.2
    num_steps = 3
    mode_boundary = 0

    sol_mat = expm(dynamics_mat * time_step)

    # init_box.append([-0.2, 0.2])
    # init_a_mat = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    init_zono = Zonotope(init_box)
    init_zono.a_mat = np.array([[0.5, 0], [0, 0.5]], dtype=float)
    init_zono.b_vec = np.array([[-4.5], [0.5]])

    waiting_list = []  # 3-tuples: zonotope, mode, step number
    waiting_list.append((init_zono, 1, 0))

    while waiting_list:
        z, mode, step = waiting_list.pop()

        while step < num_steps and is_inside_invariant(z, mode, mode_boundary):
            # plot

            if mode == 1:
                z.plot('r-')
            else:
                z.plot('b:')

            step += 1
            print("Time: ", step)
            print("A_mat: ", z.a_mat)
            print("B_vec: ", z.b_vec)

            # advance
            if mode == 1:
                z.a_mat = sol_mat.dot(z.a_mat)
                z.b_vec = sol_mat.dot(z.b_vec)
            else:
                z.b_vec += np.array([[dx_mode_2], [0]])

            # check if inside guard
            if mode == 1:
                max_x_pt = z.max([1, 0])
                max_x_val = max_x_pt[0]

                if max_x_val >= mode_boundary:  # guard is true
                    waiting_list.append((deepcopy(z), 2, step))

    init_box = [[-1.0, 1.0], [-1.0, 1.0]]

    dynamics_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)  # mode 1: x' = y, y' = -x
    dx_mode_2 = 2.0  # mode 2: x' = 2, y' = 0
    time_step = 0.002
    num_steps = 600
    mode_boundary = 0

    sol_mat = expm(dynamics_mat * time_step)

    plt.plot([mode_boundary, mode_boundary], [-2, 6], 'k--')
    init_zono = Zonotope(init_box)
    init_zono.a_mat = np.array([[0.3, 0], [0, 0.3]], dtype=float)
    init_zono.b_vec = np.array([[-4.5], [0.5]])

    waiting_list = []  # 3-tuples: zonotope, mode, step number
    waiting_list.append((init_zono, 1, 0))

    while waiting_list:
        z, mode, step = waiting_list.pop()

        while step < num_steps and is_inside_invariant(z, mode, mode_boundary):
            # plot

            if mode == 1:
                z.plot('r-')
            else:
                z.plot('b:')

            step += 1
            print("Time: ", step)
            print("A_mat: ", z.a_mat)
            print("B_vec: ", z.b_vec)

            # advance
            if mode == 1:
                z.a_mat = sol_mat.dot(z.a_mat)
                z.b_vec = sol_mat.dot(z.b_vec)
            else:
                z.b_vec += np.array([[dx_mode_2], [0]])

            # check if inside guard
            if mode == 1:
                max_x_pt = z.max([1, 0])
                max_x_val = max_x_pt[0]

                if max_x_val >= mode_boundary:  # guard is true
                    waiting_list.append((deepcopy(z), 2, step))

    plt.show()
    # plt.savefig('zono.png')


if __name__ == '__main__':
    ()
    main()
