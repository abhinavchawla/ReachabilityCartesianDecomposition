import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import block_diag
from scipy.linalg import expm
import math

from zono_sol import Zonotope
import time


def linearMap(zono, phi):
    return Zonotope(zono.box, a_mat=phi.dot(zono.a_mat), b_vec=phi.dot(zono.b_vec))


def minkowskiSum(zono1, zono2):
    if zono1 == None:
        return zono2
    return Zonotope(box=np.concatenate((zono1.box, zono2.box), axis=0),
                    a_mat=np.concatenate((zono1.a_mat, zono2.a_mat), axis=1),
                    b_vec=zono1.b_vec + zono2.b_vec)


def cartesianProduct(zono1, zono2):
    if zono2 is None:
        return zono1
    if zono1 is None:
        return zono2
    new_a_mat = block_diag(zono1.a_mat, zono2.a_mat)
    new_b_vec = np.concatenate((zono1.b_vec, zono2.b_vec), axis=0)
    new_box = np.concatenate((zono1.box, zono2.box), axis=0)
    return Zonotope(new_box, a_mat=new_a_mat, b_vec=new_b_vec)


def init_plot():
    'initialize plotting style'

    try:
        matplotlib.use('TkAgg')  # set backend
    except:
        pass

    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

    plt.axis('equal')


def main():
    'main entry point'

    # init_box = [[-1.0, 1.0], [-1, 1.0], [-1.0, 1.0]]
    # gens = [[0.5, 0, 0], [0, 0.5, 0.5]]
    fp = open("zonotope.txt")
    d=6
    p=2
    gens= [[] for i in range(d)]
    init_box=[]
    b_vec=[]
    for line in (fp):
        float_list = [float(i) for i in line.split()]
        for i in range(d):
            gens[i].append(float_list[i])
        init_box.append([-1.0,1.0])
    for i in range(d):
        b_vec.append([1])
    print(len(gens[0]))
    print(len(b_vec))

    init_zono = Zonotope(init_box, np.array(gens))
    init_zono.b_vec = np.array(b_vec)
    init_zono.plot()

    zonos = []
    dims = len(gens)
    i=0
    while i < dims:
        tmp_zono=None
        j=0
        while j<p and i<dims:
            direction = [0] * dims
            direction[i] = -1
            x1 = init_zono.max(direction)[i][0]
            direction[i] = 1
            x2 = init_zono.max(direction)[i][0]
            init_box_x = [[-1.0, 1.0]]
            gens_x = [[(x2 - x1) / 2]]
            init_zono_x = Zonotope(init_box_x, np.array(gens_x))
            init_zono_x.b_vec = np.array([[(x2 + x1) / 2]])
            tmp_zono = cartesianProduct(tmp_zono, init_zono_x)
            i+=1
            j+=1
        zonos.append(tmp_zono)

    cartesianProduct(zonos[0], zonos[1]).plot(color="b:o")

    # init_zono.print()
    # init_zono_x.print()
    # init_zono_y.print()
    # cartesian_prod_zono.print()

    dynamics_mat = np.array([[-0.3, 1.6, -0.3, 1.6, -0.3, -0.3], [-1.2, 0.8, -1.2, 0.8, -1.2, -1.2], [-0.3, 1.6, -0.3, 1.6, -0.3, 1.6], [-1.2, 0.8, -1.2, 0.8, -1.2, -1.2], [-0.3, 1.6, -0.3, 1.6, -0.3, 1.6], [-1.2, 0.8, -1.2, 0.8, -1.2, -1.2]], dtype=float)  # mode 1: x' = y, y' = -x

    b = len(zonos)
    time_step = 1
    num_steps = 600
    sol_mat = expm(dynamics_mat * time_step)
    Q = sol_mat

    start = time.time()
    print(Q)

    for step in range(num_steps):
        tmp_zonos = []
        for i in range(b):
            tmp_x = None
            for j in range(b):
                tmp_x = minkowskiSum(tmp_x, linearMap(zonos[j], Q[i*p:(i*p)+p, j*p:(j*p)+p]))
            tmp_zonos.append(tmp_x)
        cartesianProduct(tmp_zonos[0], tmp_zonos[1]).plot(color="b:o")
        Q = Q.dot(sol_mat)
    end = time.time()

    print("TIME: ", end - start)

    start = time.time()
    for step in range(num_steps):
        init_zono.a_mat = sol_mat.dot(init_zono.a_mat)
        init_zono.b_vec = sol_mat.dot(init_zono.b_vec)
        init_zono.plot()
    end = time.time()
    print("TIME: ", end - start)
    plt.show()


if __name__ == '__main__':
    init_plot()
    main()
