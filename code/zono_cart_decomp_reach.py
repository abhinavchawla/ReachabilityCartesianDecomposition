import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import block_diag
from scipy.linalg import expm
import math

from zono_sol import Zonotope
import time


def linearMap(zono, phi):
    return Zonotope(zono.box, a_mat=zono.a_mat.dot(phi), b_vec=zono.b_vec * phi)


def minkowskiSum(zono1, zono2):
    if zono1 == None:
        return zono2
    return Zonotope(box=np.concatenate((zono1.box, zono2.box), axis=0),
                    a_mat=np.concatenate((zono1.a_mat, zono2.a_mat), axis=1),
                    b_vec=zono1.b_vec + zono2.b_vec)


def cartesianProduct(zono1, zono2):
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

    init_box = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    gens = [[0.2, 0.3, 0.6], [0.5, 0.5,0.3]]
    init_zono = Zonotope(init_box, np.array(gens))
    init_zono.b_vec = np.array([[2], [4]])
    init_zono.plot()



    x1 = init_zono.max([-1, 0])[0][0]
    x2 = init_zono.max([1, 0])[0][0]

    init_box_x = [[-1.0, 1.0]]
    gens_x = [[(x2 - x1) / 2]]
    init_zono_x = Zonotope(init_box_x, np.array(gens_x))
    init_zono_x.b_vec = np.array([[(x2 + x1) / 2]])
    # init_zono_x.plot(xdim=0, ydim=-1)

    y1 = init_zono.max([0, -1])[1][0]
    y2 = init_zono.max([0, 1])[1][0]

    gens_y = [[(y2 - y1) / 2]]

    init_zono_y = Zonotope(init_box_x, np.array(gens_y))
    init_zono_y.b_vec = np.array([[(y2 + y1) / 2]])
    # init_zono_y.plot(xdim=-1, ydim=0)

    new_a_mat = block_diag(init_zono_x.a_mat, init_zono_y.a_mat)
    new_b_vec = np.concatenate((init_zono_x.b_vec, init_zono_y.b_vec), axis=0)
    cartesian_prod_zono = Zonotope(init_box[0:2], a_mat=new_a_mat, b_vec=new_b_vec)
    cartesian_prod_zono.plot(color="b:o")

    # init_zono.print()
    init_zono_x.print()
    init_zono_y.print()
    # cartesian_prod_zono.print()

    dynamics_mat = np.array([[-0.3, 1.6], [-1.2, 0.8]], dtype=float)  # mode 1: x' = y, y' = -x
    zonos = [init_zono_x, init_zono_y]
    b = len(zonos)
    time_step = math.pi / 8
    num_steps = 1
    sol_mat = expm(dynamics_mat * time_step)
    Q = sol_mat

    start = time.time()

    for step in range(num_steps):
        tmp_zonos = []
        for i in range(b):
            tmp_x = None
            for j in range(b):
                tmp_x = minkowskiSum(tmp_x, linearMap(zonos[j], Q[i][j]))
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
