import numpy as np
import matplotlib.pyplot as plt
import time


def find_circle_pixels(r):
    """
    Bresenham algorithm.
    find ij indexes (x,y) on circle with radius r.
    assuming centre is at (0,0), return only the first octant.
    notice the coordinates are chosen as consistent with matrix indexing.
    input:
        r: integer, radius
    return:
        circle_pixels: 2D array, (num_pixel, 2)
    """
    x = 0
    y = r
    d = 3-2*r

    circle_pixels = []
    while y >= x:
        # 1st octants
        circle_pixels.append((x,y))

        if d <= 0:
            d += 4*x + 6
        else:
            d += 10 + 4*(x - y)
            y -= 1
        # x is always increased by 1
        x += 1 

    return np.array(circle_pixels)

def xy_symmetry(points):
    """
    find xy symmetry of points.
    notice that points2 and point can have one common point aw x=y.
    input:
        points: 2D array, (num_points, 2)
    return:
        points2: 2D array, (num_points, 2). xy symmetry of points
    """
    n = points.shape[0]
    points = np.array(points)
    points2 = points[:,[1,0]]

    if np.all(points[n-1] == points2[n-1]):
        points2 = np.delete(points2, n-1, axis = 0)

    return points2

def fill_4_quadrants(points):
    """
    given points in one quadrants, find all other points by sign-alternating symmetry.
    input:
        points: 2D array, (num_points, 2), points in one quadrant
    return:
        points: points in all 4 quadrant without repeatedness.
    """
    if len(points) == 0:
        return points
    quadrants = [points]
    for sign in [np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])]:
        quadrants.append(points * sign)
    return np.concatenate(quadrants)


def find_shell_pixels(rmin, rmax):
    """
    find all pixels in the shell defined by (rmin, rmax).
    assuming centre is at (0,0).

    """
    if rmin >= rmax:
        print("rmin must be smaller than rmax")
        return

    # 1st octant of 2 circles
    cin0 = find_circle_pixels(rmin)
    cout0 = find_circle_pixels(rmax)

    # 2nd octant of 2 circles
    cin1 = xy_symmetry(cin0)
    cout1 = xy_symmetry(cout0)

    # concatenate the 2 octants to form pixels in first quadrant. 
    # index start from 1 to remove pixels on x,y axis.
    cin = np.concatenate((cin0[1:], cin1[1:]))
    cout = np.concatenate((cout0[1:], cout1[1:]))

    # find other pixels in other 3 quadrants.
    cin = fill_4_quadrants(cin)
    cout = fill_4_quadrants(cout)

    # add pixels on axes
    f = lambda r: np.array([[0,r],[r,0],[0,-r],[-r,0]])
    cin_on_axis = f(rmin)
    cout_on_axis = f(rmax)
    cin = np.concatenate((cin, cin_on_axis))
    cout = np.concatenate((cout, cout_on_axis))

    # 4-part filling
    # For the first 2 parts, i and j start from 1 to remove pixels on x,y axis. 
    # This is to avoid repeatedness when applying fill_4_quadrants.
    fill = [(i, j) for i in range(1, len(cin0)) for j in range(cin0[i, 1] + 1, cout0[i, 1])]
    fill += [(i, j) for j in range(1, len(cin1)) for i in range(cin1[j, 0] + 1, cout1[j, 0])]
    fill += [(i, j) for i in range(cin0[-1][0] + 1, cout0[-1][0] + 1) for j in range(i, cout0[i, 1])]
    fill += [(i, j) for j in range(cin1[-1][1] + 1, cout1[-1][1] + 1) for i in range(j+1, cout1[j, 0])]
    if len(fill) == 0:
        return cin, cout

    fill = np.array(fill)
    fill = fill_4_quadrants(fill)
    if rmax - rmin > 1:
        fill_y_right = np.array([[0,y] for y in range(rmin + 1, rmax)])
        fill_y_left = fill_y_right * np.array([1, -1])
        fill_x_down = np.array([[x,0] for x in range(rmin + 1, rmax)])
        fill_x_up = fill_x_down * np.array([-1, 1])
        fill = np.concatenate((fill, fill_y_right, fill_y_left, fill_x_down, fill_x_up))

    return cin, cout, fill



if __name__ == "__main__":

    # arr = np.zeros((128,128))
    # c0 = arr.shape[0]//2
    # c1 = arr.shape[1]//2

    # circle_pixels = find_circle_pixels(30)
    # circle_pixels2 = xy_symmetry(circle_pixels)
    # pixels = np.concatenate((circle_pixels, circle_pixels2))

    # arr[c0+circle_pixels.T[0], c1+circle_pixels.T[1]] = 1
    # arr[c0+circle_pixels2.T[0], c1+circle_pixels2.T[1]] = 3
    # print(len(circle_pixels), len(circle_pixels2))


    # plt.imshow(arr, cmap = "jet")
    # plt.show()

    # arr = np.zeros((10,10))
    # x0 = arr.shape[0]//2
    # y0 = arr.shape[1]//2

    # objs = find_shell_pixels(2, 4)

    # fill_values = [30, 30, 10]
    # for i, o in enumerate(objs):
    #     print(o.shape)
    #     arr[x0 + o.T[0], y0 + o.T[1]] = fill_values[i]

    # plt.imshow(arr, cmap = "jet", vmin = 0, vmax = 30)
    # plt.show()


    # test for computational complexity
    rr = np.arange(1,1000)
    t = np.zeros(len(rr))
    for i, rmin in enumerate(rr):
        t0 = time.time()
        objs = np.array(find_shell_pixels(rmin, rmin+1))
        t[i] = time.time() - t0
        pixels = np.concatenate(objs, axis = 0)
        n1 = len(pixels)
        n2 = len(np.unique(pixels, axis = 0))
        if not n1 == n2:
            print("error!")
        print("rmin = {}, rmax = {}, {}".format(rmin, rmin+1, n1==n2))

    plt.plot(rr, t, "-")
    plt.show()










