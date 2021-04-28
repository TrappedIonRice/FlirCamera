import numpy as np
import time
from scipy import optimize


class Parameter:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def __call__(self):
        return self.value


def fit(function, parameters, y, x=None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p)


def fitgauss1d(xx, yy, truncate=True):
    '''
    fit 1d gaussian function using scipy.optimize.leastsq
    :param xx:
    :param yy:
    :param truncate:
    :return:
    '''
    if truncate:
        x0, xx, yy = truncate_center(xx, yy)
    else:
        x0 = np.sum(xx * yy) / np.sum(yy)
    mu = Parameter(x0)
    background = Parameter(min(yy))
    height = Parameter(max(yy) - background())
    prep_sigma = xx[yy > (height() * np.exp(-1 / 2)) + background()]
    sigma = Parameter(abs(prep_sigma[-1] - prep_sigma[0]) / 2)

    def f(x):
        return height() * np.exp(-((x - mu()) / sigma()) ** 2 / 2) + background()

    return fit(f, [mu, sigma, height, background], yy, x=xx)


def fitgauss1d_moment(xx, yy, truncate=True):
    '''
    Fit 1d gaussian function using moment.
    :param xx:
    :param yy:
    :param truncate:
    :return: x0, sigma
    '''
    if truncate:
        x0, xx, yy = truncate_center(xx, yy)
    else:
        x0 = np.sum(xx * yy) / np.sum(yy)
    cm0 = np.sum(yy)
    cm2 = np.sum((xx - x0) ** 2 * yy)
    cm4 = np.sum((xx - x0) ** 4 * yy)
    s0 = xx.size
    s2 = np.sum((xx - x0) ** 2)
    s4 = np.sum((xx - x0) ** 4)
    if (cm4 * s0 - cm0 * s4) ** 2 - 4 * 3 * (cm2 * s0 - cm0 * s2) * (cm4 * s2 - cm2 * s4) > 0:
        sigma = (np.sqrt(np.abs((cm4 * s0 - cm0 * s4 + np.sqrt(
            (cm4 * s0 - cm0 * s4) ** 2 - 4 * 3 * (cm2 * s0 - cm0 * s2) * (cm4 * s2 - cm2 * s4))) / (
                                        6 * (cm2 * s0 - cm0 * s2)))))
    else:
        sigma = np.sqrt(np.abs(np.sum((xx - x0) ** 2 * yy) / np.sum(yy)))
    return x0, sigma


def truncate_center(xx, yy):
    '''
    Truncate the data to reduce the effect of the noise.
    The final x0 will be at the center of xx1.
    This method is only valid when xx is ascending equal-spacing array
    and yy is a symmetric function with ignorable tail at the edge of xx.
    :param xx:
    :param yy:
    :return: x0, xx, yy
    '''
    x0 = np.sum(xx * yy) / np.sum(yy)
    xx0 = []
    while not np.array_equal(xx0, xx):
        xx0 = xx
        if x0 - xx0[0] < xx0[-1] - x0:
            xx = xx0[:xx0[xx0 < x0 * 2 - xx0[0]].size + 1]
            yy = yy[:xx0[xx0 < x0 * 2 - xx0[0]].size + 1]
        elif x0 - xx0[0] > xx0[-1] - x0:
            xx = xx0[-xx0[xx0 > x0 * 2 - xx0[-1]].size - 1:]
            yy = yy[-xx0[xx0 > x0 * 2 - xx0[-1]].size - 1:]
        else:
            xx = xx0
        x0 = np.sum(xx * yy) / np.sum(yy)

    return x0, xx, yy


def gauss1d(mu, sigma, m, x):
    return m * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fitgauss2d_section(xx, yy, zz):
    '''
    fit the x and y sections of a 2d gaussian function

    :param xx: 1d array
        x coordinate, correspond to the second index of zz, must be ascending equal-spacing.
    :param yy: 1d array
        y coordinate, correspond to the first index of zz, must be ascending equal-spacing.
    :param zz: numpy.ndarray 2d
        data to be fitted
    :return:
    p: 1d array
        mu_x, mu_y, sigma_x, sigma_y, height
    ier: int
        An integer flag. It equals to 1 if everything was fine.
    '''
    fit_int_x = fitgauss1d(xx, zz.sum(axis=0))
    fit_int_y = fitgauss1d(yy, zz.sum(axis=1))
    # print(fit_int_x)
    # print(fit_int_y)
    # fit_x = fitgauss1d(xx, zz[np.min(np.abs(yy - fit_int_y[0][0])) == np.abs(yy - fit_int_y[0][0]), ::].flatten())
    # fit_y = fitgauss1d(yy, zz[::, np.min(np.abs(xx - fit_int_x[0][0])) == np.abs(xx - fit_int_x[0][0])].flatten())
    fit_x = fitgauss1d(xx, zz[round(fit_int_y[0][0]), ::])
    fit_y = fitgauss1d(yy, zz[::, round(fit_int_x[0][0])])

    p = (fit_x[0][0], fit_y[0][0], fit_x[0][1], fit_y[0][1], (fit_x[0][2] + fit_y[0][2]) / 2, (fit_x[0][3] + fit_y[0][3]) / 2, )

    if all([fit_int_x[1] == 1, fit_int_y[1] == 1, fit_x[1] == 1, fit_y[1] == 1]):
        ier = 1
    else:
        ier = 0
    return p, ier


def gauss2d(mu_x, mu_y, sigma_x, sigma_y, height, theta, x, y):
    return height * np.exp(-((x - mu_x) * np.cos(theta) + (y - mu_y) * np.sin(theta)) ** 2 / (2 * sigma_x ** 2) -
                           (-(x - mu_x) * np.sin(theta) + (y - mu_y) * np.cos(theta)) ** 2 / (2 * sigma_y ** 2))


def fitguase2d_int():
    return 0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx,yy = np.meshgrid(np.arange(4000),np.arange(3000))
    zz = (gauss2d(1000, 1000, 100, 400, 1, 0, xx, yy) + 1 + np.random.rand(*(xx.shape)) * 0.5)*100
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()
    start_time = time.time()
    for i in range(1):
        fitgauss2d_section(np.arange(0, 4000), np.arange(0, 3000), zz)
    print("--- %.8f seconds ---" % (time.time() - start_time))
    print(fitgauss2d_section(np.arange(0, 4000), np.arange(0, 3000), zz))

    # xx = np.arange(0, 4000, 1) * 0.57
    # yy = gauss1d(1000, 200, 1, xx) + 1 + np.random.rand(xx.size) * 0.5
    # plt.plot(xx, yy)
    # plt.show()
    # x0, sigma = fitgauss1d_moment(xx, yy)
    # print(x0, sigma)
    # result = fitgauss1d(xx, yy)
    # print(result)

    # start_time = time.time()
    # for i in range(1000):
    #     x0, sigma = fitgauss1d_moment(xx, yy)
    # print("--- %.8f seconds ---" % (time.time() - start_time))
    # print(x0,sigma)
    # start_time = time.time()
    # for i in range(1000):
    #     result = fitgauss1d(xx, yy)
    # print("--- %.8f seconds ---" % (time.time() - start_time))
    # print(result)
