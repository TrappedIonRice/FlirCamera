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

def pickpeak(x, npicks=2, rdiff=5):
    """
Search for peaks in data. Return arrays may contain NaN, e.g., if less peaks than required are found.
@param x: data sequence
@type x: ndarray
@param npicks: number of peaks to return.
@param rdiff: minimum spacing (in data points) between two peaks
@return: tuple of ndarray.
@rtype: (peak values, peak locations)
"""

    # initialize result values with NaN
    vals = np.array([np.NaN] * npicks)
    loc = np.array([0] * npicks)

    rmin = np.nanmin(x) - 1
    dx = np.diff(np.r_[rmin, x, rmin])

    # find position and their values of peaks (local maxima)
    pos_peaks, = np.nonzero((dx[0:-1] >= 0.0) & (dx[1:] <= 0.0))
    val_peaks = x[pos_peaks]  # corresponding peak value

    # select peaks in descending order, seperated by at least rdiff
    for i in range(npicks):

        mi = np.nanargmax(val_peaks)  # find index of largest peak

        peakval = val_peaks[mi]
        peakpos = pos_peaks[mi]

        vals[i] = peakval
        loc[i] = peakpos

        # for next iteration: only keep peaks at least rdiff points
        # distance from last peak
        ind = np.nonzero(abs(pos_peaks - peakpos) > rdiff)
        if len(ind) == 0:  # nothing left!
            break
        val_peaks = val_peaks[ind]
        pos_peaks = pos_peaks[ind]
        if np.isfinite(val_peaks).sum() == 0:
            break
    return vals, loc

def find_startpar_gauss(x, prof):
    """
    find good initial estimates for fit parameters based on
    horizontal or vertical profiles

    @param x: x or y values
    @param prof: horizontal of vertical profiles
    @return: [A, mu, sigma, offset]
    """

    Nsh = 20  # number (half) of points for smoothing
    gs = Nsh / 2  # width gaussian

    # use gaussian for smoothing
    gx = np.arange(2 * Nsh + 1) - Nsh
    gy = np.exp(-gx ** 2 / gs ** 2)
    gy /= gy.sum()

    # smooth profil, limit axes to valid values
    profsmooth = np.convolve(prof, gy, mode='valid')
    xs = x[Nsh:-Nsh]

    # estimate peak position and fwhm width
    peakval, peakpos = pickpeak(profsmooth, 1)

    try:
        halfval, halfpos = pickpeak(
            -np.abs(profsmooth - (peakval + np.nanmin(profsmooth)) / 2.0),
            npicks=2)
        width = np.abs(np.diff(xs[halfpos]))
    except:
        print("Warning: can't determine initial guess for width", sys.exc_info())
        width = 20

    off = np.nanmin(profsmooth)  # TODO: can we do better (robust?)
    try:
        m = xs[peakpos]
    except IndexError:
        m = 0.5 * (x[0] + x[-1])

    s = width
    A = peakval - off

    # make gaussian fit
    #startpars = np.r_[A, m, s, off]
    startpars = np.r_[m, s, A, off]

    def gauss1d(pars, x, v0 = 0):
        """calculate 1d gaussian.
        @return: difference of 1d gaussian and reference (data) values
        @param pars: parameters of gaussian. see source.
        @param x: x values
        @param v0: reference value
        """
        m, s, A, offs = pars[0:4]
        # Add the conditions to prevent truedivide by 0 warning in some cases
        if s == 0:
            s = 0.0001
            v = A*np.exp(- (x-m)**2 / (2*s**2)) + offs
        else:
            v = A * np.exp(- (x - m) ** 2 / (2 * s ** 2)) + offs
        return v-v0

    fitpar = optimize.leastsq(gauss1d,startpars,args = (x, prof))
    # else:
    #     fitpar = LM.LM(self.fJgauss1d,
    #                    startpars,
    #                    args=(x, prof),
    #                    kmax=30,
    #                    eps1=1e-6,
    #                    eps2=1e-6,
    #                    verbose=self.verbose,
    #                    )
    return fitpar


# def fit(function, parameters, y, x=None):
#     def f(params):
#         i = 0
#         for p in parameters:
#             p.set(params[i])
#             i += 1
#         return y - function(x)
#
#     if x is None: x = np.arange(y.shape[0])
#     p = [param() for param in parameters]
#     return optimize.leastsq(f, p)

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

def fitgauss1d(xx, zz, truncate=True):
    '''
    fit 1d gaussian function using scipy.optimize.leastsq
    :param xx:
    :param yy:
    :param truncate:
    :return:
    '''
    if truncate:
        x0, xx, zz = truncate_center(xx, zz)
    else:
        x0 = np.sum(xx * zz) / np.sum(zz)
    # mu = Parameter(x0)
    # background = Parameter(min(zz))
    # # background = Parameter(1 / np.average([1/n**2 for n in yy]))
    # height = Parameter(max(zz) - background())
    # prep_sigma = xx[zz > (height() * np.exp(-1 / 2)) + background()]
    # sigma = Parameter(abs(prep_sigma[-1] - prep_sigma[0]) / 2)

    def f(x):
        return height() * np.exp(-((x - mu()) / sigma()) ** 2 / 2) + background()
    #fitresults=fit(f, [mu, sigma, height, background], zz, x=xx) old bad function
    #print("Newfit = ",find_startpar_gauss(xx,zz))
    return find_startpar_gauss(xx,zz)


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
    import cv2
    data = cv2.imread("test_image1.jpg")
    lx, ly, lz = data.shape
    print(data.shape)
    xx = np.arange(lx)
    yy = np.arange(ly)
    xx, yy = np.meshgrid(np.arange(ly), np.arange(lx))
    zz = data[::, :: ,2]
    #print(xx.shape,yy.shape, zz.shape)

    start_time = time.time()
    for i in range(1):
        fitgauss2d_section(np.arange(0, ly), np.arange(0, lx), zz)
    print("--- %.8f seconds ---" % (time.time() - start_time))
    #print(fitgauss2d_section(np.arange(0, ly), np.arange(0, lx), zz))






    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    #xx,yy = np.meshgrid(np.arange(4000),np.arange(3000))
    #zz = (gauss2d(1000, 1000, 100, 400, 1, 0, xx, yy) + 1 + np.random.rand(*(xx.shape)) * 0.5)*100

    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()

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
