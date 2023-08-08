import sys
import numpy as np
import time
from scipy import optimize,signal

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

    Nsh = 20   # number (half) of points for smoothing. Needs to be even no. Choose according to feature that needs to be resolved
    gs = Nsh / 2  # width gaussian

    # use normalized gaussian for smoothing
    gx = np.arange(2 * Nsh + 1) - Nsh
    gy = np.exp(-gx ** 2 / gs ** 2)
    gy /= gy.sum()

    # smooth profile, limit axes to valid values
    profsmooth = np.convolve(prof, gy, mode='valid')  #reduced in length by 2Nsh
    #profsmooth = signal.savgol_filter(prof,Nsh,Nsh/2) #adj, Nsh is the window
    #profsmooth =profsmooth[Nsh:-Nsh] # adj
    xs = x[Nsh:-Nsh]

    # estimate peak position and fwhm width
    peakval, peakpos = pickpeak(profsmooth, 1)

    off = np.nanmin(profsmooth)  # TODO: can we do better (robust?)
    try:
        halfval, halfpos = pickpeak(
            -np.abs(profsmooth - (peakval + off) / 2.0),
            npicks=2)
        width = np.abs(np.diff(xs[halfpos]))
    except:
        # print("Warning: can't determine initial guess for width", sys.exc_info())
        width = 20

    try:
        m = xs[peakpos]
    except IndexError:  # clearly a known error that forces the scipy.optimize.leastsq to evaluate needlessly
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

    #def f(x):
    #    return height() * np.exp(-((x - mu()) / sigma()) ** 2 / 2) + background()
    #fitresults=fit(f, [mu, sigma, height, background], zz, x=xx) old bad function
    #print("Newfit = ",find_startpar_gauss(xx,zz))
    return find_startpar_gauss(xx,zz)


def fitgauss1d_multiple_oneatatime(xx, y, zz, startpars, par_bounds, pars_to_fit, dir=-2, truncate=True):
    """
    Fit (in 1d) one of multiple gaussians on a cross-section of a 2d image using initial parameters

    :param xx: a 1d list with values equal to indices representing the (x or y) coordinates to fit over
    :param y: the (y or x) coordinate of the cross-section to fit the gaussian on
    :param zz: a 1d list of 8-bit integer values corresponding to the camera intensity readings over the cross-section
    :param startpars: a list of the initial parameters for each 2d gaussian
    :param par_bounds: the bounds for each parameter to be fit between
    :param pars_to_fit: An integer representing the index in startpars of the parameters of the gaussian to fit
    :param dir: an integer indicating the direction of the cross-section- dir <=0 for x direction, dir > 0 for y
    :param truncate: whether to truncate the data before fitting

    :return: the modified startpars containing the newly fitted parameters at index pars_to_fit
    """

    # Truncate the data before fitting
    if truncate:
        x0, xx, zz = truncate_center(xx, zz)

    # Enforce parameter bounds on startpars
    for i in range(len(startpars)):
        for j in range(len(startpars[i])):
            if startpars[i][j] < par_bounds[0][j]:
                startpars[i][j] = par_bounds[0][j]
            elif startpars[i][j] > par_bounds[1][j]:
                startpars[i][j] = par_bounds[1][j]

    # Extract the parameters to fit from startpars, as a copy (so it can be modified)
    startpars_to_fit = startpars[pars_to_fit][:]

    # Make a mask of zz so it can be modified
    zz_reduced = [0 for _ in zz]

    # Isolate the gaussian we want to fit by decrementing (into zz_reduced) zz by the effects of every other gaussian
    for i in range(len(xx)):
        for j in range(len(startpars)):
            if j != pars_to_fit:
                if dir <= 0:
                    zz_reduced[i] = max(zz[i] - gauss2d(startpars[j][0], startpars[j][1], startpars[j][2],
                                                        startpars[j][3], startpars[j][4], startpars[j][5], xx[i], y), 0)
                else:
                    zz_reduced[i] = max(zz[i] - gauss2d(startpars[j][0], startpars[j][1], startpars[j][2],
                                                        startpars[j][3], startpars[j][4], startpars[j][5], y, xx[i]), 0)

    def gauss2d_wrapper(x, mu_x, mu_y, sigma_x, sigma_y, height, theta, y, dir=0):
        """
        Wrapper for gauss2d so that optimize.curve_fit can interact with it properly

        :param x: the (x or y) coordinate at which to evaluate gauss2d
        :param mu_x: the x center of the gaussian
        :param mu_y: the y center of the gaussian
        :param sigma_x: the x standard deviation of the gaussian
        :param sigma_y: the y standard deviation fo the gaussian
        :param height: the height of the gaussian
        :param theta: the floor of the gaussian
        :param y: the (y or x) coordinate at which to evaluate gauss2d
        :param dir: an integer indicating the direction of the cross-section- dir <=0 for x direction, dir > 0 for y

        :return: the result of evaluating the gaussian at the point
        """
        if dir <= 0:
            return gauss2d(mu_x, mu_y, sigma_x, sigma_y, height, theta, x, y)
        else:
            return gauss2d(mu_x, mu_y, sigma_x, sigma_y, height, theta, y, x)

    # Add the (y or x) coordinate of the cross-section and the direction indicator to the parameters so curve_fit can
    # use them
    startpars_to_fit.extend([y, dir])
    bounds = (par_bounds[0][:], par_bounds[1][:])
    bounds[0].extend([y, dir])
    bounds[1].extend([y + 1, dir + 1])

    # Fit the parameters in startpars_to_fit using optimize.curve_fit
    optim, pcov = optimize.curve_fit(gauss2d_wrapper, xx, zz_reduced, p0=startpars_to_fit, bounds=bounds)

    # Modify startpars to contain the newly fitted parameters
    startpars[pars_to_fit] = optim[:-2].tolist()

    return startpars


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


def detect_peaks(image, num_fits, slices, blacklist=()):
    """
    Estimate at most num_fits peaks in a 2d image, excluding any peaks on blacklist

    :param image: a 2d array of 8-bit ints representing the intensity at every pixel on the camera's image
    :param num_fits: the number of peaks to find
    :param slices: the number of equal slices to divide the image into when looking for peaks
    :param blacklist: a tuple/list of peaks to exclude from the result

    :return: a list of at most num_fits peaks, each represented by a list of [x, y] coordinates
    """

    # Return list
    peaks = []

    # Take horizontal row samples of the image to check for peaks between
    horiz_slices = []
    for i in range(slices):
        horiz_slices.append(image[i * len(image) // slices])

    # Keep track of peaks in each horizontal slice
    horiz_peaks = [[] for _ in horiz_slices]

    # Iterate over each horizontal slice to find num_fits peaks for each slice
    for slice in range(len(horiz_slices)):

        # List of all local maximums found within the slice
        hrz_pks = []

        prev_slope = 0
        prev_idx = 0

        # Iterate over the slice within slices steps
        for i in range(slices):
            # Calculate the indices at the beginning and end of the step
            bef_idx = i * len(horiz_slices[slice]) // slices
            aft_idx = min((i + 1) * len(horiz_slices[slice]) // slices, len(horiz_slices[slice]) - 1)

            # Do a linear fit to find the average slope over the step
            slope = np.polyfit(np.arange(start=bef_idx, stop=aft_idx), horiz_slices[slice][bef_idx:aft_idx], 1)[0]

            # Compare to the slope from the previous step to determine if this pair of steps contains a local maximum
            if slope < 0 < prev_slope:

                # Save the peak (exact location by finding the simple maximum between prev_idx and aft_idx on the slice)
                # to hrz_pks, along with a score indicating the severity of the peak
                hrz_pks.append([horiz_slices[slice].tolist().index(max(horiz_slices[slice][prev_idx:aft_idx]),
                                                                   prev_idx, aft_idx), prev_slope**2 + slope**2])

                # Exclude this peak if it appears in blacklist
                if hrz_pks[-1][0] in [blpk[1] for blpk in blacklist]:
                    hrz_pks.pop(-1)

            prev_slope = slope
            prev_idx = bef_idx

        # Save the num_fits most prominent peaks in the slice to horiz_peaks
        for _ in range(num_fits):
            if len(hrz_pks) > 0:

                # Find and append the most prominent peak remaining in hrz_pks to horiz_peaks
                max_pk = 0
                for j in range(len(hrz_pks)):

                    # Compare peaks using a hybrid score of (peak height * slope change severity score)
                    if horiz_slices[slice][hrz_pks[j][0]] * hrz_pks[j][1] >\
                            horiz_slices[slice][hrz_pks[max_pk][0]] * hrz_pks[max_pk][1]:
                        max_pk = j
                horiz_peaks[slice].append(hrz_pks.pop(max_pk)[0])

        # Sort the peaks within the slice, so they will line up with each other between slices, for merging
        horiz_peaks[slice].sort()

    # Average each peak (assumed to be in order from the sort) over all the slices
    horiz_peaks_condensed = []
    peak_totals = [0 for _ in range(num_fits)]
    for i in range(slices):
        for j in range(min(num_fits, len(horiz_peaks[i]))):
            if len(horiz_peaks_condensed) <= j:
                horiz_peaks_condensed.append(0)
            horiz_peaks_condensed[j] += horiz_peaks[i][j] * horiz_slices[i][horiz_peaks[i][j]] ** 2
            peak_totals[j] += horiz_slices[i][horiz_peaks[i][j]] ** 2
    for i in range(len(horiz_peaks_condensed)):
        horiz_peaks_condensed[i] /= peak_totals[i]

    # Take one vertical slice for each horizontal peak found
    vert_slices = []
    for j in range(len(horiz_peaks_condensed)):
        vert_slices.append([image[i][int(horiz_peaks_condensed[j])] for i in range(len(image))])

    # List of all vertical local maximums found within the slices in vert_slices
    vrt_pks = []
    for j in range(len(vert_slices)):

        prev_slope = 0
        prev_idx = 0

        # Iterate over the slice within slices steps
        for i in range(slices):

            # Calculate the indices at the beginning and end of the step
            bef_idx = i * len(vert_slices[j]) // slices
            aft_idx = min((i + 1) * len(vert_slices[j]) // slices, len(vert_slices[j]) - 1)

            # Do a linear fit to find the average slope over the step
            slope = np.polyfit(np.arange(start=bef_idx, stop=aft_idx), vert_slices[j][bef_idx:aft_idx], 1)[0]

            # Compare to the slope from the previous step to determine if this pair of steps contains a local maximum
            if slope < 0 < prev_slope:

                # Save the peak (exact location by finding the simple maximum between prev_idx and aft_idx on the slice)
                # to vrt_pks, along with a score indicating the severity of the peak and the number of the slice it was
                # found on
                vrt_pks.append([vert_slices[j].index(max(vert_slices[j][prev_idx:aft_idx]), prev_idx, aft_idx),
                                prev_slope**2 + slope**2, j])

                # Exclude this peak if it appears in blacklist
                if vrt_pks[-1][0] in [blpk[0] for blpk in blacklist]:
                    vrt_pks.pop(-1)

            prev_slope = slope
            prev_idx = bef_idx

    # Store the num_fits most prominent peaks
    vert_peaks = []

    for _ in range(num_fits):
        if len(vrt_pks) > 0:
            # Find the most prominent peak in vrt_pks
            max_pk = 0
            for k in range(len(vrt_pks)):

                # Compare peaks using a hybrid score of (peak height * slope change severity score)
                if vert_slices[vrt_pks[k][2]][vrt_pks[k][0]] * vrt_pks[k][1] >\
                        vert_slices[vrt_pks[max_pk][2]][vrt_pks[max_pk][0]] * vrt_pks[max_pk][1]:
                    max_pk = k

            # Append the peak to vert_peaks and save the horizontal slice number
            j = vrt_pks[max_pk][2]
            vert_peaks.append(vrt_pks.pop(max_pk)[0])

            # Save the peak to the return list
            peaks.append([int(vert_peaks[-1]), int(horiz_peaks_condensed[j])])
            # peaks.append([round(horiz_peaks_condensed[j]), round(vert_peaks[-1])])

    # Sort peaks by height so that they're returned in a somewhat consistent order
    for i in range(len(peaks)):
        k = 0
        for j in range(len(peaks)):
            if image[peaks[j][0]][peaks[j][1]] > image[peaks[k][0]][peaks[k][1]]:
                k = j
        if k != i:
            peaks.insert(i, peaks.pop(k))

    return peaks


def fitgauss2d_multiple(image, xx, yy, num_fits, slices):
    """
    Fit at most num_fits 2d gaussians to the image represented by image

    :param image: a 2d array of 8-bit ints representing the intensity at every pixel on the camera's image
    :param xx: a 1d array of size equal to the x dimension of the camera's image, with values equal to indices
    :param yy: a 1d array of size equal to the y dimension of the camera's image, with values equal to indices
    :param num_fits: the number of gaussians to fit
    :param slices: the number of equal slices to divide the image into when looking for initial peaks

    :return: a list of at most num_fits 2d gaussian fits, each represented by a list containing (in order) x center,
    y center, x standard deviation, y standard deviation, height, and floor
    """

    # Initial estimate of peak positions
    peaks = detect_peaks(image, num_fits, slices)

    def find_fwhm(zz, peak):
        """
        Calculate the FWHM of one non-overlapping peak in a 1d cross-section that may have multiple peaks

        :param zz: a 1d array representing the x or y position (indices) and data (values)
        :param peak: an integer representing the location of the peak to find the FWHM of

        :return: the FWHM of the peak
        """
        hwhm = [10, 10]
        for i in range(len(zz)):
            if peak + i < len(zz) and hwhm[0] <= 10 and zz[peak + i] <= zz[peak] / 2:
                hwhm[0] = i
            if peak - i >= 0 and hwhm[1] <= 10 and zz[peak - i] <= zz[peak] / 2:
                hwhm[1] = i
        return max(hwhm[0] + hwhm[1], 20)

    min_val = int(np.nanmin(image))

    # Estimate for initial parameters
    guess = []

    # List to store peaks that are deemed too much overlapping another, if we decide later we want to use them
    rejected_peaks = []

    # Loop once for each peak/gaussian
    i = 0
    while i < len(peaks):
        peak = peaks[i]

        # Initialize ith parameter estimate
        guess.append([0, 0, 0, 0, 0, 0])

        # Store guess for mu_x and mu_y
        guess[-1][0] = peak[0]
        guess[-1][1] = peak[1]

        # Calculate and store guess for sigma_x
        guess[-1][2] = find_fwhm(image[::, peak[1]], peak[0]) / (2 * np.sqrt(2 * np.log(2)))

        # Calculate and store guess for sigma_y
        guess[-1][3] = find_fwhm(image[peak[0], ::], peak[1]) / (2 * np.sqrt(2 * np.log(2)))

        # Calculate and store guess for the gaussian's height
        try:
            local_image = image[round(peak[0] - guess[-1][2]):round(peak[0] + guess[-1][2]),
                                round(peak[1] - guess[-1][3]):round(peak[1] + guess[-1][3])]
            guess[-1][4] = int(np.nanmax(local_image))
            updated_peak = np.where(local_image == guess[-1][4])
            guess[-1][0] = updated_peak[0][0] + round(peak[0] - guess[-1][2])
            guess[-1][1] = updated_peak[1][0] + round(peak[1] - guess[-1][3])
        except ValueError:
            guess[-1][4] = int(image[peak[0]][peak[1]] * 1.5)

        # Guess for the floor
        guess[-1][5] = min_val

        # Check if any peaks overlap too much, and remove (and store in rejected_peaks) and replace them if they do
        for j in range(len(guess) - 2, -1, -1):
            overlap = 2
            if (guess[j][0] - guess[j][2] * overlap < guess[-1][0] < guess[j][0] + guess[j][2] * overlap) and \
                    (guess[j][1] - guess[j][3] * overlap < guess[-1][1] < guess[j][1] + guess[j][3] * overlap) and\
                    i < num_fits * 4:
                rejected_peaks.append((guess[-1], np.abs(guess[-1][0] - guess[j][0]) / guess[j][2] +
                                       np.abs(guess[-1][1] - guess[j][1]) / guess[j][3]))
                peaks.extend(detect_peaks(image, 1, slices, blacklist=[(peak[0], peak[1]) for peak in peaks]))
                guess.pop(-1)
            elif (guess[-1][0] - guess[-1][2] * overlap < guess[j][0] < guess[-1][0] + guess[-1][2] * overlap) and \
                    (guess[-1][1] - guess[-1][3] * overlap < guess[j][1] < guess[-1][1] + guess[-1][3] * overlap) and \
                    i < num_fits * 4:
                rejected_peaks.append((guess[j], np.abs(guess[j][0] - guess[-1][0]) / guess[-1][2] +
                                       np.abs(guess[j][1] - guess[-1][1]) / guess[-1][3]))
                peaks.extend(detect_peaks(image, 1, slices, blacklist=[(peak[0], peak[1]) for peak in peaks]))
                guess.pop(j)

        # Stop adding guesses if we've reached num_fits
        if len(guess) >= num_fits:
            break
        i += 1

    # Remove any duplicate guesses
    for gs1 in guess:
        for gs2 in guess:
            if (gs1 is not gs2) and gs1 == gs2:
                guess.remove(gs2)

    # If we've not yet reached num_fits and have leftover peaks in rejected_peaks, add the least overlapping ones to
    # fill out the guesses
    while len(guess) < num_fits and len(rejected_peaks) > 0:
        max_dev = 0
        for i in range(len(rejected_peaks)):
            if rejected_peaks[i][1] > rejected_peaks[max_dev][1]:
                max_dev = i
        if not rejected_peaks[max_dev][0] in guess:
            guess.append(rejected_peaks.pop(max_dev)[0])
        else:
            rejected_peaks.pop(max_dev)

    # Sort the guesses so the fits occur in a consistent order
    guess.sort()
    # for gs in guess:
    #     gs.insert(0, gs.pop(1))
    #     gs.insert(2, gs.pop(3))
    # return guess

    p = []

    for i in range(len(guess)):
        i %= len(guess)

        local_bounds = ((max(round(guess[i][0] - guess[i][2] * 2.5), 0),
                         min(round(guess[i][0] + guess[i][2] * 2.5), len(image[::, 0]))),
                        (max(round(guess[i][1] - guess[i][2] * 2.5), 0),
                         min(round(guess[i][1] + guess[i][2] * 2.5), len(image[0, ::]))))
        local_image = image[local_bounds[0][0]:local_bounds[0][1], local_bounds[1][0]:local_bounds[1][1]]

        try:
            p.append(list(fitgauss2d_section(np.arange(local_image.shape[1]),
                                             np.arange(local_image.shape[0]), local_image)[0]))
            p[-1][0] += local_bounds[0][0]
            p[-1][1] += local_bounds[1][0]
            if p[-1][0] < 0 or p[-1][0] > len(image[::, 0]):
                p[-1][0] = guess[-1][0]
            if p[-1][1] < 0 or p[-1][1] > len(image[0, ::]):
                p[-1][1] = guess[-1][1]
            if p[-1][2] < guess[i][2] / 2 or p[-1][2] > guess[i][2] * 2:
                p[-1][2] = guess[i][2]
            if p[-1][3] < guess[i][3] / 2 or p[-1][3] > guess[i][3] * 2:
                p[-1][3] = guess[i][3]
            if p[-1][4] <= 0 or p[-1][4] > max([gs[4] * 2 for gs in guess]):
                p[-1][4] = guess[i][4]
            if p[-1][5] < 0 or p[-1][5] > min(max([gs[4] / 4 for gs in guess]), min_val * 3):
                p[-1][5] = min_val
        except (IndexError, TypeError) as e:
            # print('Fitting failed for curve', i + 1, ':', e)
            p.append(guess[i])

    # Sort the gaussian fits, so they are returned in a consistent order
    p.sort()

    # Swap x and y parameters in the result
    for gs in p:
        gs.insert(0, gs.pop(1))
        gs.insert(2, gs.pop(3))

    return p


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
#    fit_int_x = fitgauss1d(xx, zz.sum(axis=0))
#    fit_int_y = fitgauss1d(yy, zz.sum(axis=1))
    fit_int_x = fitgauss1d(xx, np.nanmax(zz,axis=0))      #adj
    fit_int_y = fitgauss1d(yy, np.nanmax(zz,axis=1))      #adj
    # print(fit_int_x)
    # print(fit_int_y)
    # fit_x = fitgauss1d(xx, zz[np.min(np.abs(yy - fit_int_y[0][0])) == np.abs(yy - fit_int_y[0][0]), ::].flatten())
    # fit_y = fitgauss1d(yy, zz[::, np.min(np.abs(xx - fit_int_x[0][0])) == np.abs(xx - fit_int_x[0][0])].flatten())
    #print('\n',fit_int_y[0][0]) #debugging

    # fit_x = fitgauss1d(xx, zz[round(fit_int_y[0][0]), ::])
    # fit_y = fitgauss1d(yy, zz[::, round(fit_int_x[0][0])])

    # The estimate may be out of range.
    if 1 <= fit_int_y[0][0] <= zz.shape[0]:
        fit_x = fitgauss1d(xx, zz[round(fit_int_y[0][0]), ::])
    else:
        fit_x = [[0, float('inf'), 0, 0], 1]
    if 1 <= fit_int_x[0][0] <= zz.shape[1]:
        fit_y = fitgauss1d(yy, zz[::, round(fit_int_x[0][0])])
    else:
        fit_y = [[0, float('inf'), 0, 0], 1]
    p = (fit_x[0][0], fit_y[0][0], fit_x[0][1], fit_y[0][1], (fit_x[0][2] + fit_y[0][2]) / 2, (fit_x[0][3] + fit_y[0][3]) / 2, )

    if all([fit_int_x[1] == 1, fit_int_y[1] == 1, fit_x[1] == 1, fit_y[1] == 1]):
        ier = 1
    else:
        ier = 0
    return p, ier


def gauss2d(mu_x, mu_y, sigma_x, sigma_y, height, theta, x, y):
    return height * np.exp(-((x - mu_x) * np.cos(theta) + (y - mu_y) * np.sin(theta)) ** 2 / (2 * sigma_x ** 2) -
                           (-(x - mu_x) * np.sin(theta) + (y - mu_y) * np.cos(theta)) ** 2 / (2 * sigma_y ** 2))


def gauss2d_multiple(pars, x, y):
    """
    Evaluate the sum of multiple 2d gaussians at (x, y)

    :param pars: a list with a list of the parameters (mu_x, mu_y, sigma_x, sigma_y, height, theta) for each gaussian
    :param x: the x coordinate at which to evaluate the gaussians
    :param y: the y coordinate at which to evaluate the gaussians

    :return: the result of evaluating the sum of all the gaussians at (x, y)
    """

    result = 0

    for parset in pars:
        result += gauss2d(parset[0], parset[1], parset[2], parset[3], parset[4], parset[5], x, y)

    return result



def fitguase2d_int():
    return 0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
   # data = cv2.imread("test_image.jpg")
    data=cv2.imread(r'C:\Software Programming RiceYb\FlirCamera\IndividualAddressingPics\After_x8_BeamExpander_1_4_18_2022.jpg')
    lx, ly, lz = data.shape
    print(data.shape)
    xx = np.arange(lx)
    yy = np.arange(ly)
    xx, yy = np.meshgrid(np.arange(ly), np.arange(lx))
    zz = data[::, :: ,2]
    # print(xx.shape,yy.shape, zz.shape)

    start_time = time.time()
    #for i in range(1):
    #    fitgauss2d_section(np.arange(0, ly), np.arange(0, lx), zz)
    print("--- %.8f seconds ---" % (time.time() - start_time))
    print(fitgauss2d_section(np.arange(0, ly), np.arange(0, lx), zz))

    [par,ier]=fitgauss2d_section(np.arange(0, ly), np.arange(0, lx), zz) # fitting parameters adj




    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    xx,yy = np.meshgrid(np.arange(4000),np.arange(3000))
    #zz = (gauss2d(1000, 1000, 100, 400, 1, 0, xx, yy) + 1 + np.random.rand(*(xx.shape)) * 0.5)*100
    fit_Xzz=np.zeros(zz.shape)
    fit_Yzz=np.zeros(zz.shape)
    for i in range(lx):
        fit_Xzz[i,:]= gauss1d(par[1], par[3], par[5], yy[i,0])#* gauss1d(par[1], par[3], par[5], xx[0,j])  # no offset
            #print(gauss1d(par[0], par[2], par[4], yy[i,0]))#* gauss1d(par[1], par[3], par[5], xx[0,j]))
    for j in range(ly):
        fit_Yzz[:,j] = gauss1d(par[0], par[2], par[4], xx[0,j])
    surf2 = ax.plot_surface(xx, yy, fit_Xzz, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    surf3= ax.plot_surface(xx, yy, fit_Yzz, cmap=cm.hot, linewidth=0, antialiased=True)
    ax.set_xlabel('yy: 0-4000')
    ax.set_ylabel('xx: 0-3000')

    plt.show()

'''
     xx = np.arange(0, 4000, 1) * 0.57
     yy = gauss1d(1000, 200, 1, xx) + 1 + np.random.rand(xx.size) * 0.5
     plt.plot(xx, yy)
     plt.show()
     x0, sigma = fitgauss1d_moment(xx, yy)
     print(x0, sigma)
     result = fitgauss1d(xx, yy)
     print(result)

     start_time = time.time()
     for i in range(1000):
         x0, sigma = fitgauss1d_moment(xx, yy)
     print("--- %.8f seconds ---" % (time.time() - start_time))
     print(x0,sigma)
     start_time = time.time()
     for i in range(1000):
         result = fitgauss1d(xx, yy)
     print("--- %.8f seconds ---" % (time.time() - start_time))
     print(result)
'''