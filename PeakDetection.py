from numpy import NaN, Inf, arange, isscalar, array
from numpy import float as npfloat


class PeakDetection():
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html.

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """

    def __init__(self, y, sel, x=None):
        self.max_location = []
        self.max_magnitude = []
        self.min_location = []
        self.min_magnitude = []
        """Lauch the function."""
        if x is None:
            x = arange(len(y))
        y = array(y, dtype=npfloat)

        if len(y) != len(x):
            raise PeakDetectionError('The length of y and x must be the same')

        if not isscalar(sel) or sel <= 0:
            raise PeakDetectionError('The selectivity must be a positive \
                scalar')

        temp_max_bound, temp_min_bound = Inf, -Inf
        temp_max_loc, temp_min_loc = NaN, NaN

        look_for_max = True
        """
        look_for_max : true if we are looking for a peak, False if find a
        valley
        """

        for i in arange(len(y)):
            this = y[i]
            if this > temp_min_bound:
                temp_min_bound = this
                temp_min_loc = x[i]
            if this < temp_max_bound:
                temp_max_bound = this
                temp_max_loc = x[i]

            if look_for_max:
                if this < temp_min_bound-sel:
                    self.max_location.append(temp_min_loc)
                    self.max_magnitude.append(temp_min_bound)
                    temp_max_bound = this
                    temp_max_loc = x[i]
                    look_for_max = False
            else:
                if this > temp_max_bound+sel:
                    self.min_location.append(temp_max_loc)
                    self.min_magnitude.append(temp_max_bound)
                    temp_min_bound = this
                    temp_min_loc = x[i]
                    look_for_max = True

    def __str__(self):
        """Print the peaks locations."""
        return str(self.max_location+self.min_location)


class PeakDetectionError(Exception):
    """Error class for PeakDetection."""

    pass


if __name__ == "__main__":
    from matplotlib.pyplot import plot, scatter, show, figure
    import numpy as np

    # Example 1
    series = [0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0]

    peaks = PeakDetection(series, .3)

    figure('Example 1')
    plot(series)
    scatter(peaks.max_location, peaks.max_magnitude, color='blue')
    scatter(peaks.min_location, peaks.min_magnitude, color='red')

    # Example 2
    t = np.linspace(0, 20, 1000)
    x0 = (np.sin(2*np.pi*.2*t) * np.cos(2*np.pi*.05*t)) * \
        np.exp(-t/10) + \
        np.random.rand(len(t))/20

    peaks2 = PeakDetection(x0, .1, t)

    figure('Example 2')
    plot(t, x0, 'k')
    scatter(peaks2.max_location, peaks2.max_magnitude, color='blue')
    scatter(peaks2.min_location, peaks2.min_magnitude, color='red')

    # Expample 3
    t = np.linspace(0, 20, 1000)
    mu1 = 3
    s1 = 2
    mu2 = 12
    s2 = 5
    g1 = np.exp(-1/2*(t-mu1)**2/s1**2) / (s1*np.sqrt(2*np.pi))
    g2 = np.exp(-1/2*(t-mu2)**2/s2**2) / (s2*np.sqrt(2*np.pi))
    sig = g1+g2+np.random.rand(len(t))/80

    peaks3 = PeakDetection(sig, .02, t)

    figure('Example 3')
    plot(t, sig, 'k')
    scatter(peaks3.max_location, peaks3.max_magnitude, color='blue')
    scatter(peaks3.min_location, peaks3.min_magnitude, color='red')

    show()
