"""
Store every experimental informations, and some externals calculation.


"""

from numpy import array as nparray
from numpy import nan as npnan
from numpy import ones as npones
from numpy import mean as npmean

from matplotlib import pyplot as plt


class ExperimentalInfo(object):
    """
    This class let you access to all information we get on experiments.
    """

    def __init__(self):
        pass

    def __str__(self):
        pass

    # === general informations about exp === #
    def rep1(serie=[], option={}):
        n_serie = 10
        if type(serie) != int:
            # it's a list
            if len(serie) == 0:
                serie = range(0, n_serie)
            if len(serie) > n_serie:
                raise ExperimentalInfoError(
                    'Serie list is too long.'
                )
            if max(serie) > n_serie:
                raise ExperimentalInfoError(
                    'Serie index out of range.'
                )
        else:
            # it's a int
            if serie > n_serie:
                raise ExperimentalInfoError(
                    'Serie index out of range.'
                )

        # === user entrance information === #
        t_givrage = nparray([
            40, 43, 51, 46, 46,
            38, 80, 76, 70, npnan,
        ])
        t_total = nparray([
            120, 266, 177, 193, 329,
            324, 80, 90, 70, 166
        ])
        air_temperature = nparray([
            20.6, 20.6, 21., 21.2, 21.2,
            21.6, 21.6, 21.8, 21.5, 21.5
        ])

        ca_left = nparray([
            0
        ])

        ca_right = nparray([
            0
        ])

        ca = (ca_left + ca_right) / 2

        ca = nparray([
            51., 78., 74., 76., 56.,
            62., 63., 75., 64., 81,
        ])
        Tw = -7.*npones(n_serie)

        # /!\ l'angle 9, 7 on été mesuré après début du givrage #
        try:
            if option['wall temperature']:
                return Tw
        except Exception as e:
            pass

        return(
            t_givrage[serie], t_total[serie],
            air_temperature[serie], ca[serie]
        )

    def rep2(serie=[], option={}):
        n_serie = 10
        if type(serie) != int:
            # it's a list
            if len(serie) == 0:
                serie = range(0, n_serie)
            if len(serie) > n_serie:
                raise ExperimentalInfoError(
                    'Serie list is too long.'
                )
            if max(serie) > n_serie:
                raise ExperimentalInfoError(
                    'Serie index out of range.'
                )
        else:
            # it's a int
            if serie > n_serie:
                raise ExperimentalInfoError(
                    'Serie index out of range.'
                )

        t_givrage = nparray([
            14, 20, 44, 40, 38,
            34, 34, 35, 35, 37,
        ])
        t_total = nparray([
            158, 86, 46, 50, 51,
            50, 54, 56, 45, 52,
        ])
        air_temperature = nparray([
            22.0, 22.8, 22.6, 22.8, 22.8,
            22.8, 22.8, 22.8, 22.7, 22.7,
        ])
        ca_left = nparray([
            0
        ])

        ca_right = nparray([
            0
        ])

        Tw = -7.*npones(n_serie)

        try:
            if option['wall temperature']:
                return Tw
        except Exception as e:
            pass

        ca = (ca_right + ca_left) / 2
        ca = nparray([
            60., 64., 79., 79., 79.,
            71., 76., 75., 72., 80.,
        ])

        return(
            t_givrage[serie], t_total[serie],
            air_temperature[serie], ca[serie]
        )





class ExperimentalInfoError(Exception):
    pass

if __name__ == '__main__':
    import numpy as np

    a = ExperimentalInfo.rep1()
    b = ExperimentalInfo.rep2()

    idx = np.argsort(a[3])
    ca1 = a[3][idx]
    tt1 = a[1][idx]
    tg1 = a[0][idx]

    idx = np.argsort(b[3])
    ca2 = b[3][idx]
    tt2 = b[1][idx]
    tg2 = b[0][idx]

    plt.figure(figsize=(11, 7))
    plt.plot(
        ca1, tt1,
        '--sk', linewidth=2, markersize=8,
        label='rep1 : total'
    )
    plt.plot(
        ca1, tg1,
        '--sb', linewidth=2, markersize=8,
        label='rep1 : solidification')

    plt.plot(
        ca2, tt2,
        '--*k', linewidth=2, markersize=8,
        label='rep2 : total'
    )
    plt.plot(
        ca2, tg2,
        '--*b', linewidth=2, markersize=8,
        label='rep2 : solidification'
    )

    plt.xlabel('Contact angle (°)', fontsize=20)
    plt.ylabel('Temps de solidification', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.grid(True)
    plt.legend(shadow=True, fontsize=15)
    plt.show()
