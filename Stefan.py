"""
Return theoritical front soldification.

Models :
    1 - steady monophasic
    2 - unsteady monophasic
    3 - pseudo 2D ()

:auth: Emeryk Ablonet - eablonet
:phD Student
:copyright: IMFT and Toulouse INP

:version: 4.1
"""
import numpy as np
import scipy.optimize as sp  # to have fsolve(), ...
import scipy.special as spe  # for special function (erf...)
from matplotlib import pyplot as plt


class Stefan(object):
    """Calculate the stefan based front position."""
    Tm = 273.15
    Tc = 273.15 - 9
    rho = [8000, 916.2, 999.89, 1.2]
    # density of substrat, solid phase, liquid phase and gaz surrounding
    k = [401, 2.22, .58, .024]
    # conductivty of substrat, solid phase, liquid phase and gaz surrounding
    alpha = [1.11e-4, 1.2e-6, .143e-6, 21.70e-6]
    # diffusivity of substrat, solid phase, liquid phase and gaz surrounding
    Lf = 333500

    def __init__(self, t):
        """Create the class."""
        self.time = t

    def __str__(self):
        """Return the wall temperature."""
        return str(self.Tc)

    def monophasic_steady(self):
        """
        Raise the front over time.

        Classical 1D Stefan model for solidification.
        """
        A = np.sqrt(
            2 * self.k[1] * (self.Tm-self.Tc) /
            (self.rho[1] * self.Lf)
        )
        self.front_position = A*np.sqrt(self.time)

    def monophasic_unsteady(self, st_point=.1, plot=False):
        """
        Raise the front over time.

        Classical unsteady Stefan Probelm solution founded by Neumann.
        """
        cp = self.k[1] / (self.rho[1]*self.alpha[1])
        Ste = cp * (self.Tm - self.Tc) / self.Lf

        def f(x):
            f = x * spe.erf(x) / np.exp(x**2) - Ste/np.sqrt(np.pi)
            return f
        delta = sp.fsolve(f, st_point)

        if plot:
            x = np.linspace(0, 1, 1000)
            plt.figure()
            plt.plot(x, f(x), '-k', linewidth=1)
            plt.plot(
                delta, f(delta),
                '.r', markersize=2, label='Root : '+str(delta)
            )
            plt.legend(fancybox=True, shadow=True)
            plt.grid(True)
            plt.ylabel(r'$\frac{xerf(x)}{exp(x^2)}-\frac{Ste}{\sqrt{\pi}}$')
            plt.xlabel(r'$x$')
            plt.show()

        self.front_position = 2*delta*np.sqrt(self.alpha[1]*self.time)

    def monophasic_lateral_flux(self, phi, theta, r0):
        """
        Raise the front over time.

        Model developed by D. Legendre for taking account of lateral flux from
        air.

        :param phi: flux lateral / flux conductif
        :param theta: angle de contact avec le substrat
        :param r0: rayon au pied de la goutte
        """
        ts = r0**2*self.Lf*self.rho[1] / (self.k[1]*(self.Tm - self.Tc))
        t = self.time/ts
        s = [0]
        for i in range(1, len(t)):
            s.append(
                np.sqrt(
                    s[i-1]**2 +
                    2*(t[i] - t[i-1]) *
                    (
                        1 +
                        phi / (1 - s[i-1]**2 - s[i-1] / np.tan(theta))
                    )
                )
            )
        s = np.array(s)
        self.front_position = s*r0

    def set_wall_temperature(self, Tc):
        """Define the wall temperature."""
        self.Tc = Tc + 273.15

    def set_density(self, value, phase=1):
        """
        Define the density.

        Sources:
        :copper: http://bit.ly/metals_allows_density -- ?°C
        :ice: http://bit.ly/ice_properties -- at 0°C
        :water: http://bit.ly/water_density_properties -- at 0°C
        :air: http://bit.ly/air_densiy_properties -- at 20°C
        """
        try:
            float(value)  # check if value is an int/float
            self.rho[phase] = value
        except ValueError:
            if value == 'copper':
                self.rho[phase] = 8940
            elif value == 'water':
                self.rho[phase] = 999.89
            elif value == 'ice':
                self.rho[phase] = 916.2
            elif value == 'air':
                self.rho[phase] = 1.204

    def set_enthalpie_fusion(self, Lf):
        """Define the enthalpi of fusion."""
        self.Lf = Lf

    def set_conductivity(self, value, phase=1):
        """Define the conductivity.

        Sources:
            :copper: http://bit.ly/conductivities -- at 25°C
            :water: http://bit.ly/conductivities -- at 25°C
            :ice: http://bit.ly/ice_properties -- at 0°C
            :air: http://bit.ly/conductivities -- at 25°C
        """
        try:
            float(value)  # check if value is an int/float
            self.k[phase] = value
        except ValueError:
            if value == 'copper':
                self.k[phase] = 401
            elif value == 'water':
                self.k[phase] = .58
            elif value == 'ice':
                self.k[phase] = 2.22
            elif value == 'air':
                self.k[phase] = .024

    def set_diffusivity(self, value, phase=1):
        """Define the thermal diffusivity.

        Sources:
            :copper:
            :water:
            :ice:
            :air: http://bit.ly/air_diffusivity -- at 20°C
        """
        try:
            float(value)  # check if value is an int/float
            self.alpha[phase] = value
        except ValueError:
            if value == 'copper':
                self.alpha[phase] = 1.11e-4
            elif value == 'water':
                self.alpha[phase] = .143e-6
            elif value == 'ice':
                self.alpha[phase] = 1.12e-6
            elif value == 'air':
                self.alpha[phase] = 21.7e-6


if __name__ == '__main__':
    t = np.linspace(0, 50, 3000)
    ste = Stefan(t)
    plt.figure()

    ste.set_wall_temperature(-7)
    ste.monophasic_unsteady()
    plt.plot(
        t, ste.s,
        '-r', linewidth=1,
        label='monophasic unsteady'
    )

    ste.monophasic_steady()
    plt.plot(
        t, ste.s,
        '-k', linewidth=1,
        label='monophasic steady'
    )
    flux = [-.2, -.1, -.05, .05, .1, .2]
    for phi in flux:
        ste.monophasic_lateral_flux(phi, 65*np.pi/180, 2.46e-3)
        plt.plot(
            t, ste.s,
            '-', linewidth=1,
            label='monophasic lateral : ' + str(phi)
        )

    plt.legend(fancybox=True, shadow=True)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Solidification front (m)')
    plt.show()
