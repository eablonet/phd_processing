# -*- coding: utf-8 -*-
"""
StacProcessing for solidification front detection.

Allow you to treating a stack of image.

:auth: Emeryk Ablonet - eablonet
:phD Student
:copyright: IMFT and Toulouse INP
:date : 2018

:version: 5.0
"""

import numpy as np
from scipy import optimize
from lmfit import Model as lmmodel
from lmfit import Parameters as lmparameter

import progressbar
import glob
import os

import PeakDetection as pf
import ImageProcessing as ip

from matplotlib import pyplot as plt
import cv2 as cv
from skimage import morphology
from skimage import measure

from scipy import ndimage
from scipy import interpolate

import Stefan as ste


class StackProcessing(object):
    """This class allow you to treat image by block."""

    def __init__(self, directory):
        """
        Class initiation.

        In this method we define the working directory, the number of image
        and we load the first image.

        :param directory: define the working directory

        :variables:
            :variable data_directory: working directory
            :variable ed: exporting directory
            :variable image_list: image list with ".tif" extension
            :variable ni_start: index of first image
            :variable im: image object created from ImageProcessing

        :methods:
            :method read_image: read an image

        """
        # initiate directories
        self.data_directory = directory
        self.define_exporting_directory()

        self.update_lists()

        self.current_image_number = 0
        self.read_image()

        # if there is no spatios, do not load the spatios
        if self.n_spatiox_tot < 1:
            self.current_spatiox_number = -1
        else:
            self.current_spatiox_number = 0

        if self.n_spatioy_tot < 1:
            self.current_spatioy_number = -1
        else:
            self.current_spatioy_number = 0

        self.read_spatio(type='x')
        self.read_spatio(type='y')

    def __str__(self):
        """Print the data directory on the terminal."""
        return self.data_directory

    def define_exporting_directory(self):
        """Define output directories."""
        self.exporting_directory = self.data_directory + 'export_image/'

        if not os.path.isdir(self.exporting_directory):
            os.makedirs(self.exporting_directory + 'lastOp/')
            os.makedirs(self.exporting_directory + 'spatiox/')
            os.makedirs(self.exporting_directory + 'spatioy/')
            print(
                'New directories have been created in :',
                self.exporting_directory
            )
        self.front_directory = self.exporting_directory.replace(
            'export_image', 'front'
        )
        if not os.path.isdir(self.front_directory):
            os.makedirs(self.front_directory)
        self.contour_directory = self.exporting_directory.replace(
            'export_image', 'contour'
        )
        if not os.path.isdir(self.contour_directory):
            os.makedirs(self.contour_directory)

        self.drop_info_directory = self.exporting_directory.replace(
            'export_image', 'drop_info'
        )
        if not os.path.isdir(self.drop_info_directory):
            os.makedirs(self.drop_info_directory)

    def update_lists(self):
        """
        Update the lists of image, spatiox and Spatio_y.

        ..todo : everything
        """
        # image list
        self.image_list = glob.glob((
            self.exporting_directory + 'lastOp/*.npy'
        ))
        if len(self.image_list) < 1:
            self.image_list = glob.glob((self.data_directory + 'cam1*.tif'))
        self.n_image_tot = len(self.image_list)

        # spatios lists
        self.spatiox_list = glob.glob((
            self.exporting_directory + 'spatiox/*.npy'
        ))
        self.n_spatiox_tot = len(self.spatiox_list)

        self.spatioy_list = glob.glob((
            self.exporting_directory + 'spatioy/*.npy'
        ))
        self.n_spatioy_tot = len(self.spatioy_list)

    def print_info(self, type=''):
        """
        Show info about the stack.

        Print :
            1. directory which is loaded
            2. number of image loaded
            3. current image selected
            4. first image number of the stack
            5. last image number of the stack
        """
        print(
            '\nDirectory containing data :\n   {0}'.format(self.data_directory)
        )
        print(
            'Directory for exporting :\n   {0}'
            .format(self.exporting_directory)
        )
        print('---image info---')
        print('   Number of image : {0}'.format(self.n_image_tot))
        print(
            '   Current image selected : {0}'
            .format(self.current_image_number)
        )
        print('---spatios info---')
        print('   Number of spatiox : {0}'.format(self.n_spatiox_tot))
        print('   Number of spatioy : {0}'.format(self.n_spatioy_tot))
        print(
            '   Current spatiox selected : {0}'
            .format(self.current_spatiox_number)
        )
        print(
            '   Current spatioy selected : {0}'
            .format(self.current_spatioy_number)
        )
        if type == 'all' or type == 'image':
            print('\n\n\n---List of images---')
            for i in range(len(self.image_list)):
                print('   #{0} : {1}'.format(i, self.image_list[i]))
        condition = (
            (
                type == 'all' or
                type == 'spatios' or
                type == 'spatiox'
            ) and
            self.n_spatiox_tot != 0
        )
        if condition:
            print('\n\n\n---List of spatiox---')
            for i in range(len(self.spatiox_list)):
                print('   #{0} : {1}'.format(i, self.spatiox_list[i]))
        condition = (
            (
                type == 'all' or
                type == 'spatios' or
                type == 'spatioy'
            ) and
            self.n_spatioy_tot != 0
        )
        if condition:
            print('\n\n\n---List of spatioy---')
            for i in range(len(self.spatioy_list)):
                print('   #{0} : {1}'.format(i, self.spatioy_list[i]))

        print('---end---\n\n')

    def read_image(self, ni=-1):
        """
        Read the image indicate by is number.

        :param ni : id number of the image to load
        """
        if ni == -1:
            ni = self.current_image_number
        else:
            self.current_image_number = ni
        try:
            self.current_image = ip.ImageProcessing(
                    self.image_list[ni]
            )
        except IndexError:
            print(
                'Image number', ni,
                'doesn''t exist, so first images was load'
            )
            self.read_image(0)

    def create_spatio(self, read_mask=False):
        """
        Create image spatio-temporal of the view.

        This action will create spatio, and save them in 2_results directory.
        This a little bit long (few minutes.)

        To optimise, crop the image before.
        """
        # === start progressbar === #
        widgets = ['Creating spatios',
                   ' ', progressbar.Percentage(),
                   ' ', progressbar.Bar('=', '[', ']'),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(
            widgets=widgets,
            maxval=self.current_image.size[0] + self.current_image.size[1]
        )
        pbar.start()

        # === create spatioy === #
        for row_id in range(0, self.current_image.size[1]):
            spy = np.zeros(
                [len(self.current_image.image[:, row_id]), self.n_image_tot]
            )
            # # initiate matrix size for spatioy
            for i in range(self.n_image_tot):
                self.read_image(i)
                if read_mask:
                    if i+1 < 9:
                        mask = np.load(
                            self.contour_directory + '000' + str(i+1) +
                            '_mask' + '.npy'
                        )
                    elif i+1 < 99:
                        mask = np.load(
                            self.contour_directory + '00' + str(i+1) +
                            '_mask' + '.npy'
                        )
                    elif i+1 < 999:
                        mask = np.load(
                            self.contour_directory + '0' + str(i+1) +
                            '_mask' + '.npy'
                        )
                    self.current_image.image[mask == 0] = 0
                spy[:, i] = self.current_image.image[:, row_id]
            if row_id < 10:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_000' +
                    str(row_id),
                    spy
                )
            elif row_id < 100:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_00' +
                    str(row_id),
                    spy
                )
            elif row_id < 1000:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_0' +
                    str(row_id),
                    spy
                )
            elif row_id < 10000:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_' +
                    str(row_id),
                    spy
                )
            pbar.update(row_id)

        # === create spatiox === #
        for col_id in range(0, self.current_image.size[0]):
            spx = np.zeros(
                [len(self.current_image.image[col_id, :]), self.n_image_tot]
            )
            # initiate matrix size for spatiox
            for i in range(self.n_image_tot):
                self.read_image(i)
                if read_mask:
                    if i+1 < 9:
                        mask = np.load(
                            self.contour_directory + '000' + str(i+1) +
                            '_mask' + '.npy'
                        )
                    elif i+1 < 99:
                        mask = np.load(
                            self.contour_directory + '00' + str(i+1) +
                            '_mask' + '.npy'
                        )
                    elif i+1 < 999:
                        mask = np.load(
                            self.contour_directory + '0' + str(i+1) +
                            '_mask' + '.npy'
                        )
                    self.current_image.image[mask == 0] = 0
                spx[:, i] = self.current_image.image[col_id, :]
            if col_id < 10:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_000' +
                    str(col_id),
                    spx
                )
            elif col_id < 100:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_00' +
                    str(col_id),
                    spx
                )
            elif col_id < 1000:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_0' +
                    str(col_id),
                    spx
                )
            elif col_id < 10000:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_' +
                    str(col_id),
                    spx
                )
            pbar.update(row_id + col_id)
        pbar.finish()

        # ===  update saptios infos === #
        self.update_lists()
        self.current_spatiox_number = 0
        self.current_spatioy_number = 0
        self.read_spatio(type='x')
        self.read_spatio(type='y')

    def create_spatio2(self, read_mask=False):
        """
        Create image spatio-temporal of the view.

        This action will create spatio, and save them in 2_results directory.
        This a little bit long (few minutes.)

        To optimise, crop the image before.
        """
        # === start progressbar === #
        widgets = ['Creating spatios',
                   ' ', progressbar.Percentage(),
                   ' ', progressbar.Bar('=', '[', ']'),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(
            widgets=widgets,
            maxval=(
                self.n_image_tot +
                self.current_image.size[0] + self.current_image.size[1]
            )
        )
        pbar.start()
        spy = np.zeros(
                [self.current_image.size[1],
                 self.current_image.size[0],
                 self.n_image_tot]
        )
        spx = np.zeros(
                [self.current_image.size[0],
                 self.current_image.size[1],
                 self.n_image_tot]
        )
        for i in range(self.n_image_tot):
            self.read_image(i)
            if read_mask:
                if i+1 < 10:
                    mask = np.load(
                        self.contour_directory + '000' + str(i+1) +
                        '_mask' + '.npy'
                    )
                elif i+1 < 100:
                    mask = np.load(
                        self.contour_directory + '00' + str(i+1) +
                        '_mask' + '.npy'
                    )
                elif i+1 < 1000:
                    mask = np.load(
                        self.contour_directory + '0' + str(i+1) +
                        '_mask' + '.npy'
                    )
                self.current_image.image[mask == 0] = 0

            # === create spatioy === #
            for row_id in range(0, self.current_image.size[1]):
                spy[row_id, :, i] = self.current_image.image[:, row_id]
            for col_id in range(0, self.current_image.size[0]):
                spx[col_id, :, i] = self.current_image.image[col_id, :]
            pbar.update(i)

        for col_id in range(self.current_image.size[1]):
            if col_id < 10:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_000' +
                    str(col_id),
                    spy[col_id, :, :]
                )
            elif col_id < 100:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_00' +
                    str(col_id),
                    spy[col_id, :, :]
                )
            elif col_id < 1000:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_0' +
                    str(col_id),
                    spy[col_id, :, :]
                )
            elif col_id < 10000:
                np.save(
                    self.exporting_directory + 'spatioy/' + 'i_' +
                    str(col_id),
                    spy[col_id, :, :]
                )
            pbar.update(self.n_image_tot + col_id)

        for row_id in range(self.current_image.size[0]):
            if row_id < 10:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_000' +
                    str(row_id),
                    spx[row_id, :, :]
                )
            elif row_id < 100:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_00' +
                    str(row_id),
                    spx[row_id, :, :]
                )
            elif row_id < 1000:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_0' +
                    str(row_id),
                    spx[row_id, :, :]
                )
            elif row_id < 10000:
                np.save(
                    self.exporting_directory + 'spatiox/' + 'i_' +
                    str(row_id),
                    spx[row_id, :, :]
                )
            pbar.update(self.n_image_tot + self.current_image.size[1] + row_id)
        pbar.finish()

        # ===  update saptios infos === #
        self.update_lists()
        self.current_spatiox_number = 0
        self.current_spatioy_number = 0
        self.read_spatio(type='x')
        self.read_spatio(type='y')

    def read_spatio(self, ni=-1, type='y'):
        """
        Read the spatio relative to is number.

        Need spatiox and/or spatioy were already created.

        :param ni: image number to read
        :param type: type of spatio. 'x' or 'y'. Default 'y'.
        """
        if ni == -1:
            if type == 'x':
                ni = self.current_spatiox_number
            elif type == 'y':
                ni = self.current_spatioy_number
        else:
            if type == 'x':
                self.current_spatiox_number = ni
            elif type == 'y':
                self.current_spatioy_number = ni

        if type == 'x' and ni >= 0 and ni < self.n_spatiox_tot:
                self.current_spatiox = ip.ImageProcessing(
                    self.spatiox_list[ni]
                    )
        elif type == 'y' and ni >= 0 and ni < self.n_spatioy_tot:
                self.current_spatioy = ip.ImageProcessing(
                    self.spatioy_list[ni]
                )
        elif type == 'x' and ni == -1:
            print('There is no spatiox')
            self.current_spatiox = []
        elif type == 'y' and ni == -1:
            print('There is no spatioy')
            self.current_spatioy = []
        else:
            print('Index out of the boundary, first spatio will be loaded')
            self.read_spatio(ni=ni, type=type)

    def remove_spatios(self):
        """Erase both spatios files."""
        for i in self.spatiox_list:
            os.remove(i)
        for i in self.spatioy_list:
            os.remove(i)
        self.update_lists()
        self.current_spatiox_number = -1
        self.current_spatioy_number = -1
        self.read_spatio(type='x')
        self.read_spatio(type='y')

    def treatment(self, treatment, *args, plot=True):
        """
        Apply a treatment to the current_image, and displays it.

        Only three treatments are available for now:
        ...crop : to crop the current_image (need 4 parameters:
                xmin, xmax, ymin, ymax)
        ...equalize : to equalize the image (global equalization, 2 parameters:
                limit = .2 by default, size= 8 by default)
        ...clahe : to equalize the image with clahe algorithm, local
                equalization, 2 parameters:
                    limit = .2 by default,
                    size= 8 by default
        """
        if treatment == 'crop':
            if len(args) == 4:
                self.current_image.crop(*args)
            else:
                raise StackProcessingError(
                    'Crop treatment requires 4 arguments'
                )
        elif treatment == 'clahe':
            if len(args) == 2 or len(args) == 0:
                self.current_image.equalize_hist_by_clahe(*args)
            else:
                raise StackProcessingError(
                    'Equalize treatment require 0 or 2 arguments'
                )
        elif treatment == 'equalize':
            if len(args) == 1 or len(args) == 0:
                self.current_image.equalize_hist(*args)
            else:
                raise StackProcessingError(
                    'Equalize treatment require 0 or 1 argument'
                )
        else:
            raise StackProcessingError(
                'The treatment you are trying to apply does not exist'
            )

        if plot:
            self.current_image.show_image()

    def remove_treatments(self):
        """Remove all lastOp files."""
        files = glob.glob((
            self.exporting_directory + 'lastOp/*.npy'
        ))
        for i in files:
            os.remove(i)
        self.update_lists()
        self.read_image()

    def save_treatment(self, treatment, *args):
        """
        Treat all stakck image and save them in the results directory.

        Arguments are needed. There are the same than for treament, because
        save_treatment call recursivly treament.
        """
        # === initiate progressbar === #
        widgets = [treatment,
                   ' ', progressbar.Percentage(),
                   ' ', progressbar.Bar('=', '[', ']'),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(self.image_list)
        )
        pbar.start()
        # === loop over images === #
        for i in range(self.n_image_tot):
            self.read_image(i)
            self.treatment(treatment, *args, plot=False)
            if i < 9:
                np.save(
                    self.exporting_directory + 'lastOp/' + 'i_000' + str(i+1),
                    self.current_image.image
                )
            elif i < 99:
                np.save(
                    self.exporting_directory + 'lastOp/' + 'i_00' + str(i+1),
                    self.current_image.image
                )
            elif i < 999:
                np.save(
                    self.exporting_directory + 'lastOp/' + 'i_0' + str(i+1),
                    self.current_image.image
                )
            pbar.update(i)
        pbar.finish()

        # === update images infos === #
        self.update_lists()
        self.read_image()

    def set_time_reference(self, time_error=2, plot=False):
        """
        Define start point for time and for base line.

        Need contour to work.

        :param time_error:  percentage of error accepted on time
        """
        if self.current_spatioy_number == -1:
            print('Impossible to set the reference before having create\
                spatios')

        temp = self.current_spatioy_number

        # === read the center spatio === #
        ref_x = np.load(self.contour_directory + 'ref_x.npy')
        xc = int(np.mean(ref_x))  # index of the center of drop
        self.read_spatio(xc)

        # === define secure percentage === #
        t_add = np.array(
            np.rint(self.current_spatioy.size[0]*time_error/100),
            dtype=int
        )

        # === get point === #
        fig, ax = plt.subplots(
            1, 1,
            figsize=(11, 7),
        )
        fig.canvas.set_window_title('Set reference time')
        ax.imshow(self.current_spatioy.image, cmap='gray')
        ax.autoscale(False)
        ax.set_title(
            'Please select a point on the start solidification front'
        )

        pts = []
        while len(pts) < 1:
            pts = plt.ginput(1, timeout=-1)
        pts = np.array(pts, dtype=int)

        # == plot point === #
        if plot:
            ax.plot(
                [pts[0][0], pts[0][0]],
                [0, self.current_spatioy.size[0]],
                color='red'
            )
            ax.plot(
                [
                    pts[0][0],
                    pts[0][0]
                ]-t_add,
                [0, self.current_spatioy.size[0]],
                color='green'
            )
            plt.show()
        else:
            plt.close(fig)

        # === return value === #
        time_ref = pts[0][0] - t_add
        self.read_spatio(temp)

        return time_ref

    def get_baseline(self, plot=False):
        """Determine the positon of the baseline.

        Need spatioy to work and contour
        """
        temp = self.current_spatioy_number  # stock the current spatio

        # === x positions where there is no drop === #
        ref_x = np.load(self.contour_directory + 'ref_x.npy')
        x_left = np.arange(0, ref_x[0])
        x_right = np.arange(ref_x[-1], self.current_image.size[1])
        x = np.append(x_left, x_right)
        # x = x_left

        # === loop over all x to get the position === #
        ps = []  # preallocate the baseline position
        pbar = progressbar.ProgressBar(
            maxval=len(x),
            widgets=['Evaluating baseline: ', progressbar.AnimatedMarker()]
        )
        pbar.start()
        for i, i_im in enumerate(x):
            self.read_spatio(i_im)
            self.current_spatioy.image = self.current_spatioy.gradient(type='sobel', size=5, out='y')
            p = []
            for i_s in range(self.current_spatioy.size[1] - 4):
                # "-4" because of the gradient
                _, it = stack.current_spatioy.col_intensity(i_s)
                start = np.argmin(it)
                it = it[start:]
                p = np.append(p, start+np.argmax(it))
            pbar.update(i)
            ps = np.append(ps, np.mean(p))
        pbar.finish()

        # === baseline = mean(ps) === #
        baseline = int(np.mean(ps))

        # === display current image with baseline === #
        if plot:
            print('Baseline position : ', baseline)
            plt.figure()
            plt.imshow(self.current_image.image, cmap='gray')
            plt.plot(
                x,
                self.baseline*np.ones_like(x),
                '.r'
            )
            plt.show()

        self.read_spatio(temp)  # reload the current spatio

        return baseline

    def get_baseline2(self, plot=False):
        """Determine the positon of the baseline.

        Detect contour before please.
        """
        ni = self.n_image_tot
        if ni < 9:
            Cx = np.load(
                self.contour_directory + '000' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '000' +
                str(ni) + '_Cy' +
                '.npy'
            )
        elif ni < 99:
            Cx = np.load(
                self.contour_directory + '00' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '00' +
                str(ni) + '_Cy' +
                '.npy'
            )
        elif ni < 999:
            Cx = np.load(
                self.contour_directory + '0' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '0' +
                str(ni) + '_Cy' +
                '.npy'
            )
        x_mean_left = Cx[np.argmin(Cy)]
        x_mean_right = Cx[::-1][np.argmin(Cy[::-1])]
        x_mean = np.int((x_mean_left + x_mean_right) / 2)
        Cx_left = Cx[Cx <= x_mean]
        Cx_right = Cx[Cx > x_mean]
        Cy_left = Cy[Cx <= x_mean]
        Cy_right = Cy[Cx > x_mean]

        y_base_left = np.max(Cy_left)
        y_base_right = np.max(Cy_right)
        x_base_left = Cx_left[np.argmax(Cy_left)]
        x_base_right = Cx_right[np.argmax(Cy_right)]

        pente = (y_base_right - y_base_left) / (x_base_right - x_base_left)
        y0 = y_base_left - pente*x_base_left

        # === display current image with baseline === #
        if plot:
            print('Inclination : ', pente*180/np.pi)
            plt.figure()
            temp = self.current_image_number
            self.read_image(ni-1)
            plt.imshow(self.current_image.image, cmap='gray')
            self.read_image(temp)
            plt.plot(Cx_left, Cy_left, '.b', markersize=1)
            plt.plot(Cx_right, Cy_right, '.g', markersize=1)
            plt.plot(
                np.arange(self.current_image.size[1]),
                pente*np.arange(self.current_image.size[1])+y0,
                '--r',
            )
            plt.show()

        return [pente*np.arange(self.current_image.size[1])+y0, pente, y0],\
            [np.int(x_base_left), np.int(x_base_right)],

    def set_contour_ref(self, upper=25, lower=10):
        """
        Create 3 arrays to pre-locate the front.

        Thanks to three points definition on the first image, and the fitting
        of a circle equation, define an area in the wich the front must be
        located
        """
        temp = self.current_image_number
        self.read_image(0)

        fig, ax = plt.subplots(
            1, 1,
            figsize=(10, 8),
        )
        fig.canvas.set_window_title('Image visualisation')
        fig.suptitle('Pick three points please')
        ax.imshow(self.current_image.image, cmap='gray')
        ax.autoscale(False)

        pts = []
        while len(pts) < 3:
            pts = plt.ginput(3, timeout=-1, show_clicks=True)
        pts = np.array(pts, dtype=int)

        # solve system for circle
        x = np.complex(pts[0][0], pts[0][1])
        y = np.complex(pts[1][0], pts[1][1])
        z = np.complex(pts[2][0], pts[2][1])
        w = z-x
        w /= y-x
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
        a = abs(c.real)
        b = abs(c.imag)
        r = abs(c+x)
        ref_x = []
        ref_ymin = []
        ref_ymax = []
        left_boundary = int(
            np.min([pts[0][0], pts[1][0], pts[2][0]]) -
            (
                np.max([pts[0][0], pts[1][0], pts[2][0]]) -
                np.min([pts[0][0], pts[1][0], pts[2][0]])
            )*.01
        )
        if left_boundary < 0:
            left_boundary = 0

        right_boundary = int(
            np.max([pts[0][0], pts[1][0], pts[2][0]]) +
            (
                np.max([pts[0][0], pts[1][0], pts[2][0]]) -
                np.min([pts[0][0], pts[1][0], pts[2][0]])
            )*.05
        )
        if right_boundary > self.current_image.size[1]:
            right_boundary = self.current_image.size[1]

        for k in np.arange(left_boundary, right_boundary):
            if abs(k-a) < r:
                fmoins = -np.sqrt(r**2 - (k-a)**2) + b
                ref_x.append(k)
                ref_ymin.append(
                    fmoins + lower*(
                        np.max([pts[0][1], pts[1][1], pts[2][1]]) -
                        np.min([pts[0][1], pts[1][1], pts[2][1]])
                    )/100
                )
                ref_ymax.append(
                    fmoins -
                    1.5 * upper * (
                        np.max([pts[0][1], pts[1][1], pts[2][1]]) -
                        np.min([pts[0][1], pts[1][1], pts[2][1]])
                    )/100 +
                    .5 * upper * (
                        np.max([pts[0][1], pts[1][1], pts[2][1]]) -
                        np.min([pts[0][1], pts[1][1], pts[2][1]])
                    )/100 *
                    (k - (right_boundary+left_boundary)/2)**2 /
                    (left_boundary - (right_boundary+left_boundary)/2)**2
                )
                # ref_ymax = yfit(==fmoins) - 1.5*p + .5*p*(x-xc)^2/(xmin-xc)^2
                # on créé un grossissement au niveau de la pointe pour prendre
                # en compte la dilatation
        ref_x = np.array(ref_x, dtype=int)
        ref_ymin = np.array(ref_ymin, dtype=int)
        ref_ymax = np.array(ref_ymax, dtype=int)

        ref_ymin[
            ref_ymin > self.current_image.size[1]
        ] = self.current_image.size[1]
        ref_ymax[
            ref_ymax > self.current_image.size[1]
        ] = self.current_image.size[1]
        ref_ymin[ref_ymin < 0] = 0
        ref_ymax[ref_ymax < 0] = 0

        ax.plot(
            ref_x, ref_ymin,
            '.g'
        )
        ax.plot(
            ref_x, ref_ymax,
            '.b'
        )
        plt.show()
        self.read_image(temp)
        return ref_x, ref_ymin, ref_ymax

    def get_contour(self, fac=1, plot=False):
        """
        Determine the contour of the current images.

        Please provide x, ymin and ymax from set_contour_ref().
        """
        self.current_image.image = self.current_image.gradient(type='sobel', size=5)

        thres = np.mean(self.current_image.image) +\
            fac*np.std(self.current_image.image)
        self.current_image.image[self.current_image.image < thres] = 0
        self.current_image.image[self.current_image.image > 0] = 1

        im_er = morphology.binary_erosion(
            np.array(self.current_image.image, dtype=bool),
            morphology.square(2)
        )

        im_rm = morphology.remove_small_objects(
            im_er, 6
        )

        im_dilate = morphology.binary_dilation(
            im_rm,
            morphology.square(2)
        )

        Cx = Cy = np.array([])
        for i in range(0, self.current_image.size[1]-4, 1):
            it = im_dilate[:, i]
            if np.argmax(it) > 0:
                Cx = np.append(Cx, i)
                Cy = np.append(Cy, np.argmax(it))

        if plot:
            fig, ax = plt.subplots(figsize=(11, 7))
            fig.canvas.set_window_title('Contour visualisation')

            self.read_image(self.current_image_number)
            ax.imshow(self.current_image.image, cmap='gray')
            ax.plot(
                Cx,
                Cy,
                '.g', markersize='3',
                label='Contour2 detected'
            )

            plt.legend(fancybox=True, shadow=True)
            plt.grid(True)
            plt.show()
        return Cx, Cy

    def get_contours(self):
        """
        Determine contour for all images.

        Full automatic, no parameters are needed.
        """
        temp = self.current_image_number
        widgets = ['Detecting contour',
                   ' ', progressbar.Percentage(),
                   ' ', progressbar.Bar('=', '[', ']'),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(self.image_list)
        )
        pbar.start()

        for i in range(len(self.image_list)):
            self.read_image(i)
            Cx, Cy = self.get_contour()
            if i < 9:
                np.save(
                    self.contour_directory + '/000' + str(i+1) + '_Cx',
                    Cx
                )
                np.save(
                    self.contour_directory + '/000' + str(i+1) + '_Cy',
                    Cy
                )
            elif i < 99:
                np.save(
                    self.contour_directory + '/00' + str(i+1) + '_Cx',
                    Cx
                )
                np.save(
                    self.contour_directory + '/00' + str(i+1) + '_Cy',
                    Cy
                )
            elif i < 999:
                np.save(
                    self.contour_directory + '/0' + str(i+1) + '_Cx',
                    Cx
                )
                np.save(
                    self.contour_directory + '/0' + str(i+1) + '_Cy',
                    Cy
                )
            pbar.update(i)
        pbar.finish()
        self.read_image(temp)

    def display_contour(self, ni=-1, ref=False):
        """
        Display image with the contour.

        If ni = -1, displays the current_image, else displays image ni
        """
        if ni > -1:
            self.read_image(ni)

        if self.current_image_number < 9:
            Cx = np.load(
                self.contour_directory + '000' +
                str(self.current_image_number+1) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '000' +
                str(self.current_image_number+1) + '_Cy' +
                '.npy'
            )
        elif self.current_image_number < 99:
            Cx = np.load(
                self.contour_directory + '00' +
                str(self.current_image_number+1) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '00' +
                str(self.current_image_number+1) + '_Cy' +
                '.npy'
            )
        elif self.current_image_number < 999:
            Cx = np.load(
                self.contour_directory + '0' +
                str(self.current_image_number+1) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '0' +
                str(self.current_image_number+1) + '_Cy' +
                '.npy'
            )

        fig = plt.figure(figsize=(11, 7))
        fig.canvas.set_window_title(
            'Image with contour n°' + str(self.current_image_number)
        )
        plt.imshow(self.current_image.image, cmap='gray')
        plt.plot(
            Cx, Cy,
            '.', markersize=5, color='#BB0029',
            label='Contour'
        )
        plt.legend(shadow=True, fancybox=True, loc='best')
        plt.show()

    def display_sol(self, n, ratio, fps):
        t = np.load(
            self.front_directory + 'x_' + str(n) + '_time.npy'
        )
        front = np.load(
            self.front_directory + 'x_' + str(n) + '_height.npy'
        )
        ref_x = np.load(
            self.contour_directory + 'ref_x.npy'
        )
        contour = []
        for i_t in np.arange(0, self.current_spatioy.size[1]-1):
            if i_t < 9:
                y = np.load(
                    self.contour_directory + 'y_000' +
                    str(i_t+1) +
                    '.npy'
                )
            elif i_t < 99:
                y = np.load(
                    self.contour_directory + 'y_00' +
                    str(i_t+1) +
                    '.npy'
                )
            elif i_t < 999:
                y = np.load(
                    self.contour_directory + 'y_0' +
                    str(i_t+1) +
                    '.npy'
                )
            idx = ref_x == n
            contour.append(y[idx])
        contour = np.array(contour, dtype=int)

        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        fig.canvas.set_window_title('Front et contour')

        self.read_image(50)
        ax[0, 0].imshow(self.current_image.image, cmap='gray')
        ax[0, 0].plot(
            [n, n], [0, self.current_image.size[0]-1],
            '--k', linewidth=1
        )

        self.read_image(250)
        ax[0, 1].imshow(self.current_image.image, cmap='gray')
        ax[0, 1].plot(
            [n, n], [0, self.current_image.size[0]-1],
            '--k', linewidth=1
        )

        self.read_image(329)
        ax[0, 2].imshow(self.current_image.image, cmap='gray')
        ax[0, 2].plot(
            [n, n], [0, self.current_image.size[0]-1],
            '--k', linewidth=1
        )

        self.read_spatio(n)
        ax[1, 0].imshow(self.current_spatioy.image, cmap='gray')
        ax[1, 0].plot(
            [50, 50],
            [0, self.current_spatioy.size[0]-1],
            '--k', linewidth=1
        )
        ax[1, 0].plot(
            [250, 250],
            [0, self.current_spatioy.size[0]-1],
            '--k', linewidth=1
        )
        ax[1, 0].plot(
            [329, 329],
            [0, self.current_spatioy.size[0]-1],
            '--k', linewidth=1
        )
        ax[1, 0].plot(
            np.arange(0, self.current_spatioy.size[1]-1), contour,
            'xg', markersize=2
        )
        ax[1, 0].plot(
            t, front, 'xr', markersize=2
        )

        contour = self.current_spatioy.size[0] - contour
        front = self.current_spatioy.size[0] - front

        ax[1, 1].plot(
            t, front, '.b'
        )
        ax[1, 1].plot(
            np.arange(0, self.current_spatioy.size[1]-1),
            contour,
            '.g'
        )

        t_a = np.arange(0, max(t)-min(t), 1)/fps
        front = (front-72)*ratio
        ax[1, 2].plot(
            (t-178)/fps, front,
            '.b', markersize=2
        )

        analytical = ste.Stefan(t_a)
        analytical.set_wall_temperature(-7)
        analytical.monophasic_steady()
        ax[1, 2].plot(
            t_a, analytical.s,
            '--k', linewidth=1
        )

        analytical.monophasic_unsteady()
        ax[1, 2].plot(
            t_a, analytical.s,
            '--r', linewidth=1
        )

        analytical.monophasic_lateral_flux(0.01, 65*np.pi/180, 2.46e-3)
        ax[1, 2].plot(
            t_a, analytical.s,
            '--g', linewidth=1
        )
        ax[1, 2].grid(True)

        plt.show()

    def get_time_front(self, rows, time_ref, option={}, plot=True):
        """
        Determine the fronts postions in current_spatioy for the indicated row.

        :parm row: at which rows to detect the front. Can be a list or a int.
        :param time_ref: the reference time to detect the front.
        :parm plot: True to display results, False neither. If row is a
        vector, this is automatically false.

        :param option:
            You can define different option.
            front_thickness : the thickness of the front.

        ..todo:
        """
        def normalized(x, minor=0, major=1):
            x = np.array(x, dtype=float)
            x = (
                (
                    (major-minor)*x + minor*x.max() - major*x.min()
                ) /
                (x.max() - x.min())
            )
            return x

        try:
            ft = option['front_thickness']
        except KeyError:
            ft = 8
        try:
            mpn = option['max_peaks_number']
        except KeyError:
            mpn = 8

        if not rows:
            raise StackProcessingError('Need a list a row')
        if not time_ref:
            raise StackProcessingError('Need a time reference')

        if type(rows) != int:
            if len(rows) > 1:
                plot = False
            else:
                rows = rows[0]

        if plot:
            # === create image === #
            fig, ax = plt.subplots(
                2, 2,
                figsize=(11, 7)
            )
            fig.canvas.set_window_title('Row detection'+str(rows))

            # === display spatio === #
            ax[0, 0].imshow(self.current_spatioy.image, cmap='gray')
            ax[0, 0].plot(
                [0, self.current_spatioy.size[1]-1],
                [rows, rows],
                '--k', linewidth=1
            )

            # === intensity current_spatio === #
            it = self.current_spatioy.image
            it = it[rows, :]
            it = it[time_ref:]
            it = normalized(it - np.mean(it), -1, 1)
            ax[1, 0].plot(
                it,
                '-k', linewidth=1,
                label='Spatio intensity'
            )

            # ===  intensity sobel (5x5) of current_spatio === #
            self.current_spatioy.image = self.current_spatioy.gradient(type='sobel', size=5, out='mag')
            it_sbl = self.current_spatioy.image
            it_sbl = it_sbl[rows, :]
            it_sbl = it_sbl[time_ref:]
            it_sbl = normalized(it_sbl - np.mean(it_sbl), -1, 1)
            ax[1, 0].plot(
                it_sbl,
                '-g', linewidth=1,
                label='Sobel magnitude intensity'
            )

            # === intensity sobel /y (5x5) of current_images === #
            self.read_spatio()
            self.current_spatioy.image = self.current_spatioy.gradient(type='sobel', size=5, out='y')
            # apply sobel gradient
            it_sby = 1 - self.current_spatioy.image
            it_sby = it_sby[rows, :]
            it_sby = it_sby[time_ref:]
            it_sby = normalized(it_sby - np.mean(it_sby), -1, 1)
            ax[1, 0].plot(
                it_sby,
                '-b', linewidth=1,
                label=r'sobel intensity $\overrightarrow{e_y}$'
            )

        # ==== intensity sobel /t (5x5) of current_images === #
        self.read_spatio()  # read current spatio /y
        self.current_spatioy.image = self.current_spatioy.gradient(type='sobel', size=5, out='mag')
        # apply gradient 2D by sobel
        it_sbt = self.current_spatioy.image  # inverse the image

        if type(rows) == int:
            rows = [rows]
        front = np.empty([0, ], dtype=int)
        start_solidification = np.empty([0, ], dtype=int)
        front_new = np.empty([0, ], dtype=int)
        start_solidification_new = np.empty([0, ], dtype=int)

        front_new2_left = np.empty([0, ], dtype=int)
        front_new2_right = np.empty([0, ], dtype=int)
        start_solidification_new2 = np.empty([0, ], dtype=int)

        for row in rows:
            it_sbt = it_sbt[row, :]  # get the row intensity
            it_sbt = it_sbt[time_ref:]
            ita_sbt = np.convolve(it_sbt, np.ones([5, ])/5, 'same')
            ita_sbt[:2] = it_sbt[:2]
            ita_sbt[-2:] = it_sbt[-2:]
            # only keep information in interesting domain
            it_sbt = normalized(it_sbt - np.mean(it_sbt), -1, 1)
            # normalized information
            if plot:
                ax[1, 0].plot(
                    it_sbt,
                    '-r', linewidth=1,
                    label=r'sobel intensity $\overrightarrow{e_t}$'
                )

            # === get peaks === #
            fac = 1
            peaks = pf.PeakDetection(
                it_sbt - np.mean(it_sbt),
                fac*np.std(it_sbt)
            )

            conditon = len(peaks.max_location) + len(peaks.min_location) > mpn
            counter = 0
            # if there is two much peaks, we increase the selectivy
            while conditon and counter < 100:
                fac += .05  # increase by 5% the selectivity
                peaks = pf.PeakDetection(
                    it_sbt - np.mean(it_sbt),
                    fac*np.std(it_sbt)
                )
                conditon = len(peaks.max_location) + \
                    len(peaks.min_location) > mpn
                counter += 1

            peaks.min_location = np.array(peaks.min_location, dtype=int)
            peaks.max_location = np.array(peaks.max_location, dtype=int)

            t_3percents = np.array(
                np.rint(self.current_spatioy.size[0]*3/100),
                dtype=int
            )

            if plot:
                ax[0, 1].plot(
                    it_sbt,
                    '-k', linewidth=1,
                    label=r'sobel intensity $\overrightarrow{e_t}$'
                )
                ax[0, 1].plot(
                    peaks.max_location, peaks.max_magnitude,
                    '.b', markersize=3,
                    label='Maximum peaks'
                )
                ax[0, 1].plot(
                    peaks.min_location, peaks.min_magnitude,
                    '.r', markersize=3,
                    label='Minimum peaks'
                )
                ax[0, 1].plot(
                    [t_3percents, t_3percents],
                    [-1, 1],
                    '--k'
                )
                # axe des deviations standard
                ax[0, 1].plot(
                    [0, len(it_sbt)-1],
                    [np.mean(it_sbt)+np.std(it_sbt),
                    np.mean(it_sbt)+np.std(it_sbt)],
                    '--k'
                )
                ax[0, 1].plot(
                    [0, len(it_sbt)-1],
                    [np.mean(it_sbt)-np.std(it_sbt),
                    np.mean(it_sbt)-np.std(it_sbt)],
                    '--k'
                )


            # === get front and start_solidification times === #
            # t_3percents = np.array(
            #     np.rint(self.current_spatioy.size[0]*3/100),
            #     dtype=int
            # )  # 3% of the time
            """
            Si le premier pic trouvé se trouve dans les 3premiers
            pourcent, le front se situe au pic qui a la plus grosse
            intensité, sinon c'est le dernier pic détecter
            """
            ######## actual method ########
            if len(peaks.max_location) > 0:
                if peaks.max_location[0] > t_3percents:
                    # le premier pic est pas dans les 3 premiers %
                    start_solidification = np.append(
                        start_solidification, 0
                    )
                    if len(peaks.min_location) > 0:
                        front = np.append(front, max(
                            peaks.max_location[np.argmax(
                                peaks.max_magnitude
                            )],
                            peaks.min_location[np.argmin(
                                    peaks.min_magnitude
                                )]
                        ))
                    else:
                        front = np.append(
                            front,
                            peaks.max_location[np.argmax(
                                peaks.max_magnitude
                            )]
                        )
                else:
                    # le premier pic n'est pas dans les 3 premiers %
                    start_solidification = np.append(
                        start_solidification,
                        peaks.max_location[0]
                    )
                    if len(peaks.min_location) > 0:
                        front = np.append(front, max(
                            peaks.max_location[-1],
                            peaks.min_location[-1]
                        ))
                    else:
                        front = np.append(front, peaks.max_location[-1])
            else:
                front = np.append(front, 0)
                start_solidification = np.append(
                    start_solidification,
                    0
                )


            ######## new method test ########
            # === sorting peaks === #
            max_location = np.array(peaks.max_location)
            max_magnitude = np.array(peaks.max_magnitude)
            max_location = max_location[
                np.argsort(
                    max_magnitude
                )
            ][::-1]
            max_magnitude = np.sort(
                max_magnitude
            )[::-1]

            # === find front and start === #
            if max_location[0] < t_3percents:
                if len(max_location) > 1:
                    condition = len(peaks.min_location) > 0 and (
                        np.argmin(peaks.min_location) <
                        max_location[1] + 10 and
                        np.argmin(peaks.min_location) >
                        max_location[1] - 10
                    )
                    if condition:
                        front_new = np.append(
                            front_new,
                            np.argmin(peaks.min_location)
                        )
                    else:
                        front_new = np.append(
                            front_new, max_location[1]
                        )
                else:
                    front_new = np.append(
                        front_new, max_location[0]
                    )
                start_solidification_new = np.append(
                    start_solidification_new,
                    max_location[0]
                )
            else:
                if len(max_location) > 1:
                    condition = len(peaks.min_location) > 0 and (
                        np.argmin(peaks.min_location) <
                        max_location[1] + 10 and
                        np.argmin(peaks.min_location) >
                        max_location[1] - 10
                    )
                    if condition:
                        front_new = np.append(
                            front_new, np.argmin(peaks.min_location)
                        )
                    else:
                        front_new = np.append(
                            front_new, max_location[0]
                        )
                else:
                    front_new = np.append(
                        front_new, max_location[0]
                    )
                start_solidification_new = np.append(
                    start_solidification_new, 0
                )

            ### third method ###
            if len(peaks.max_location) == 1:
                if peaks.max_location[0] < t_3percents:
                    # le front et le début du givrage sont confondus.
                    start_solidification_new2 = np.append(
                        start_solidification_new2,
                        peaks.max_location[0]
                    )
                    front_new2_left = np.append(
                        front_new2_left,
                        peaks.max_location[0]
                    )
                    front_new2_right = np.append(
                        front_new2_right,
                        peaks.max_location[0]
                    )
                else:
                    # on ne voit que le front (on est surement au dessus du
                    # contour initiale de la goutte)
                    if len(peaks.min_location) > 1:
                        min_pos = peaks.min_location[
                            peaks.min_location < peaks.max_location[0]+ft
                        ]
                        min_pos = min_pos[
                            min_pos > peaks.max_location[0]-ft
                        ]
                        if len(min_pos) > 1:
                            min_pos = min_pos[
                                np.argmin(
                                    abs(min_pos - peaks.max_location[0])
                                )
                            ]
                            front_new2_left = np.append(
                                front_new2_left,
                                min(
                                    min_pos, peaks.max_location[0]
                                )
                            )
                            front_new2_right = np.append(
                                front_new2_right,
                                max(
                                    min_pos, peaks.max_location[0]
                                )
                            )
                        else:
                            front_new2_left = np.append(
                                front_new2_left,
                                peaks.max_location[0]
                            )
                            front_new2_right = np.append(
                                front_new2_right,
                                peaks.max_location[0]
                            )
                    else:
                        front_new2_left = np.append(
                            front_new2_left,
                            peaks.max_location[0]
                        )
                        front_new2_right = np.append(
                            front_new2_right,
                            peaks.max_location[0]
                        )
            # = 2 pics de déteté = #
            elif len(peaks.max_location) == 2:
                # il y 2 pics, un est le front l'autre le début de la sol.
                if peaks.max_location[0] < t_3percents:
                    # si le premier peaks est dans la bande à 3% il s'agit du
                    # début de la solidification. Le second est le front.
                    start_solidification_new2 = peaks.max_location[0]
                    if len(peaks.min_location) > 1:
                        min_pos = peaks.min_location[
                            peaks.min_location < peaks.max_location[1]+ft
                        ]
                        min_pos = min_pos[
                            min_pos > peaks.max_location[1]-ft
                        ]
                        if len(min_pos) > 1:
                            min_pos = min_pos[
                                np.argmin(
                                    abs(min_pos - peaks.max_location[1])
                                )
                            ]
                            front_new2_left = np.append(
                                front_new2_left,
                                min(
                                    min_pos, peaks.max_location[1]
                                )
                            )
                            front_new2_right = np.append(
                                front_new2_right,
                                max(
                                    min_pos, peaks.max_location[1]
                                )
                            )
                        else:
                            front_new2_left = np.append(
                                front_new2_left,
                                peaks.max_location[1]
                            )
                            front_new2_right = np.append(
                                front_new2_right,
                                peaks.max_location[1]
                            )
                    else:
                        front_new2_left = np.append(
                            front_new2_left,
                            peaks.max_location[1]
                        )
                        front_new2_right = np.append(
                            front_new2_right,
                            peaks.max_location[1]
                        )
                else:
                    pass

            if plot:
                # plot on ax[1, 1] : intensity, front and start_solidification
                ax[1, 1].plot(
                    it_sbt - np.mean(it_sbt),
                    '-k', linewidth=1,
                    label=r'sobel intensity $\overrightarrow{e_t}$'
                )
                ax[1, 1].plot(
                    front, it_sbt[front] - np.mean(it_sbt),
                    '.b', markersize=5,
                    label='Front'
                )
                ax[1, 1].plot(
                    start_solidification,
                    it_sbt[start_solidification] - np.mean(it_sbt),
                    '.r', markersize=3,
                    label='Start solidification'
                )
                ax[1, 1].plot(
                    [t_3percents, t_3percents],
                    [-1, 1],
                    '--k'
                )

                ### add new method in green (front) and magenta (start) ###
                ax[1, 1].plot(
                    front_new, it_sbt[front_new] - np.mean(it_sbt),
                    '.g', markersize=5,
                    label='Front new'
                )
                ax[1, 1].plot(
                    start_solidification_new,
                    it_sbt[start_solidification_new] - np.mean(it_sbt),
                    '.m', markersize=3,
                    label='Start solidification new'
                )
                ### end ###

                ### add third method ###
                ax[1, 1].plot(
                    front_new2_left,
                    it_sbt[front_new2_left] - np.mean(it_sbt),
                    '*b', markersize=7,
                    label='Front new 2'
                )
                ax[1, 1].plot(
                    front_new2_right,
                    it_sbt[front_new2_right] - np.mean(it_sbt),
                    '*r', markersize=8,
                    label='Front new 2'
                )
                ax[1, 1].plot(
                    start_solidification_new2,
                    it_sbt[start_solidification_new2] - np.mean(it_sbt),
                    'sy', markersize=3,
                    label='Front new 2'
                )
                ### end third method ###

                # plot on ax[0, 0] (spatio) front and start_solidification
                ax[0, 0].plot(
                    time_ref + front + 1, row,
                    '.b', markersize=5,
                    label='Front'
                )
                ax[0, 0].plot(
                    time_ref + start_solidification + 1,
                    row,
                    '.r', markersize=3,
                    label='Start solidification'
                )
                ### add new method for front(green) and start(magenta) ###
                ax[0, 0].plot(
                    time_ref + front_new + 1, row,
                    '.g', markersize=5,
                    label='Front new'
                )
                ax[0, 0].plot(
                    time_ref + start_solidification_new + 1,
                    row,
                    '.m', markersize=3,
                    label='Start solidification new'
                )
                ### end ###

                # add reading facilites
                ax[0, 0].legend(
                    shadow=True, fancybox=True, loc='best'
                )
                ax[1, 0].grid(True)
                ax[1, 0].legend(
                    shadow=True, fancybox=True, loc='best'
                )
                ax[0, 1].grid(True)
                ax[0, 1].legend(
                    shadow=True, fancybox=True, loc='best'
                )
                ax[1, 1].grid(True)
                ax[1, 1].legend(
                    shadow=True, fancybox=True, loc='best'
                )
                # plt.show()

        # === retunr values === #
        self.read_spatio()  # reinitialisate the spatio
        front += time_ref + 1
        start_solidification += time_ref + 1
        return front, start_solidification

    def get_time_front_all(self, baseline, time_ref, plot=False):
        """Detect front & start for one spatioy."""
        # === check input === #
        if not baseline:
            raise StackProcessingError('Need a baseline')
        if not time_ref:
            raise StackProcessingError('Need time reference')

        # === get the contour for current_spatioy === #
        contour = []
        for i_t in np.arange(1, self.current_spatioy.size[1]):
            ref_x = np.load(
                self.contour_directory + 'ref_x.npy')
            if i_t < 10:
                y = np.load(
                    self.contour_directory + 'y_000' +
                    str(i_t) +
                    '.npy'
                )
            elif i_t < 100:
                y = np.load(
                    self.contour_directory + 'y_00' +
                    str(i_t) +
                    '.npy'
                )
            elif i_t < 1000:
                y = np.load(
                    self.contour_directory + 'y_0' +
                    str(i_t) +
                    '.npy'
                )
            idx = ref_x == self.current_spatioy_number
            contour.append(y[idx])
        contour = np.array(contour, dtype=int)

        cols = np.arange(np.min(contour), baseline, 1)
        front, start = self.get_time_front(cols, time_ref, False)

        mat = np.array([1, 1, 1, 1, 1]) / 5

        front2 = np.convolve(front, mat, 'same')
        front2[:2] = front[:2]
        front2[-2:] = front[-2:]

        start0 = int(np.mean(start[start > contour.max()]))
        t_front = np.arange(start0, self.current_spatioy.size[1])
        y_front = []

        # p = np.polyfit(np.sqrt(front - start0), self.baseline - cols, 1)
        # y_fit = p[1]*np.sqrt(t_front - start0) + p[0]
        # y_fit = self.baseline - y_fit

        for i, t in enumerate(t_front):
            potential_front = cols[front == t]
            if len(potential_front) == 0:
                y_front = np.append(y_front, np.nan)
            else:
                y_front = np.append(
                    y_front, np.mean(potential_front)
                )
        """
        if np.isnan(y_front[0]):
            print('0 mod')
            y_front[0] = self.baseline
        i = -1
        while np.isnan(y_front[i]):
            print(str(i) +'mod')
            y_front[i] = cols.min()
            i -= 1
        for i in range(1, len(t_front)-1):
            if np.isnan(y_front[i]):
                j = i+1
                while np.isnan(y_front[j]):
                    j+=1
                r = np.arange(i, j)
                for k in r:
                y_front[i] = (y_front[i-1]+y_front[i+1]) / 2
        """

        plt.figure()
        plt.plot(t_front, y_front, '--b')
        plt.plot(front, cols, '--r')

        if plot:
            fig, ax = plt.subplots(figsize=(11, 7))
            fig.canvas.set_window_title('Front & Start')

            ax.imshow(self.current_spatioy.image, cmap='gray')
            ax.plot(front, cols, '.r', markersize=6, label='Front')
            ax.plot(t_front, y_front, '.b', markersize=4, label='Front')
            """
            ax.plot(
                start, cols, '.b', markersize=6,
                label='Star solidification'
            )
            """
            ax.plot(
                [0, self.current_spatioy.size[1]-1],
                [self.baseline, self.baseline],
                '--k', linewidth=4
            )
            start = start[start > contour.max()]
            ax.plot(
                [np.mean(start), np.mean(start)],
                [0, self.current_spatioy.size[0]-1],
                '--k', linewidth=4
            )
            ax.plot(
                np.arange(0, self.current_spatioy.size[1]-1), contour,
                '.g', markersize=6, label='Contour'
            )
            plt.legend(
                fancybox=True, shadow=True,
                loc='upper left',
                fontsize=15
            )
            plt.xlabel('Time (frame)', fontsize=20)
            plt.ylabel('y (px)', fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=15)

            fig1, ax1 = plt.subplots(figsize=(11, 7))
            fig1.canvas.set_window_title('Front vs stefan')

            ratio = 9e-3/1197
            fps = 4.0

            t_ana = np.arange(
                0, self.current_spatioy.size[1]-int(np.mean(start))
            )/fps
            analytical = ste.Stefan(t_ana)
            analytical.set_wall_temperature(-7)
            analytical.monophasic_steady()
            ax1.plot(
                t_ana, analytical.front_position,
                '--k', linewidth=4,
                label='Analytical steady'
            )

            ax1.plot(
                (front-np.mean(start))/fps,
                (self.baseline-cols)*ratio,
                '.r', markersize=8,
                label='Front measured'
            )

            plt.xlabel('Time (s)', fontsize=20)
            plt.ylabel('Position (m)', fontsize=20)
            plt.legend(fancybox=True, shadow=True, fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.grid(True)
        plt.show()

        return t_front, y_front

    def get_front(self, row, time_ref, smooth=3, fac=.75, plot=False):
        """
        Determine the fronts postions in current_spatioy for the indicated row.

        :param row: at which rows to detect the front. Can be a list or a int.
        :param time_ref:

        """
        self.read_spatio()  # read current spatio /y
        grad = self.current_spatioy.gradient(type='sobel', size=5, out='mag', border='valid')
        it_sbt = self.current_spatioy.image
        it_sbt[2:-2, 2:-2] = grad

        # flux at north, southe east & west
        it_sbt[:2, 2:-2] = it_sbt[2, 2:-2]
        it_sbt[-2:, 2:-2] = it_sbt[-3, 2:-2]
        it_sbt[2:-2, :2] = np.transpose(np.tile(it_sbt[2:-2, 2], (2, 1)))
        it_sbt[2:-2, -2:] = np.transpose(np.tile(it_sbt[2:-2, -3], (2, 1)))

        # top left corner
        it_sbt[1, 0] = it_sbt[2, 0]
        it_sbt[0, 1] = it_sbt[0, 2]
        it_sbt[1, 1] = (it_sbt[2, 1] + it_sbt[1, 2])/2
        it_sbt[0, 0] = (it_sbt[1, 0] + it_sbt[0, 1])/2

        # bottom left corner
        it_sbt[-2, 0] = it_sbt[-3, 0]
        it_sbt[-1, 1] = it_sbt[-1, 2]
        it_sbt[-2, 1] = (it_sbt[-3, 1] + it_sbt[-2, 2])/2
        it_sbt[-1, 0] = (it_sbt[-2, 0] + it_sbt[-1, 1])/2

        # top right corner
        it_sbt[0, -2] = it_sbt[0, -3]
        it_sbt[1, -1] = it_sbt[2, -1]
        it_sbt[1, -2] = (it_sbt[1, -3] + it_sbt[2, -2])/2
        it_sbt[0, -1] = (it_sbt[0, -2] + it_sbt[1, -1])/2

        # bottom right corner
        it_sbt[-2, -1] = it_sbt[-3, -1]
        it_sbt[-1, -2] = it_sbt[-1, -3]
        it_sbt[-2, -2] = (it_sbt[-3, -2] + it_sbt[-2, -3])/2
        it_sbt[-1, -1] = (it_sbt[-1, -2] + it_sbt[-2, -1])/2


        # apply gradient 2D by sobel
        it_sbt = it_sbt[row, time_ref:]  # get the row intensity
        ita_sbt = np.convolve(it_sbt, np.ones([smooth, ])/smooth, 'same')
        ita_sbt[:2] = it_sbt[:2]
        ita_sbt[-2:] = it_sbt[-2:]

        if (
            np.argmax(it_sbt) in [0, 1, len(it_sbt)-2, len(it_sbt)-1] and
            np.std(it_sbt) > 1e-2
        ):
            peaks = pf.PeakDetection(ita_sbt, fac*np.std(ita_sbt))
            max_location = np.array(peaks.max_location)
            max_magnitude = np.array(peaks.max_magnitude)
            max_location = max_location[
                max_magnitude >= np.mean(it_sbt)
            ]

            if len(max_location) == 2:
                front = max_location[1]
            elif len(max_location) == 1:
                if max_location[0] == 0:
                    it_sbt2 = it_sbt[2:]
                    ita_sbt2 = np.convolve(it_sbt2, np.ones([smooth, ])/smooth, 'same')
                    peaks2 = pf.PeakDetection(ita_sbt2, fac*np.std(ita_sbt2))
                    max_location2 = np.array(peaks2.max_location)
                    max_magnitude2 = np.array(peaks2.max_magnitude)
                    max_location2 = max_location2[
                        max_magnitude2 >= np.mean(it_sbt2)
                    ]
                    if plot:
                        plt.figure()
                        plt.plot(it_sbt)
                        plt.plot(ita_sbt)
                        plt.plot(np.arange(2, len(it_sbt)), it_sbt2)
                        plt.plot(np.arange(2, len(it_sbt)), ita_sbt2)
                        plt.plot(max_location2+2, it_sbt[max_location2+2], 'or')

                    if len(max_location2) == 1:
                        front = max_location2[0]+2
                    else:
                        print('No statut for this case...', row)
                        print(len(max_location2))
                        front = np.nan
                else:
                    front = max_location[0]
            elif len(max_location) == 0:
                print('No peak here', row)
                front = np.nan
            else:
                front = np.nan
                print('To much peaks here', row)

            if plot:
                print(np.std(ita_sbt), np.std(it_sbt))

                plt.figure()
                plt.plot(it_sbt)
                plt.plot(ita_sbt)
                plt.plot(max_location, ita_sbt[max_location], 'or')
                plt.plot(
                    [0, len(it_sbt)],
                    [np.mean(it_sbt), np.mean(it_sbt)],
                    '--k'
                )
                # plt.plot(peaks.min_location, ita_sbt[peaks.min_magnitude], 'ob')
        elif np.argmax(it_sbt) == 0 and np.std(it_sbt <= 1e-2):
            print('No front here ', row)
            front = np.nan
        else:
            front = np.argmax(it_sbt)

        if plot:
            print('Front position : ', front)

            plt.figure()
            plt.imshow(self.current_spatioy.image[:, time_ref:], cmap='gray')
            plt.plot(front, row, 'ob')

            plt.figure()
            plt.plot(it_sbt)
            plt.plot(ita_sbt)
            # plt.plot(front, it_sbt[front], 'ob')
            plt.show()

        return front

    def get_front_all_row(self, time_ref, smooth=3, fac=.5, plot=False):
        """
        Determine the fronts postions in current_spatioy for the indicated row.

        :param time_ref:
        """
        front = np.zeros(self.current_spatioy.size[0])
        for row in range(0, self.current_spatioy.size[0]):
            front[row] = self.get_front(row, time_ref, smooth ,fac, False)

        if plot:
            plt.figure()
            plt.imshow(self.current_spatioy.image[:, time_ref:], cmap='gray')
            plt.plot(front, np.arange(0, self.current_spatioy.size[0]), '.b')
            plt.show()



    def calcul_one_volume_total(self, plot=True):
        """Calculate the volume of one frame."""
        # ===  load x_ref -- value of x === #
        ref_x = np.load(self.contour_directory + 'ref_x.npy')

        # === load contour position for current image === #
        if self.current_image_number < 9:
            contour = np.load(
                self.contour_directory + 'y_000' +
                str(self.current_image_number+1) +
                '.npy'
            )
        elif self.current_image_number < 99:
            contour = np.load(
                self.contour_directory + 'y_00' +
                str(self.current_image_number+1) +
                '.npy'
            )
        elif self.current_image_number < 999:
            contour = np.load(
                self.contour_directory + 'y_0' +
                str(self.current_image_number+1) +
                '.npy'
            )

        # === smooth contour === #
        mean = np.array([1, 1, 1, 1, 1])/5
        contour_2 = np.convolve(contour, mean, 'same')
        contour_2[:2] = contour[:2]
        contour_2[-2:] = contour[-2:]
        contour_2 = np.array(contour_2, dtype=int)

        # === get center === #
        xc = int(np.mean(ref_x)) - np.min(ref_x)

        # === separate left and right contour ==== #
        cl = []
        cr = []
        for i in range(0, xc):
            cl.append(contour_2[i])
        for i in range(xc, len(ref_x)):
            cr.append(contour_2[i])
        cl = np.array(cl)
        cr = np.array(cr)

        # === inversing interface position function === #
        Rr = []
        Rl = []
        left_x = ref_x[:xc] - np.mean(ref_x)
        right_x = ref_x[xc:] - np.mean(ref_x)
        for j in range(min(contour_2), self.baseline):
            try:
                Rr.append(min(right_x[cr == j]))
            except ValueError:
                try:
                    Rr.append(Rr[j-min(contour_2)-1])
                except IndexError:
                    Rr.append(0)
            try:
                Rl.append(max(left_x[cl == j]))
            except ValueError:
                try:
                    Rl.append(Rl[j-min(contour_2)-1])
                except IndexError:
                    Rl.append(0)

        # === fit profil by 2nd & 3rd order polynome === #
        """
        On fit le profil par un polynome d'ordre 2 (~equation de cercle)
        pour la partie inférieure de la goutte, et par un polynome
        d'ordre 3 pour la partie surpérieure (afin de rendre la pointe si
        necessaire).
        On impose 3 conditions au polynome d'ordre 3:
            1. Le point au sommnet est x = 0
            2. Le continuité des fits x_bottom[y_rac] = x_top[y_rac]
            3. La continuité de la dérivée des fits (fonction C1)
        """
        y = np.arange(min(contour_2), self.baseline)
        val = 20  # position de raccord

        # fit the bottom #
        def f2(x, a, b, c):
            return a*x**2 + b*x + c

        f2model = lmmodel(f2)
        params2 = lmparameter()

        params2.add('a', value=1, vary=True)
        params2.add('b', value=1, vary=True)
        params2.add('c', value=1, vary=True)

        left1 = f2model.fit(Rl[val:], params2, x=y[val:])
        right1 = f2model.fit(Rr[val:], params2, x=y[val:])

        left_bottom = [
            left1.params['a'].value,
            left1.params['b'].value,
            left1.params['c'].value
        ]
        right_bottom = [
            right1.params['a'].value,
            right1.params['b'].value,
            right1.params['c'].value
        ]

        # fit the top #
        def f3(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d

        f3model = lmmodel(f3)
        params3 = lmparameter()

        # left conditons
        d_left_expr = (
            '-a*' + str(y[val]**3) + '+'
            '(' + str(left_bottom[0]) + '-b)*' + str(y[val]**2) + '+' +
            '(' + str(left_bottom[1]) + '-c)*' + str(y[val]) + '+' +
            str(left_bottom[2])
        )
        c_left_expr = (
            '-a*' + str(3*y[val]**2) + '+' +
            str(2*y[val]) + '*(' + str(left_bottom[0]) + '-b) +' +
            str(left_bottom[1])
        )
        b_left_expr = (
            str(
                -1*(
                    left_bottom[0]*y[val]**2 +
                    left_bottom[1]*y[val] +
                    left_bottom[2]
                ) /
                (
                    (y.min() - y[val])**2
                )
            ) + '-' +
            str(
                2*left_bottom[0]*y[val] / (y.min() - y[val])
            ) + '-' +
            str(
                left_bottom[1] / (y.min() - y[val])
            ) + '+' +
            'a*(' +
            str(
                3*y[val]**2 / (y.min() - y[val])
            ) + '-' +
            str(
                (y.min()**3 - y[val]**3) /
                (y.min() - y[val])**2
            ) + ')'
        )

        # right conditons
        d_right_expr = (
            '-a*' + str(y[val]**3) + '+'
            '(' + str(right_bottom[0]) + '-b)*' + str(y[val]**2) + '+' +
            '(' + str(right_bottom[1]) + '-c)*' + str(y[val]) + '+' +
            str(right_bottom[2])
        )
        c_right_expr = (
            '-a*' + str(3*y[val]**2) + '+' +
            str(2*y[val]) + '*(' + str(right_bottom[0]) + '-b) +' +
            str(right_bottom[1])
        )
        b_right_expr = (
            str(
                -1*(
                    right_bottom[0]*y[val]**2 +
                    right_bottom[1]*y[val] +
                    right_bottom[2]
                ) /
                (
                    (y.min() - y[val])**2
                )
            ) + '-' +
            str(
                2*right_bottom[0]*y[val] / (y.min() - y[val])
            ) + '-' +
            str(
                right_bottom[1] / (y.min() - y[val])
            ) + '+' +
            'a*('
            + str(
                3*y[val]**2 / (y.min() - y[val])
            ) + '-' +
            str(
                (y.min()**3 - y[val]**3) /
                (y.min() - y[val])**2
            ) + ')'
        )

        # solve left
        params3.add('a', value=1, vary=True)
        params3.add('b', expr=b_left_expr, vary=True)
        params3.add('c', expr=c_left_expr, vary=True)
        params3.add('d', expr=d_left_expr, vary=True)

        left2 = f3model.fit(Rl[:val], params3, x=y[:val])
        left_top = [
            left2.params['a'].value,
            left2.params['b'].value,
            left2.params['c'].value,
            left2.params['d'].value
        ]

        # solve right
        params3.add('a', value=1, vary=True)
        params3.add('b', expr=b_right_expr, vary=True)
        params3.add('c', expr=c_right_expr, vary=True)
        params3.add('d', expr=d_right_expr, vary=True)

        right2 = f3model.fit(Rr[:val], params3, x=y[:val])
        right_top = [
            right2.params['a'].value,
            right2.params['b'].value,
            right2.params['c'].value,
            right2.params['d'].value
        ]

        # final complete fit #
        Rl_fit = left_top[0]*y[:val]**3 +\
            left_top[1]*y[:val]**2 +\
            left_top[2]*y[:val] +\
            left_top[3]
        Rr_fit = right_top[0]*y[:val]**3 +\
            right_top[1]*y[:val]**2 +\
            right_top[2]*y[:val] +\
            right_top[3]
        Rl_fit = np.append(
            Rl_fit,
            left_bottom[0]*y[val:]**2 +
            left_bottom[1]*y[val:] +
            left_bottom[2])
        Rr_fit = np.append(
            Rr_fit,
            right_bottom[0]*y[val:]**2 +
            right_bottom[1]*y[val:] +
            right_bottom[2]
        )

        # ===  calcul the volume === #
        """
        Le volume est égale à la somme des demi-volume de chaque solide
        de révolution autour de l'axe z de courbe Rr et Rl.
        """
        volume = np.pi/2*(
            np.sum(np.square(Rr_fit)) +
            np.sum(np.square(Rl_fit))
        )

        # === calcul contact angles & apex angle === #
        ca_left = np.pi/2 - np.arctan(
            -1 / (
                2*left_bottom[0]*y[0]+left_bottom[1]
            )
        )
        ca_right = np.pi/2 - np.arctan(
            1 / (
                2*right_bottom[0]*y[0]+right_bottom[1]
            )
        )
        a_apex = (
            np.arctan(
                -1 / (
                    3*left_top[0]*y.min()**2 +
                    2*left_top[1]*y.min() +
                    left_top[2]
                )
            ) +
            np.arctan(
                1 / (
                    3*right_top[0]*y.min()**2 +
                    2*right_top[1]*y.min() +
                    right_top[2]
                )
            )
        )

        # === plot informations === #
        if plot:
            print('Volume (in px**3) (with fit)', volume)
            print('Angle à gauche (deg) : ', ca_left*180/np.pi)
            print('Angle à droite (deg) : ', ca_right*180/np.pi)
            print('Angle au sommet (deg) : ', a_apex*180/np.pi)

            fig = plt.figure(figsize=(11, 7))
            fig.canvas.set_window_title(
                'Image with contour n°' + str(self.current_image_number)
            )
            plt.imshow(self.current_image.image, cmap='gray')
            plt.plot(
                ref_x, contour,
                '.', markersize=5, color='#BB0029',
                label='Contour'
            )
            plt.plot(
                ref_x, contour_2,
                '-b',
                label='Contour mean'
            )
            plt.plot(
                Rl_fit+np.mean(ref_x), y,
                '-y',
                label='Fit left contour'
            )
            plt.plot(
                Rr_fit+np.mean(ref_x), y,
                '--k',
                label='Fit right contour'
            )
            plt.legend(shadow=True, fancybox=True, loc='best')

            fig = plt.figure(figsize=(11, 7))
            plt.plot(
                Rl, y,
                '.b',
                label='Left contour detected'
            )
            plt.plot(
                Rr, y,
                '.r',
                label='Right contour detected'
            )
            plt.plot(
                Rl_fit, y,
                '--g',
                label='Fit left contour'
            )
            plt.plot(
                Rr_fit, y,
                '--m',
                label='Fit right contour'
            )
            plt.grid(True)
            plt.legend(shadow=True, fancybox=True, loc='best')
            plt.show()

        # give the volume back
        return volume, ca_left, ca_right, a_apex, Rl_fit, Rr_fit, y

    def calcul_all_volume_total(self, plot=True):
        """Calculate all volume."""
        temp = self.current_image_number  # stocker l'image actuelle
        v = []
        ca_left = []
        ca_right = []
        a_apex = []
        # create wait bar
        widgets = ['Calculating volumes',
                   progressbar.Percentage(),
                   ' ', progressbar.Bar('=', '[', ']'),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(self.image_list)
        )
        pbar.start()

        t_front, y_front = self.get_time_front_all(False)

        tg = []
        vg = []
        # loop calcul volumes
        for i_t in range(0, len(self.image_list)):
            self.read_image(i_t)
            vol, ca_l, ca_r, a_ax, rl, rr, y = \
                self.calcul_one_volume_total(False)
            v = np.append(v, vol)
            ca_left = np.append(ca_left, ca_l)
            ca_right = np.append(ca_right, ca_r)
            a_apex = np.append(ca_right, a_ax)
            if i_t >= t_front[0]:
                idx = [
                    idx for idx, x in enumerate(y) if x == y_front[
                        t_front == i_t
                    ]
                ]
                if len(idx) == 1:
                    tg = np.append(tg, i_t)
                    vg = np.append(vg, np.pi/2*(
                        np.sum(np.square(rl[idx[0]:])) +
                        np.sum(np.square(rr[idx[0]:]))
                    ))
            pbar.update(i_t)
        pbar.finish()

        if plot:
            # === diplay volume + spatio with contour === #
            temp = self.current_spatioy_number
            # we select the center of the image
            self.read_spatio(
                np.array(
                    np.rint(len(self.spatioy_list)/2),
                    dtype=int
                )
            )
            ref_x = np.load(
                self.contour_directory + 'ref_x.npy'
            )
            self.read_spatio(
                int(np.mean(ref_x))
            )
            contour = []
            for i_t in np.arange(
                1, self.current_spatioy.size[1]
            ):

                if i_t < 10:
                    y = np.load(
                        self.contour_directory + 'y_000' +
                        str(i_t) +
                        '.npy'
                    )
                elif i_t < 100:
                    y = np.load(
                        self.contour_directory + 'y_00' +
                        str(i_t) +
                        '.npy'
                    )
                elif i_t < 1000:
                    y = np.load(
                        self.contour_directory + 'y_0' +
                        str(i_t) +
                        '.npy'
                    )
                idx = ref_x == self.current_spatioy_number
                contour.append(y[idx])
            contour = np.array(contour, dtype=int)

            fig, ax1 = plt.subplots(figsize=(11, 7))
            fig.canvas.set_window_title('Volume vs time')

            ax1.plot(
                np.arange(0, self.current_spatioy.size[1]),
                v,
                '.b', markersize=4
            )
            ax1.plot(
                tg, vg*(1 - 916/999) + np.mean(v[:150]),
                '.r', markersize=4
            )
            plt.ylabel(r'Volume ($px^3$)')
            plt.xlabel('Time (frame)')

            plt.grid(True)

            ax2 = fig.add_axes([.1, .4, .4, .4])
            ax2.imshow(self.current_spatioy.image, cmap='gray')
            ax2.plot(
                np.arange(0, self.current_spatioy.size[1]-1),
                contour,
                '.b', markersize=2
            )
            ax2.plot(
                t_front,
                y_front,
                '.r', markersize=2
            )
            ax2.set_title('Spatio_y ' + str(self.current_spatioy_number))

            # === diplay left and right  contact angles === #
            fig_ca, ax_ca = plt.subplots(figsize=(11, 7))
            fig_ca.canvas.set_window_title('Contact angle vs time')

            ax_ca.plot(
                ca_left*180/np.pi, '-r', linewidth=2,
                label='Left contact angle'
            )
            ax_ca.plot(
                ca_right*180/np.pi, '-b', linewidth=2,
                label='Right contact angle'
            )

            plt.xlabel('Time (frame)')
            plt.ylabel('Angle (rad)')
            plt.grid(True)
            plt.legend(fancybox=True, shadow=True)

            self.read_spatio(temp)

            # === diplay apex angles === #
            fig_apex, ax_apex = plt.subplots(figsize=(11, 7))
            fig_apex.canvas.set_window_title('Apex angle vs time')

            ax_apex.plot(
                a_apex*180/np.pi, '-k', linewidth=2,
            )
            plt.xlabel('Time (frame)')
            plt.ylabel('Angle (rad)')

            plt.grid(True)

            # === display rho_alpha === #
            v0 = np.mean(v[:150])
            rho_a = []
            for i_t in range(len(tg)):
                t_tot = np.arange(0, self.current_spatioy.size[1])
                idt = [i for i, x in enumerate(t_tot) if x == tg[i_t]]

                rho_a = np.append(
                    rho_a,
                    (999*v0 - 916*vg[i_t]) /
                    (v[idt] - vg[i_t])
                )
            plt.figure()
            plt.plot(tg[:-1], rho_a[:-1], '.k')

            # === display all figure == #
            plt.show()

        self.read_image(temp)

    def drop_info(
        self, x0, pente, baseline, apex_data=[20, 2],
        px_mm_ratio=-1, plot=False
    ):
        """Calculate drop information.

        :params: x0 : 2 point list containing x position of intersection
            between baseline and drop. (from get_baseline2)
        :params: pente : baseline inclination (from get_baseline2)
        :params: baseline : baseline vector (from get_baseline2)
        :params: px_mm_ratio: aspect ratio (default -1, is none calculate, and
            results are in px)

        :output: mask: mask of 1 where there is the drop and 0 otherwise
        :output: volume: volume of the drop (this doesn't work... because
            axisymétric hypothsesi is wrong)
        :output: area: number of pixel in the drop
        :output: theta: vector containing [0] left contact angle, [1] right
            contact angle
        :output: alpha: angle at the apex
        """
        ni = self.current_image_number+1
        if self.current_image_number < 9:
            Cx = np.load(
                self.contour_directory + '000' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '000' +
                str(ni) + '_Cy' +
                '.npy'
            )
        elif self.current_image_number < 99:
            Cx = np.load(
                self.contour_directory + '00' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '00' +
                str(ni) + '_Cy' +
                '.npy'
            )
        elif self.current_image_number < 999:
            Cx = np.load(
                self.contour_directory + '0' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '0' +
                str(ni) + '_Cy' +
                '.npy'
            )

        # eliminate small elements (< 3 px)
        mask = np.zeros_like(self.current_image.image)
        for i in range(len(Cx)):
            mask[np.int(Cy[i]), np.int(Cx[i])] = 1

        mask_label = measure.label(mask)
        mask_regionprops = measure.regionprops(mask_label)
        for region in mask_regionprops:
            if region.area < 3:
                mask_label[mask_label == region.label] = 0
        mask[mask_label < 1] = 0

        Cx2 = np.arange(x0[0], x0[1])
        Cy2 = np.array([])
        for i in Cx2:
            col = mask[:, i]
            if np.max(col) > 0:
                Cy2 = np.append(Cy2, np.argmax(col))
            else:
                Cy2 = np.append(Cy2, np.NaN)

        nan_loc = np.argwhere(np.isnan(Cy2)).flatten()

        c0 = 0
        c = np.array([], dtype=np.int)
        while c0 < len(nan_loc):
            while (c0 < len(nan_loc)-1) and (nan_loc[c0+1] == nan_loc[c0]+1):
                c0 += 1
            if c0+1 != len(nan_loc):
                c = np.append(c, c0+1)
            c0 += 1
        nan_loc = np.split(nan_loc, c)

        for i in range(len(nan_loc)):
            if nan_loc[i][0] == 0:
                val_left = baseline[x0[0]]
            else:
                val_left = Cy2[nan_loc[i][0]-1]
            if nan_loc[i][-1] == len(Cx2)-1:
                val_right = baseline[x0[1]]
            else:
                val_right = Cy2[nan_loc[i][-1]+1]
            for j in range(len(nan_loc[i])):
                Cy2[nan_loc[i][j]] = j*(val_right-val_left)/len(nan_loc[i])+val_left

        f = interpolate.interp1d(Cx2, Cy2, 'cubic')
        Cx_new = np.arange(x0[0], x0[1], 1)
        Cy_new = f(Cx_new)

        x_mean_left = Cx2[np.argmin(Cy2)]
        x_mean_right = Cx2[::-1][np.argmin(Cy2[::-1])]
        x_mean = np.int((x_mean_left + x_mean_right) / 2)
        Cx_left = Cx2[Cx2 <= x_mean]
        Cx_right = Cx2[Cx2 > x_mean]
        Cy_left = Cy2[Cx2 <= x_mean]
        Cy_right = Cy2[Cx2 > x_mean]

        r_left = x_mean - Cx_left
        r_right = Cx_right - Cx_right

        mask = np.zeros_like(self.current_image.image)
        for i in range(x0[1]-x0[0]):
            for j in range(self.current_image.size[0]):
                if j <= baseline[i] and j >= Cy_new[i]:
                    mask[j, i+x0[0]] = 1

        area = np.sum(mask)

        volume = np.pi/2*(
            np.sum(np.square(r_left)) +
            np.sum(np.square(r_right))
        )

        if px_mm_ratio != -1:
            volume *= px_mm_ratio**3
            area *= px_mm_ratio**2

        # left & right contact angle
        nb_pt = 10
        z_left = np.polyfit(Cx_left[:nb_pt], Cy_left[:nb_pt], 1)
        z_right = np.polyfit(Cx_right[-nb_pt:], Cy_right[-nb_pt:], 1)
        p_left = np.poly1d(z_left)
        p_right = np.poly1d(z_right)
        theta_left = np.arctan(
            np.abs(
                (z_left[0] - pente) / (1 + z_left[0]*pente)
            )
        )
        theta_right = np.arctan(
            np.abs(
                (z_right[0] - pente) / (1 + z_right[0]*pente)
            )
        )

        # apex angle
        nb_pt2 = apex_data[0]
        z2_left = np.polyfit(Cx_left[-nb_pt2:], Cy_left[-nb_pt2:], apex_data[1])
        z2_right = np.polyfit(Cx_right[:nb_pt2], Cy_right[:nb_pt2], apex_data[1])
        p2_left = np.poly1d(z2_left)
        p2_right = np.poly1d(z2_right)
        p2_left_prime = np.polyder(p2_left)
        p2_right_prime = np.polyder(p2_right)
        alpha = np.arctan(
            np.abs(
                (p2_left_prime(x_mean) - p2_right_prime(x_mean)) /
                (1 + p2_left_prime(x_mean)*p2_right_prime(x_mean))
            )
        )
        alpha = np.pi - alpha

        if plot:
            plt.figure()
            plt.plot(Cx_left, Cy_left, '.r')
            plt.plot(Cx_right, Cy_right, '.b')
            plt.plot(Cx_left[:nb_pt], p_left(Cx_left[:nb_pt]), '--b')
            plt.plot(Cx_right[-nb_pt:], p_right(Cx_right[-nb_pt:]), '--r')
            plt.plot(Cx_left[-nb_pt2:], p2_left(Cx_left[-nb_pt2:]), '--b')
            plt.plot(Cx_right[:nb_pt2], p2_right(Cx_right[:nb_pt2]), '--r')
            plt.grid(True)

        if plot:
            print('Volume(px or mm^3) : ', volume)
            print('Area(px or mm^2) : ', area)
            print('Ca left(°) : ', theta_left*180/np.pi)
            print('Ca right(°) : ', theta_right*180/np.pi)
            print('Apex angle(°) : ', (np.pi-alpha)*180/np.pi)
            plt.figure()
            plt.imshow(mask, cmap='gray')
            plt.figure()
            plt.imshow(self.current_image.image, cmap='gray')
            plt.plot(np.arange(self.current_image.size[1]), baseline, '--y')
            plt.plot(Cx2, Cy2, '.b', markersize=2)
            plt.plot(Cx_new, Cy_new, '--r')
            plt.show()

        return mask, volume, area, [theta_left, theta_right], alpha

    def drop_info_all(self, x0, pente, baseline, px_mm_ratio=-1, plot=True):
        """Calculate all volume."""
        nb_pt, order = self.apex_angle(x0, pente, baseline)
        temp = self.current_image_number  # stocker l'image actuelle

        ### create wait bar ###
        widgets = ['Calculating drop infos',
                   progressbar.Percentage(),
                   ' ', progressbar.Bar('=', '[', ']'),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(self.image_list)
        )
        pbar.start()

        ### loop calcul volumes ###
        for i in range(len(self.image_list)):
            self.read_image(i)
            mask, volume, area, ca, alpha = \
                self.drop_info(
                    x0, pente, baseline, [nb_pt, order], px_mm_ratio, False
                )
            if i < 9:
                np.save(
                    self.contour_directory + '/000' + str(i+1) + '_mask',
                    mask
                )
                np.save(
                    self.drop_info_directory + '/000' + str(i+1) + '_volume',
                    volume
                )
                np.save(
                    self.drop_info_directory + '/000' + str(i+1) + '_area',
                    area
                )
                np.save(
                    self.drop_info_directory + '/000' + str(i+1) + '_ca',
                    ca
                )
                np.save(
                    self.drop_info_directory + '/000' + str(i+1) + '_alpha',
                    alpha
                )
            elif i < 99:
                np.save(
                    self.contour_directory + '/00' + str(i+1) + '_mask',
                    mask
                )
                np.save(
                    self.drop_info_directory + '/00' + str(i+1) + '_volume',
                    volume
                )
                np.save(
                    self.drop_info_directory + '/00' + str(i+1) + '_area',
                    area
                )
                np.save(
                    self.drop_info_directory + '/00' + str(i+1) + '_ca',
                    ca
                )
                np.save(
                    self.drop_info_directory + '/00' + str(i+1) + '_alpha',
                    alpha
                )
            elif i < 999:
                np.save(
                    self.contour_directory + '/0' + str(i+1) + '_mask',
                    mask
                )
                np.save(
                    self.drop_info_directory + '/0' + str(i+1) + '_volume',
                    volume
                )
                np.save(
                    self.drop_info_directory + '/0' + str(i+1) + '_area',
                    area
                )
                np.save(
                    self.drop_info_directory + '/0' + str(i+1) + '_ca',
                    ca
                )
                np.save(
                    self.drop_info_directory + '/0' + str(i+1) + '_alpha',
                    alpha
                )
            pbar.update(i)
        pbar.finish()
        self.read_image(temp)

    def display_drop_info(self):
        """Display drop informations vvs times."""
        print('Drop informations')
        vol = np.empty([self.n_image_tot, ], dtype=np.double)
        area = np.empty([self.n_image_tot, ], dtype=np.double)
        ca = np.empty([self.n_image_tot, 2], dtype=np.double)
        alpha = np.empty([self.n_image_tot, ], dtype=np.double)

        for it in range(self.n_image_tot):
            if it < 9:
                vol[it] = np.load(
                    self.drop_info_directory + '000' + str(it+1) + '_volume' +
                    '.npy'
                )
                area[it] = np.load(
                    self.drop_info_directory + '000' + str(it+1) + '_area' +
                    '.npy'
                )
                ca[it] = np.load(
                    self.drop_info_directory + '000' + str(it+1) + '_ca' +
                    '.npy'
                )
                alpha[it] = np.load(
                    self.drop_info_directory + '000' + str(it+1) + '_alpha' +
                    '.npy'
                )
            elif it < 99:
                vol[it] = np.load(
                    self.drop_info_directory + '00' + str(it+1) + '_volume' +
                    '.npy'
                )
                area[it] = np.load(
                    self.drop_info_directory + '00' + str(it+1) + '_area' +
                    '.npy'
                )
                ca[it] = np.load(
                    self.drop_info_directory + '00' + str(it+1) + '_ca' +
                    '.npy'
                )
                alpha[it] = np.load(
                    self.drop_info_directory + '00' + str(it+1) + '_alpha' +
                    '.npy'
                )
            elif it < 999:
                vol[it] = np.load(
                    self.drop_info_directory + '0' + str(it+1) + '_volume' +
                    '.npy'
                )
                area[it] = np.load(
                    self.drop_info_directory + '/0' + str(it+1) + '_area' +
                    '.npy'
                )
                ca[it] = np.load(
                    self.drop_info_directory + '/0' + str(it+1) + '_ca' +
                    '.npy'
                )
                alpha[it] = np.load(
                    self.drop_info_directory + '/0' + str(it+1) + '_alpha' +
                    '.npy'
                )







        fig, ax = plt.subplots(
            2, 2,
            figsize=(11, 7),
        )
        ax[0, 0].plot(vol)
        ax[0, 1].plot(area)
        ax[1, 0].plot(ca[:, 0]*180/np.pi, '--r')
        ax[1, 0].plot(ca[:, 1]*180/np.pi, '--b')
        ax[1, 1].plot(alpha*180/np.pi)
        plt.show()

    def apex_angle(self, x0, pente, baseline=0):
        """Display apex angle vs nb_point.

        For calculate the tangentn we need to chose an order for the polynome
        and the point number for the fit.

        :output : nb_pt : provide the number of point for the fit
        :output : order : order for polynomial fit
        """
        nb_point = np.arange(5, 200)

        ni = self.n_image_tot
        if ni < 9:
            Cx = np.load(
                self.contour_directory + '000' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '000' +
                str(ni) + '_Cy' +
                '.npy'
            )
        elif ni < 99:
            Cx = np.load(
                self.contour_directory + '00' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '00' +
                str(ni) + '_Cy' +
                '.npy'
            )
        elif ni < 999:
            Cx = np.load(
                self.contour_directory + '0' +
                str(ni) + '_Cx' +
                '.npy'
            )
            Cy = np.load(
                self.contour_directory + '0' +
                str(ni) + '_Cy' +
                '.npy'
            )

        # eliminate small elements (< 3 px)
        mask = np.zeros_like(self.current_image.image)
        for i in range(len(Cx)):
            mask[np.int(Cy[i]), np.int(Cx[i])] = 1

        mask_label = measure.label(mask)
        mask_regionprops = measure.regionprops(mask_label)
        for region in mask_regionprops:
            if region.area < 3:
                mask_label[mask_label == region.label] = 0
        mask[mask_label < 1] = 0

        Cx2 = np.arange(x0[0], x0[1])
        Cy2 = np.array([])
        for i in Cx2:
            col = mask[:, i]
            if np.max(col) > 0:
                Cy2 = np.append(Cy2, np.argmax(col))
            else:
                Cy2 = np.append(Cy2, np.NaN)

        nan_loc = np.argwhere(np.isnan(Cy2)).flatten()

        c0 = 0
        c = np.array([], dtype=np.int)
        while c0 < len(nan_loc):
            while (c0 < len(nan_loc)-1) and (nan_loc[c0+1] == nan_loc[c0]+1):
                c0 += 1
            if c0+1 != len(nan_loc):
                c = np.append(c, c0+1)
            c0 += 1
        nan_loc = np.split(nan_loc, c)

        for i in range(len(nan_loc)):
            if nan_loc[i][0] == 0:
                val_left = baseline[x0[0]]
            else:
                val_left = Cy2[nan_loc[i][0]-1]
            if nan_loc[i][-1] == len(Cx2)-1:
                val_right = baseline[x0[1]]
            else:
                val_right = Cy2[nan_loc[i][-1]+1]
            for j in range(len(nan_loc[i])):
                Cy2[nan_loc[i][j]] = \
                    j*(val_right-val_left)/len(nan_loc[i])+val_left

        x_mean_left = Cx2[np.argmin(Cy2)]
        x_mean_right = Cx2[::-1][np.argmin(Cy2[::-1])]
        x_mean = np.int((x_mean_left + x_mean_right) / 2)
        Cx_left = Cx2[Cx2 <= x_mean]
        Cx_right = Cx2[Cx2 > x_mean]
        Cy_left = Cy2[Cx2 <= x_mean]
        Cy_right = Cy2[Cx2 > x_mean]

        plt.figure()
        for nb_pt in nb_point:
            z1_left = np.polyfit(Cx_left[-nb_pt:], Cy_left[-nb_pt:], 1)
            z1_right = np.polyfit(Cx_right[:nb_pt], Cy_right[:nb_pt], 1)
            p1_left = np.poly1d(z1_left)
            p1_right = np.poly1d(z1_right)
            p1_left_prime = np.polyder(p1_left)
            p1_right_prime = np.polyder(p1_right)
            alpha = np.arctan(
                np.abs(
                    (p1_left_prime(x_mean) - p1_right_prime(x_mean)) /
                    (1 + p1_left_prime(x_mean)*p1_right_prime(x_mean))
                )
            )
            alpha = np.pi - alpha
            plt.plot(nb_pt, alpha*180/np.pi, '.k', label='order-1')

            z2_left = np.polyfit(Cx_left[-nb_pt:], Cy_left[-nb_pt:], 2)
            z2_right = np.polyfit(Cx_right[:nb_pt], Cy_right[:nb_pt], 2)
            p2_left = np.poly1d(z2_left)
            p2_right = np.poly1d(z2_right)
            p2_left_prime = np.polyder(p2_left)
            p2_right_prime = np.polyder(p2_right)
            alpha = np.arctan(
                np.abs(
                    (p2_left_prime(x_mean) - p2_right_prime(x_mean)) /
                    (1 + p2_left_prime(x_mean)*p2_right_prime(x_mean))
                )
            )
            alpha = np.pi - alpha
            plt.plot(nb_pt, alpha*180/np.pi, '.b', label='order-2')

            z3_left = np.polyfit(Cx_left[-nb_pt:], Cy_left[-nb_pt:], 3)
            z3_right = np.polyfit(Cx_right[:nb_pt], Cy_right[:nb_pt], 3)
            p3_left = np.poly1d(z3_left)
            p3_right = np.poly1d(z3_right)
            p3_left_prime = np.polyder(p3_left)
            p3_right_prime = np.polyder(p3_right)
            alpha = np.arctan(
                np.abs(
                    (p3_left_prime(x_mean) - p3_right_prime(x_mean)) /
                    (1 + p3_left_prime(x_mean)*p3_right_prime(x_mean))
                )
            )
            alpha = np.pi - alpha
            plt.plot(nb_pt, alpha*180/np.pi, '.r', label='order-3')

        plt.grid(True)

        plt.figure()
        plt.imshow(self.current_image.image, cmap='gray')
        plt.plot(Cx_left, Cy_left, '--r')
        plt.plot(Cx_right, Cy_right, '--b')
        plt.plot(
            [Cx_left[-200], Cx_right[200]],
            [Cy_left[-200], Cy_right[200]],
            '--r', linewidth=3,
        )
        plt.plot(
            [Cx_left[-5], Cx_right[5]],
            [Cy_left[-5], Cy_right[5]],
            '--b', linewidth=3,
        )
        plt.show()

        nb_pt = int(input('Which point number : '))
        order = int(input('Which order for polynome : '))

        return nb_pt, order

    def correct_mask(self, time_ref):
        """Corect the mask for image detection."""
        if time_ref+1 < 9:
            mask = np.load(
                self.contour_directory + '000' + str(time_ref+1) + '_mask' +
                '.npy'
            )
        elif time_ref+1 < 99:
            mask = np.load(
                self.contour_directory + '00' + str(time_ref+1) + '_mask' +
                '.npy'
            )
        elif time_ref+1 < 999:
            mask = np.load(
                self.contour_directory + '0' + str(time_ref+1) + '_mask' +
                '.npy'
            )

        for it in range(0, time_ref):
            if it < 9:
                np.save(
                    self.contour_directory + '/000' + str(it+1) + '_mask',
                    mask
                )
            elif it < 99:
                np.save(
                    self.contour_directory + '/00' + str(it+1) + '_mask',
                    mask
                )
            elif it < 999:
                np.save(
                    self.contour_directory + '/0' + str(it+1) + '_mask',
                    mask
                )

class StackProcessingError(Exception):
    """Create excpetion for StacProcessing class."""

    pass


if __name__ == '__main__':
    """
    Classical stack processing procedure:

    1. crop images
    2. image enhancement
    3. create spatios
    4. get contours
    5. set reference (for time)
    6. get baseline (substrat line)
    7. get time front all
    """
    # === read stack === #
    # refexp = 'rep1'
    # serie = 2

    # === creating path === #
    # path = (
    #     '/Users/eablonet/Documents/0_phd/0_data_temp/' + refexp + '/' + refexp
    # )
    # if serie < 10:
    #     path += 'n0' + str(serie) + '/'
    # else:
    #     path += 'n' + str(serie) + '/'
    # print(path)

    # === creating stack ===
    # stack = StackProcessing(path)
    # stack.print_info()
    # stack.read_image(329)
    # stack.current_image.show_image()

    # === testing crop and clahe === #
    # stack.treatment('crop', 150, 850, 500, 850)
    # stack.treatment('clahe', .2, 8)

    # === procedure final === #
    # stack.remove_treatments()
    # stack.remove_spatios()
    # stack.save_treatment('crop', 100, 900, 550, 850)
    # stack.save_treatment('clahe', .2, 8)
    # stack.create_spatio()
    # stack.get_contours()
    # stack.set_reference()
    # stack.get_baseline(True)
    # stack.get_time_front_all()

    # === ploting === #
    # # stack.display_contour()
    # stack.read_spatio(350)
    # stack.current_spatioy.show_image()
    # stack.calcul_all_volume_total()

    """
    Multiple front plot
    """
    # refexp = 'rep1'
    # series = [2, 4, 5, 6]
    # ratio = 9e-3/1197
    # fps = [4.99, 4.99, 4., 4.]
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(11, 7))
    #
    # for i, serie in enumerate(series):
    #     path = (
    #         '/Users/eablonet/Documents/0_phd/0_data_temp/' + refexp + '/' +
    #         refexp
    #     )
    #     if serie < 10:
    #         path += 'n0' + str(serie) + '/'
    #     else:
    #         path += 'n' + str(serie) + '/'
    #
    #     stack = StackProcessing(path)
    #     stack.read_spatio(400)
    #     stack.set_reference()
    #     stack.get_baseline()
    #     t_front, y_front = stack.get_time_front_all()
    #     ax.plot(
    #         (t_front - np.nanmin(t_front))/fps[i],
    #         (np.nanmax(y_front) - y_front)*ratio,
    #         '*', markersize=8,
    #         label=str(serie)
    #     )
    # plt.ioff()
    # plt.grid(True)
    # plt.show()

    """
    Test of front detection procedure.

    ..todo
    """
    # === reading random exp === #
    # refexp = 'rep1'
    # serie = np.random.randint(2, 7)
    # print('Reading serie : ', serie)
    # path = (
    #     '/Users/eablonet/Documents/0_phd/0_data_temp/' + refexp + '/' + refexp
    # )
    # if serie < 10:
    #     path += 'n0' + str(serie) + '/'
    # else:
    #     path += 'n' + str(serie) + '/'

    # === creating stack ===
    # stack = StackProcessing(path)
    #
    # # === reading random spatio === #
    # n = np.random.randint(len(stack.spatioy_list))
    # while n < 350 or n > 550:
    #     n = np.random.randint(len(stack.spatioy_list))
    # print('Reading spatio : ', n)
    # stack.read_spatio(425)

    # set time reference === #
    # t_ref = stack.set_time_reference()

    # === reading random row === #
    # c = 0
    # while c < 6:
    #     row = np.random.randint(stack.current_spatioy.size[1])
    #     while row < 74 or row > 330:
    #         row = np.random.randint(stack.current_spatioy.size[1])
    #     print('Evaluating row : ', row)
    #
    #     stack.get_time_front(
    #         rows=row, time_ref=t_ref, plot=True
    #     )
    #     c += 1
    # plt.show()

    """
    New tests
    """
    path = [
        "/Volumes/EMERYK_HD/0_data/29-03-2018/n5/",
        "/Volumes/EMERYK_HD/0_data/30-03-2018/n5/",
        "/Volumes/EMERYK_HD/0_data/06-04-2018/n5/",
        "/Volumes/EMERYK_HD/0_data/09-04-2018/n5/",
        "/Volumes/EMERYK_HD/0_data/10-04-2018/n5/",
        "/Volumes/EMERYK_HD/0_data/13-04-2018/n5/",
        "/Volumes/EMERYK_HD/0_data/17-04-2018/n5/",
        ]
    p = np.random.randint(len(path))
    print('Reading path : ', path[1])
    stack = StackProcessing(path[1])
    # stack.print_info()
    # n = np.random.randint(162)
    # print(n)
    # stack.read_image(n)
    # stack.current_image.show_image()

    # stack.remove_treatments()
    # stack.treatment('crop', 310, 1200, 650, 980)
    # stack.save_treatment('crop', 310, 1200, 650, 980)
    # stack.treatment('clahe', 2.5, 16)
    # stack.save_treatment('clahe', 2.5, 16)
    # x, ymin, ymax = stack.set_contour_ref()
    # ll = [90, 115, 130, 150, 161]
    # for i in ll:
    #     stack.read_image(i)
    #     stack.get_contour(0, 9, 0, 1, True)
    # stack.display_contour()

    # stack.get_contours()
    # for ii in ll:
    #   stack.display_contour(ii)

    # stack.read_image(161)
    # baseline, x0 = stack.get_baseline2()
    # stack.drop_info_all(x0, baseline[1], baseline[0], -1, True)
    # stack.display_drop_info()
    # stack.correct_mask(118)
    # stack.create_spatio2(True)
    stack.read_spatio(400, 'y')
    # stack.read_spatio(90, 'x')
    # stack.current_spatioy.show_image()
    # stack.current_spatiox.show_image()
    # ll = np.array([30, 90, 110, 150, 200], dtype=np.int)
    # for i in ll:
    #    stack.get_time_front(np.int(i), 118, plot=True)
    # plt.show()
    fac = .75
    stack.get_front(row=210, time_ref=118, fac=fac, plot=True)
    stack.get_front(row=217, time_ref=118, fac=fac, plot=True)

    stack.get_front_all_row(time_ref=118, fac=fac, plot=True)
