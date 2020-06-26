import numpy as np
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
from numpy import mean, std
from scipy.optimize import curve_fit
import scipy.ndimage
from scipy import ndimage as ndi

import settings


def rgb2gray(rgb):
    """
    :param rgb: image 2D list
    :return: image 2D list in gray scale
    """
    return np.dot(rgb[..., :3], [settings.RED_RATIO, settings.GREEN_RATIO, settings.BLUE_RATIO])


def read_raw_image_2gray(file):
    image = np.fromfile(file, dtype=np.uint8)
    image.shape = (settings.ROW, settings.COLUMN, settings.RGB_INDEX)
    return rgb2gray(image)


def gauss_function(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def get_gauss_fitting_params(interval, intensity_ave_no_bg):
    """
    Get Gaussian fitting parameters.
    :param interval: x range
    :param intensity_ave_no_bg: y value
    :return: popt, pcov, where popt is a list of optimal values for all fitting parameters
    """
    average = sum(intensity_ave_no_bg * interval) / sum(intensity_ave_no_bg)
    sigma = np.sqrt(np.absolute(sum(intensity_ave_no_bg * (interval - average) ** 2) / sum(intensity_ave_no_bg)))
    return curve_fit(gauss_function, interval, intensity_ave_no_bg, p0=[1., average, sigma, 0], maxfev=10000000)


def calculate_intensity_avg_no_bg(bg, intensity_ave):
    """
    Get intensity after subtracting the background.
    :param bg: a float of calculated background
    :param intensity_ave: 1D list of averaged intensity
    :return: 1D list of intensity with background subtracted
    """
    intensity_ave_no_bg = [i - bg for i in intensity_ave]
    for index in range(len(intensity_ave_no_bg)):
        intensity_ave_no_bg[index] = 0 if intensity_ave_no_bg[index] < 0 else intensity_ave_no_bg[index]
    return intensity_ave_no_bg


class ProcessRAWImage:
    def __init__(self, file_loc, file_list):
        self.file_location = file_loc
        self.file_list = file_list
        self.image_file_list = []
        self.x_rms_all = []
        self.y_rms_all = []
        self.x_rms_std_all = []
        self.y_rms_std_all = []
        self.sort_and_combine_raw = []
        self.sol_vol_list = []
        self.crop_x_start = None
        self.crop_x_end = None
        self.crop_y_start = None
        self.crop_y_end = None
        self.bg_x_start = None
        self.bg_x_end = None
        self.bg_y_start = None
        self.bg_y_end = None

        for file in sorted(self.file_list):
            if file.endswith(settings.IMAGE_FORMAT):
                self.image_file_list.append(file)
                sol_voltage = file.split("-")[0].split("_")[0].replace("p", ".")
                if sol_voltage not in self.sol_vol_list:
                    self.sol_vol_list.append(sol_voltage)

    def set_crop_region(self, crop_x_start, crop_x_end, crop_y_start, crop_y_end):
        self.crop_x_start = crop_x_start
        self.crop_x_end = crop_x_end
        self.crop_y_start = crop_y_start
        self.crop_y_end = crop_y_end

    def set_bg_region(self, bg_x_start, bg_x_end, bg_y_start, bg_y_end):
        self.bg_x_start = bg_x_start
        self.bg_x_end = bg_x_end
        self.bg_y_start = bg_y_start
        self.bg_y_end = bg_y_end

    def get_background(self, image):
        cropped_background = image[self.bg_y_start:self.bg_y_end, self.bg_x_start:self.bg_x_end]
        return np.mean(cropped_background.flatten())

    def get_intensity_ave_no_bg(self, image):
        zoom_in_image = image[self.crop_y_start:self.crop_y_end, self.crop_x_start:self.crop_x_end]

        bg = self.get_background(zoom_in_image)
        x_intensity_ave = zoom_in_image.mean(axis=0).tolist()
        y_intensity_ave = zoom_in_image.mean(axis=1).tolist()
        return calculate_intensity_avg_no_bg(bg, x_intensity_ave), calculate_intensity_avg_no_bg(bg, y_intensity_ave)

    def plot_single_frame(self, image, x_intensity_ave_no_bg, y_intensity_ave_no_bg, file, single_frame_counter=None):
        zoom_in_image = image[self.crop_y_start:self.crop_y_end, self.crop_x_start:self.crop_x_end]
        x = np.linspace(0, len(x_intensity_ave_no_bg) * settings.PIXEL_RESOLUTION, len(x_intensity_ave_no_bg))
        y = np.linspace(0, len(y_intensity_ave_no_bg) * settings.PIXEL_RESOLUTION, len(y_intensity_ave_no_bg))

        # fig = plt.figure()
        gs = gridspec.GridSpec(4, 4)
        ax_main = plt.subplot(gs[1:4, 0:3])

        ax_x_dist = plt.subplot(gs[0, 0:3], sharex=ax_main)
        ax_y_dist = plt.subplot(gs[1:4, 3], sharey=ax_main)

        ax_main.imshow(zoom_in_image, aspect='auto', cmap='jet', extent=(min(x), max(x), max(y), min(y)))
        ax_main.axis([0, len(zoom_in_image[0]) * settings.PIXEL_RESOLUTION,
                      len(zoom_in_image) * settings.PIXEL_RESOLUTION, 0])
        ax_main.set(xlabel="x (mm)", ylabel="y (mm)")

        plt.suptitle("%s \nFrame#%02i" % (file, single_frame_counter + 1))

        smooth_results_to_plt = scipy.ndimage.zoom(zoom_in_image, 1)
        X, Y = np.meshgrid(x, y)
        # ax_main.contour(X, Y, smooth_results_to_plt,
        #                 levels=[settings.CONTOUR_LEVEL],
        #                 colors='red')

        x_intensity_ave_no_bg_normalized = [i * 1 / max(x_intensity_ave_no_bg) for i in x_intensity_ave_no_bg]
        y_intensity_ave_no_bg_normalized = [i * 1 / max(y_intensity_ave_no_bg) for i in y_intensity_ave_no_bg]

        ax_x_dist.bar(x, x_intensity_ave_no_bg_normalized, .1, color='b')
        ax_x_dist.set(ylabel='Intensity')
        plt.setp(ax_x_dist.get_xticklabels(), visible=False)

        ax_y_dist.barh(y, y_intensity_ave_no_bg_normalized, .1, color='b')
        ax_y_dist.set(xlabel='Intensity')
        plt.setp(ax_y_dist.get_yticklabels(), visible=False)

        # Gaussian fit
        popt_x, pcov_x = get_gauss_fitting_params(x, x_intensity_ave_no_bg_normalized)
        popt_y, pcov_y = get_gauss_fitting_params(y, y_intensity_ave_no_bg_normalized)

        ax_y_dist.plot(gauss_function(y, *popt_y), y, label='gaussian fit', c='C03', alpha=1)
        ax_y_dist.text(0.5, ((self.crop_y_end - self.crop_y_start) / 2 - 20) * settings.PIXEL_RESOLUTION,
                       '$\sigma$$_{y}$ = %.4f mm' % (abs(popt_y[2])),
                       rotation=270)

        ax_x_dist.plot(x, gauss_function(x, *popt_x), label='gaussian fit', c='C03', alpha=1)
        ax_x_dist.text(((self.crop_x_end - self.crop_x_start) / 2 + 20) * settings.PIXEL_RESOLUTION, 0.5,
                       '$\sigma$$_{x}$ = %.4f mm' % (abs(popt_x[2])))
        plt.show()

    def solenoid_voltage_list(self):
        # print(self.sol_vol_list)
        return self.sol_vol_list

    def calculate_rms_and_error(self, plot_show=False):
        coutr = -1
        x_rms_temp = []
        y_rms_temp = []
        for file in sorted(self.image_file_list):
            image = read_raw_image_2gray(self.file_location + file)
            image = ndi.median_filter(image, 3)

            x_intensity_ave_no_bg, y_intensity_ave_no_bg = self.get_intensity_ave_no_bg(image)

            x = np.linspace(0, len(x_intensity_ave_no_bg), len(x_intensity_ave_no_bg))
            y = np.linspace(0, len(y_intensity_ave_no_bg), len(y_intensity_ave_no_bg))

            # Gaussian fit
            popt_x, pcov_x = get_gauss_fitting_params(x, x_intensity_ave_no_bg)
            popt_y, pcov_y = get_gauss_fitting_params(y, y_intensity_ave_no_bg)

            x_rms_temp.append(abs(popt_x[2]) * settings.PIXEL_RESOLUTION)
            y_rms_temp.append(abs(popt_y[2]) * settings.PIXEL_RESOLUTION)

            if plot_show:
                coutr += 1
                self.plot_single_frame(image, x_intensity_ave_no_bg, y_intensity_ave_no_bg, file,
                                       single_frame_counter=coutr % settings.FRAME_NUM)

        x_rms_all_temp = np.reshape(x_rms_temp, (-1, settings.FRAME_NUM))
        y_rms_all_temp = np.reshape(y_rms_temp, (-1, settings.FRAME_NUM))

        self.x_rms_std_all = np.std(x_rms_all_temp, axis=1)
        self.y_rms_std_all = np.std(y_rms_all_temp, axis=1)

        self.x_rms_all = np.mean(x_rms_all_temp, axis=1)
        self.y_rms_all = np.mean(y_rms_all_temp, axis=1)
        return self.x_rms_all, self.y_rms_all, self.x_rms_std_all, self.y_rms_std_all



