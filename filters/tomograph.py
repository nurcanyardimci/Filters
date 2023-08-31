import numpy as np
import bresenham
import filter
from math import sin, cos


class TomographParameters:
    def __init__(self, image, theta, detector_quantity, span, filter_type):
        self.image = image
        self.theta = np.deg2rad(float(theta))
        self.detector_quantity = int(detector_quantity)
        self.span = np.deg2rad(span)
        self.filter_type = filter_type
        self.emitter_angles = generate_angles(self.theta)
        self.theta_deg = theta
        self.span_deg = span

    def set_parameters(self, image, theta, detector_quantity, span, filter_type):
        self.image = image
        self.theta = np.deg2rad(float(theta))
        self.detector_quantity = int(detector_quantity)
        self.span = np.deg2rad(span)
        self.filter_type = filter_type
        self.emitter_angles = generate_angles(self.theta)
        self.theta_deg = theta
        self.span_deg = span


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coords(self, center, angle):
        s = sin(angle)
        c = cos(angle)
        x = self.x - center.x
        y = self.y - center.y
        nx = x * c - y * s
        ny = x * s + y * c
        x = nx + center.x
        y = ny + center.y
        return Point(round(x), round(y))


class TransformSnapshot:
    def __init__(self, sinogram=None, image_reconstructed=None, square_error=None) -> None:
        self.sinogram = sinogram
        self.image_reconstructed = image_reconstructed
        self.mse_error = square_error


class Tomograph:
    def __init__(self, params, plot, is_interactive) -> None:
        self.params = params
        self.plot = plot
        self.is_interactive = is_interactive
        self.sinogram = None
        self.image_reconstructed = None
        self.mse_error = 0
        self.refresh_sinogram = self.refresh_image_reconstructed = False
        self.snapshots = [TransformSnapshot() for _ in self.params.emitter_angles]
        self.mse_data = []

    def get_snapshot(self, i):
        i = int(i / 99 * (len(self.params.emitter_angles) - 1))  # slider takes values 0-99
        snap = self.snapshots[i]
        self.image_reconstructed = snap.image_reconstructed
        self.sinogram = snap.sinogram
        self.refresh_sinogram = True
        self.mse_error = snap.mse_error
        self.mse_data = [s.mse_error for s in self.snapshots[:i]]
        self.refresh_image_reconstructed = True

    def history_builder(self, sinogram=None, image_reconstructed=None, iteration=None):
        if sinogram is not None:
            if self.sinogram is None:
                self.sinogram = sinogram
                self.plot.put_sinogram_in_animation_buf(sinogram)
            else:
                self.sinogram = sinogram
            self.snapshots[iteration].sinogram = np.array(sinogram)
            self.refresh_sinogram = True
        if image_reconstructed is not None:
            if self.image_reconstructed is None:
                self.image_reconstructed = image_reconstructed
                self.plot.put_image_reconstructed_in_animation_buf(image_reconstructed)
            else:
                self.image_reconstructed = image_reconstructed
            self.mse_error = get_mean_squared_error(self.plot.image, image_reconstructed)
            self.snapshots[iteration].image_reconstructed = np.array(image_reconstructed)
            self.snapshots[iteration].mse_error = self.mse_error
            self.mse_data.append(self.snapshots[iteration].mse_error)
            self.refresh_image_reconstructed = True

    def image_reconstruction(self, on_finish_task):
        self.sinogram = radon(self.plot.image, self.params.emitter_angles, self.params.detector_quantity,
                              self.params.span, self.is_interactive, history_builder=self.history_builder)
        if self.params.filter_type != "None":
            sinogram_filtered = filter.filter_sinogram(self.sinogram, self.params.filter_type)
            self.image_reconstructed = inverse_radon(sinogram_filtered, self.plot.image.shape[0],
                                                     self.params.emitter_angles, self.params.detector_quantity,
                                                     self.params.span, self.is_interactive, history_builder=self.history_builder)
        else:
            self.image_reconstructed = inverse_radon(self.sinogram, self.plot.image.shape[0],
                                                     self.params.emitter_angles, self.params.detector_quantity,
                                                     self.params.span, self.is_interactive, history_builder=self.history_builder)
        if not self.is_interactive:
            self.mse_error = get_mean_squared_error(self.plot.image, self.image_reconstructed)
        on_finish_task()


def generate_angles(theta):
    full_angle = np.pi * 2
    return [theta * i for i in range(int(np.ceil(full_angle / theta)))]


def get_mean_squared_error(original, reconstructed):
    org_copy = original - original.min()
    rec_copy = reconstructed - reconstructed.min()
    org_copy_max = org_copy.max()
    rec_copy_max = rec_copy.max()
    if rec_copy_max > 0 and org_copy_max > 0 and rec_copy_max is not org_copy_max:
        rec_copy /= (rec_copy_max / org_copy_max)
    dif = org_copy - rec_copy
    dif **= 2
    return dif.sum() / dif.size


def radon(image, emitter_angles, detector_quantity, span, is_interactive=False, history_builder=None):
    sinogram = np.zeros((len(emitter_angles), detector_quantity))
    h, w = image.shape
    detector_step = span / detector_quantity
    center = Point(w // 2, h // 2)
    base = Point(w // 2, 1)
    halfspan = span / 2.0
    for i, emitter_angle in enumerate(emitter_angles):
        source = base.get_coords(center, emitter_angle)
        detectors_angles = [emitter_angle + np.pi - halfspan + k * detector_step for k in range(detector_quantity)]
        detectors_positions = [base.get_coords(center, angle) for angle in detectors_angles]
        rays = []
        for detector in detectors_positions:
            path = bresenham.bresenham_indexes(source, detector)
            path_x_coords = path[:, 0]
            path_y_coords = path[:, 1]
            rays.append(image[path_x_coords, path_y_coords].sum())
        sinogram[i] = rays
        if is_interactive:
            history_builder(sinogram=sinogram, iteration=i)
    # sinogram = sinogram / np.amax(sinogram)
    return sinogram


def inverse_radon(sinogram, size, emitter_angles, detector_quantity, span, is_interactive=False, history_builder=None):
    height = width = size
    image = np.zeros((height, width))
    detector_step = span / detector_quantity
    w, h = height - 5, width - 5
    center = Point(int(w / 2), int(h / 2))
    base = Point(int(w / 2), 0)
    halfspan = span / 2.0
    for j, (sinogram_projection, emitter_angle) in enumerate(zip(sinogram, emitter_angles)):
        source = base.get_coords(center, emitter_angle)
        detectors_angles = [emitter_angle + np.pi - halfspan + k * detector_step for k in range(detector_quantity)]
        detectors_positions = [base.get_coords(center, angle) for angle in detectors_angles]
        for i, detector in enumerate(detectors_positions):
            path = bresenham.bresenham_indexes(source, detector)
            image[path[:, 0], path[:, 1]] += sinogram_projection[i]
        if is_interactive:
            history_builder(image_reconstructed=image, iteration=j)
    return np.array(image)


params = TomographParameters("C:\\Users\\nurcan\\Desktop\\tomograph-simulator-master\\a5.jpg", 1, 100, 180, "ramp")
