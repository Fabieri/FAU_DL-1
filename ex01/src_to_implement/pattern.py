import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# might be useful: np.tile(), np.arange(), np.zeros(), np.ones(), np.concatenate() and np.expand dims()


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if self.resolution % (2 * self.tile_size) is not 0:
            return
        # self.output = np.concatenate((np.concatenate((np.ones((self.tile_size, self.tile_size)),
        #                         np.zeros((self.tile_size, self.tile_size)))), np.concatenate((np.zeros((self.tile_size, self.tile_size)),
        #                         np.ones((self.tile_size, self.tile_size))))), axis=1)

        self.output = np.tile(np.concatenate((np.concatenate((np.zeros((self.tile_size, self.tile_size)),
                                        np.ones((self.tile_size, self.tile_size)))), np.concatenate((np.ones((self.tile_size, self.tile_size)),
                                        np.zeros((self.tile_size, self.tile_size))))), axis=1),
                (int(self.resolution/(2 * self.tile_size)), int(self.resolution/(2 * self.tile_size))))

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.x_pos = position[0]
        self.y_pos = position[1]
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution))
        x, y = np.ogrid[-self.radius: self.radius, -self.radius: self.radius]
        index = x ** 2 + y ** 2 <= self.radius ** 2
        self.output[self.y_pos - self.radius:self.y_pos + self.radius, self.x_pos - self.radius:self.x_pos + self.radius][index] = 1
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        R = np.arange(0.0, 1.0, 1 / self.resolution)
        G = np.transpose(np.atleast_2d(np.arange(0.0, 1.0, 1 / self.resolution)))
        B = np.flip(np.arange(0.0, 1.0, 1 / self.resolution))

        self.output = np.zeros((self.resolution, self.resolution, 3))
        self.output[..., 0] = R
        self.output[..., 1] = G
        self.output[..., 2] = B
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()
