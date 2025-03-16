import numpy as np
class NoiseGenerator:
    def __init__(self, noise_type='gaussian', **kwargs):
        self.noise_type = noise_type
        self.params = kwargs

    def add_noise(self, image):
        if self.noise_type == 'gaussian':
            mean = self.params.get('mean', 0)
            sigma = self.params.get('sigma', 0.1)
            noise = np.random.normal(mean, sigma, image.shape)
            return image + noise

        elif self.noise_type == 'poisson':
            scale = self.params.get('scale', 1.0)
            noisy = np.random.poisson(image * scale) / float(scale)
            return noisy

        elif self.noise_type == 'salt_and_pepper':
            prob = self.params.get('prob', 0.05)
            noisy = np.copy(image)
            rnd = np.random.rand(*image.shape)
            noisy[rnd < prob / 2] = np.min(image)
            noisy[rnd > 1 - prob / 2] = np.max(image)
            return noisy

        else:
            raise ValueError("Unsupported noise type: {}".format(self.noise_type))