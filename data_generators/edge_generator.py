# edge_generator.py
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import cv2
from skimage import measure
import config
from parse import preprocess_expression

class EdgeGenerator(object):
    def __init__(self):
        self.contour = None
        self.mask = None
        # Create grid for contour generation
        x = np.linspace(0, config.pixel_x, config.pixel_x)
        y = np.linspace(0, config.pixel_y, config.pixel_y)
        self.X, self.Y = np.meshgrid(x, y)
        self.x, self.y = sp.symbols('x y')  # 定义符号变量

    def generate_contour(self, filename, method='implicity', function=None, t_range=None, pixel=True, showimg=False):
        """
        Generate contour and return contour position array.
        
        Parameters:
        - filename: File name to save the contour plot.
        - method: Contour generation method ('implicity' or 'parameter').
        - function: Implicit function f(x, y) or parametric function (fx(t), fy(t)).
        - t_range: Range for parameter t (used in 'parameter' method).
        - pixel: Whether to use pixel coordinates.
        - showimg: Whether to display the contour plot.

        Returns:
        - contours_data: List of contour position arrays.
        """
        fig, ax = plt.subplots()

        if t_range is None:
            t_range = [0, 2 * np.pi]

        if method == 'implicity':
            if function is None:
                Z = ((2 * self.X - config.pixel_x) / config.pixel_x) ** 2 + \
                    ((2 * self.Y - config.pixel_y) / config.pixel_y) ** 2 - 1
            elif callable(function):
                if pixel:
                    Z = function(self.X, self.Y)
                else:
                    x = self.X * config.ratio
                    y = self.Y * config.ratio * config.portrait
                    Z = function(x, y)
            else:
                raise ValueError("Implicit method requires a callable function f(x, y)")

            contours = measure.find_contours(Z, level=0)

        elif method == 'parameter':
            if not (isinstance(function, (list, tuple)) and len(function) == 2 and callable(function[0]) and callable(function[1])):
                raise ValueError("Parameter method requires a function tuple (fx, fy) where both are callable")

            t = np.linspace(t_range[0], t_range[1], 400)
            x = function[0](t)
            y = function[1](t)

            if not pixel:
                x *= self.config.scaling_factor
                y *= self.config.scaling_factor * self.config.portrait

            contours = [np.column_stack((x, y))]  # Format (x, y) for skimage

        else:
            raise ValueError(f"Unknown method '{method}' in contour generation function")       

        if showimg:
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], 'black')  # Plot contours
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.show()

        plt.savefig(filename)
        plt.close(fig)

        return contours
    
    def generate_mask(self, equation_strings, shape=None, showimg=False, pixel = True):
        """
        Generates a binary mask based on inequality constraints provided as strings.
        
        Parameters:
        equation_strings: list[str]  - A list of inequalities as strings (e.g., "x**2 + y**2 - 9").
        shape: tuple[int, int]       - The desired shape of the mask in (height, width). If None, defaults to config values.
        showimg: bool                - If True, the generated mask will be displayed using matplotlib. Default is False.

        Returns:
        np.ndarray                   - Binary mask (1 for valid regions, 0 otherwise).
        """
        # Determine shape based on input or use config defaults
        if shape is not None:
           width, height = shape
        else:
           width, height = config.pixel_x, config.pixel_y

        y_vals = np.linspace(0, height - 1, height)
        x_vals = np.linspace(0, width - 1, width)
        X, Y = np.meshgrid(x_vals, y_vals)

        mask = np.ones((height, width), dtype=bool)

        if not pixel:
            X, Y = X * self.scale, Y * self.scale

        # Apply each inequality constraint
        for eq_str in equation_strings:
            eq_str = preprocess_expression(eq_str)
            eq = sp.sympify(eq_str, {'x': self.x, 'y': self.y, 'config': config})  # Parse the equation string
            func = sp.lambdify((self.x, self.y), eq, 'numpy')    # Convert to numpy function
            mask &= (func(X, Y) < 0)  # Apply inequality constraint

        mask = mask.astype(np.uint8)

        # Display mask if showimg is True
        if showimg:
            plt.imshow(mask, cmap="gray", origin="lower")
            plt.colorbar(label="Mask Value")
            plt.title("Generated Mask")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.show()

        return mask
    
    def contour2mask(self, contours, shape):
        """
        Convert contour to a binary mask.

        Parameters:
        - contours: List of contour position arrays [(N, 2), ...] in (y, x) format.
        - shape: Tuple (height, width) defining mask size.

        Returns:
        - mask: Binary mask with filled contour regions.
        """
        self.mask = np.zeros(shape, dtype=np.uint8)
        contours_int = [contour.astype(np.int32) for contour in contours]  # Ensure integer coordinates
        cv2.fillPoly(self.mask, contours_int, 255)  # Fill contour with white (255)
        return self.mask

    def mask2contour(self, mask):
        """
        Extract contours from a binary mask.

        Parameters:
        - mask: Binary mask (numpy array).

        Returns:
        - contours: List of contour position arrays [(N, 2), ...] in (y, x) format.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [contour[:, 0, :] for contour in contours]  

    def add_contour(self, image, line_type='solid', color=(0, 255, 0), thickness=2):
        if self.contour is None:
            return image

        if line_type == 'solid':
            cv2.drawContours(image, [self.contour], -1, color, thickness)
        elif line_type == 'dashed':
            for i in range(0, len(self.contour), 10):
                start_point = tuple(self.contour[i][0])
                end_point = tuple(self.contour[(i + 5) % len(self.contour)][0])
                cv2.line(image, start_point, end_point, color, thickness)
        else:
            raise ValueError("line_type must be 'solid' or 'dashed'.")

        return image
