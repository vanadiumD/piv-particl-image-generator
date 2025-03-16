# flow_generator.py
import numpy as np
import sympy as sp
import config
from visualize_flow import VisualizeFlow
from parse import preprocess_expression

class FlowGenerator(object):
    def __init__(self, filename='.'):
        self.filename = filename
        # Create grid for flow field
        x = np.linspace(0, config.pixel_x - 1, config.pixel_x)
        y = np.linspace(0, config.pixel_y - 1, config.pixel_y)
        self.X, self.Y = np.meshgrid(x, y)
        # Initialize flow arrays (shape: height x width)
        self.u = np.zeros((config.pixel_y, config.pixel_x))
        self.v = np.zeros((config.pixel_y, config.pixel_x))
        self.scale = config.scaling_factor

    def parse_flow_string(self, flow_string, pixel=True):
        """ Parse flow equations and compute velocity fields. """
        x_sym, y_sym = sp.symbols('x y')

        if isinstance(flow_string, list) and len(flow_string) == 2:
            flow_string = [preprocess_expression(s) for s in flow_string]
            u_expr = sp.sympify(flow_string[0])
            v_expr = sp.sympify(flow_string[1])

            u_func = sp.lambdify((x_sym, y_sym), u_expr, 'numpy')
            v_func = sp.lambdify((x_sym, y_sym), v_expr, 'numpy')

            if pixel:
                X_data, Y_data = self.X, self.Y
            else:
                X_data = self.X * self.scale
                Y_data = self.Y * self.scale

            self.u = u_func(X_data, Y_data)
            self.v = v_func(X_data, Y_data)

            return self.u, self.v  

        else:
            raise ValueError("flow_string must be a list of two strings representing [u, v]")

    def generate_flow(self, flow_string=None, filename=None, mask=None, showimg=False, pixel=True):
        # Parse the flow string if provided
        if flow_string:
            self.parse_flow_string(flow_string, pixel)

        # Simulate constant flow if no flow string is provided
        if flow_string is None:
            self.u[:, :] = 0
            self.v[:, :] = 5
        
        if filename is not None:
            self.filename = filename
        
        # Applying mask
        if mask is None:
            print("Generating flow without mask")
        elif isinstance(mask, list):
            print("Generating flow with mask pairs")
            combined_mask = np.logical_or.reduce(mask)  # Take union of multiple masks
            self.u = self.u * combined_mask
            self.v = self.v * combined_mask
        elif isinstance(mask, np.ndarray):
            print("Generating flow with single mask")
            self.u = self.u * mask
            self.v = self.v * mask
        else:
            raise TypeError("mask must be None, a NumPy array, or a list of NumPy arrays")
        
        # Visualizing and saving the flow field
        img = VisualizeFlow(flow_data=(self.u, self.v))
        img.save_flow(self.filename)
        
        if showimg:
            img.streamplot()

        return self.u, self.v