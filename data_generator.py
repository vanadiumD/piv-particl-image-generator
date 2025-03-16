# main.py
import os
import argparse
from data_generators.picture_generator import PictureGenerator
from data_generators.flow_generator import FlowGenerator
from data_generators.edge_generator import EdgeGenerator
from data_generators.noise_generator import NoiseGenerator
import config
from typing import Optional, List
from dataset import read_gen

def generate_single_image(index: int,
    data_path: str = 'default',
    noise_type: str = "none",
    mean: float = 0.0,
    sigma: float = 0.1,
    scale: float = 1.0,
    prob: float = 0.05,
    mask_formula0: Optional[List[str]] = None,
    mask_formula1: Optional[List[str]] = None,
    flow_formula: Optional[List[str]] = None,
    rho: Optional[float] = None,
    d: Optional[float] = None,
    mask_path: Optional[str] = None,
    flow_path: Optional[str] = None
    ):

    if data_path.lower() != 'default':
        config.data_path = data_path
    filename = os.path.join(config.data_path, f"{index:04d}")
    os.makedirs(filename, exist_ok=True)
    edge_gen = EdgeGenerator()


    if mask_path:
        # Load the mask from the provided path
        mask = read_gen(mask_path)
    else:
        if mask_formula0:
            mask0 = edge_gen.generate_mask(mask_formula0, showimg=False)
        
            if mask_formula1:
                mask1 = edge_gen.generate_mask(mask_formula1, showimg=False)
                mask = [mask0, mask1]
            else:
                mask = mask0
        else:
            mask = None

    if flow_path:
        # Load the flow data from the provided path
        u, v = read_gen(flow_path)
    else:
        flow_filename = os.path.join(config.data_path, f"{index:03d}", f"flow_{index}.h5")
        flow_gen = FlowGenerator(filename=flow_filename)

        u, v = flow_gen.generate_flow(
            flow_string=flow_formula,
            showimg=False,
            mask=mask
        )

    noise_gen = None
    if noise_type != "none":
        noise_gen = NoiseGenerator(
            noise_type=noise_type,
            mean=mean,
            sigma=sigma,
            scale=scale,
            prob=prob
        )

    pic_gen = PictureGenerator(config)
    pic_gen.generate_picture(
        u, v, index,
        d=d,
        rho=rho,
        noise_generator=noise_gen,
        mask=mask
    )


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PIV Data Generator")
    parser.add_argument("--data_path", type=str, default='default', help="File to save the dataset")
    parser.add_argument("--index", type=int, default=1000, help="The index of the dataset")
    parser.add_argument("--pixel", type=bool, default=True, help="use pixel or not")
    parser.add_argument("--noise_type", type=str, default="none", choices=["none", "gaussian", "poisson", "salt_and_pepper"], help="Type of noise to add")
    parser.add_argument("--mean", type=float, default=0.0, help="Mean for Gaussian noise")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma for Gaussian noise")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for Poisson noise")
    parser.add_argument("--prob", type=float, default=0.05, help="Probability for salt and pepper noise")
    parser.add_argument("--mask_formula0", type=str, nargs='+', default=["(x / config.pixel_x) ** 2 + (y / config.pixel_y) ** 2 - 1"], help="First mask boundary equations as a list of strings")
    parser.add_argument("--mask_formula1", type=str, nargs='+', default=[], help="Second mask boundary equations as a list of strings (optional)")
    parser.add_argument("--flow_formula", type=str, nargs=2, default=["3 * cos(2 * pi * x /( 250 - 0.3 * y)) * sin(2 * pi * y / (250 - 0.3 * y))", "3 * sin(2 * pi * x /( 250 - 0.3 * y)) * cos(2 * pi * y / (250 - 0.3 * y))"], help="Flow field equations as a list of two strings [u, v]")
    parser.add_argument("--rho", type=float, default=None, help="Particle density (rho), set to None to use default")
    parser.add_argument("--d", type=float, default=None, help="Particle size (d), set to None to use default")
    parser.add_argument("--mask_path", type=str, help="Path to an external mask file (optional)")
    parser.add_argument("--flow_path", type=str, help="Path to an external flow file (optional)")

    args = parser.parse_args()

    generate_single_image(
        index=args.index,
        data_path=args.data_path,
        noise_type=args.noise_type,
        mean=args.mean,
        sigma=args.sigma,
        scale=args.scale,
        prob=args.prob,
        mask_formula0=args.mask_formula0,
        mask_formula1=args.mask_formula1,
        flow_formula=args.flow_formula,
        rho=args.rho,
        d=args.d,
        mask_path=args.mask_path,
        flow_path=args.flow_path
    )