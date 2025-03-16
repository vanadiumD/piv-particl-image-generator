import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import config
class PictureGenerator(object):
    def __init__(self, config):
        self.pixel_x = config.pixel_x  
        self.pixel_y = config.pixel_y  
        self.rho = config.rho
        self.dt = config.dt
        self.scaling_factor = config.scaling_factor
        self.data_path = config.data_path
        self.showimg = config.showimg
        self.ratio = getattr(config, 'ratio', 1)
        self.d = config.d

        x = np.linspace(0, self.pixel_x - 1, self.pixel_x)
        y = np.linspace(0, self.pixel_y - 1, self.pixel_y)
        self.X, self.Y = np.meshgrid(x, y)

    def add_gaussian_spots(self, X, Y, particle_x, particle_y, batch_size=1000, truncation_distance=80, use_gpu=True):
        assert batch_size > 0, "batch_size must be a positive integer"
        assert self.d > 0, "d must be greater than zero"
        
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {device.type.upper()}")
        
        # Convert inputs to tensors and move to the selected device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
        particle_x_tensor = torch.tensor(particle_x, dtype=torch.float32, device=device)
        particle_y_tensor = torch.tensor(particle_y, dtype=torch.float32, device=device)
        
        result = torch.zeros((self.pixel_y, self.pixel_x), dtype=torch.float32, device=device)
        N_particles = particle_y_tensor.shape[0]

        truncation_threshold_squared = truncation_distance ** 2

        X_expanded = X_tensor[..., None]
        Y_expanded = Y_tensor[..., None]
        factor = 8 / (self.d ** 2)
        
        for i in range(0, N_particles, batch_size):
            end = min(i + batch_size, N_particles)
            batch_particle_x = particle_x_tensor[i:end]
            batch_particle_y = particle_y_tensor[i:end]
            
            # Compute differences between grid points and particle positions
            diff_x = X_expanded - batch_particle_x[None, None, :]
            diff_y = Y_expanded - batch_particle_y[None, None, :]
            
            dist_squared = diff_x**2 + diff_y**2
            
            # Create mask for valid regions within truncation threshold
            valid_mask = dist_squared <= truncation_threshold_squared
            
            # Compute Gaussian values based on valid mask
            gauss_values = torch.where(valid_mask, torch.exp(-dist_squared * factor), torch.zeros_like(dist_squared))
            
            result += gauss_values.sum(dim=-1)

        return result.cpu().numpy() if device.type == 'cuda' else result.numpy()

    
    def compute_displacement(self, X0, Y0, u ,v):
        Y0floor = np.clip(np.floor(Y0).astype(int), 0, self.pixel_y - 1)
        X0floor = np.clip(np.floor(X0).astype(int), 0, self.pixel_x - 1)
        Y0ceil = np.clip(np.ceil(Y0).astype(int), 0, self.pixel_y - 1)
        X0ceil = np.clip(np.ceil(X0).astype(int), 0, self.pixel_x - 1)
        
        dY0floor = Y0 - Y0floor
        dX0floor = X0 - X0floor
        dY0ceil = Y0ceil - Y0
        dX0ceil = X0ceil - X0
        return (
            X0 + (dX0floor * dY0floor * u[Y0ceil, X0ceil] +
                dX0ceil * dY0floor * u[Y0ceil, X0floor] +
                dX0floor * dY0ceil * u[Y0floor, X0ceil] +
                dX0ceil * dY0ceil * u[Y0floor, X0floor]) * self.dt * self.scaling_factor, 
            Y0 + (dX0floor * dY0floor * v[Y0ceil, X0ceil] +
                dX0ceil * dY0floor * v[Y0ceil, X0floor] +
                dX0floor * dY0ceil * v[Y0floor, X0ceil] +
                dX0ceil * dY0ceil * v[Y0floor, X0floor]) * self.dt * self.scaling_factor
        )
    

    def apply_mask(self, X, Y, mask):
        threshold = 0.5  
        X_int = np.clip(np.round(X).astype(int), 0, self.pixel_x - 1)
        Y_int = np.clip(np.round(Y).astype(int), 0, self.pixel_y - 1)
        valid = mask[Y_int, X_int] > threshold
        return X[valid], Y[valid]
            
    def generate_picture(self, u, v, picture_idx, d = None, rho = None, noise_generator=None, mask=None, contour=None):
        print("Calculating particle positions")
        NX = int(self.pixel_x * self.rho / self.d)
        NY = int(self.pixel_y * self.rho / self.d)
        N = NX * NY

        # 交换 x0, y0 顺序
        y0 = np.linspace(0, self.pixel_y, NY)
        x0 = np.linspace(0, self.pixel_x, NX)
        X0, Y0 = np.meshgrid(x0, y0)
        Y0 = Y0.flatten() + np.random.normal(0, self.d, N)
        X0 = X0.flatten() + np.random.normal(0, self.d, N)
        if d is None:
            d = config.d

        if rho is None:
            rho = config.rho


        if mask is None or isinstance(mask, list):
            X1, Y1 = self.compute_displacement(X0, Y0, u, v)  # 计算 X1, Y1
            
            if mask is not None:
                X0, Y0 = self.apply_mask(X0, Y0, mask[0])
                X1, Y1 = self.apply_mask(X1, Y1, mask[1])
                print(f'generating mask in picture pairs, total points:{np.size(X0) + np.size(X1)}')
            else:
                print(f"generaing pictures without mask, total points:{np.size(X0) + np.size(X1)}")
        else:
            X0, Y0 = self.apply_mask(X0, Y0, mask)  # 仅对 X0, Y0 进行掩码
            X1, Y1 = self.compute_displacement(X0, Y0, u, v)  # 计算 X1, Y1
            print(f'generating mask in the first picture, total points:{np.size(X0) + np.size(X1)}')

        print("Generating grayscale image pairs")
        gray_1 = self.add_gaussian_spots(self.X, self.Y, X0, Y0)
        gray_2 = self.add_gaussian_spots(self.X, self.Y, X1, Y1)

        # 归一化灰度值
        maxgray = max(np.max(gray_1), np.max(gray_2))
        if maxgray > 0:
            gray_1 = gray_1 / maxgray * 255
            gray_2 = gray_2 / maxgray * 255

        # 加入噪声
        if noise_generator is not None:
            gray_1 = noise_generator.add_noise(gray_1)
            gray_2 = noise_generator.add_noise(gray_2)

        # 保存图片
        picture_1_path = os.path.join(self.data_path, f"{picture_idx:04d}", f"picture_1_{picture_idx}.png")
        picture_2_path = os.path.join(self.data_path, f"{picture_idx:04d}", f"picture_2_{picture_idx}.png")
        cv2.imwrite(picture_1_path, gray_1)
        cv2.imwrite(picture_2_path, gray_2)

        # 显示图片
        if self.showimg:
            plt.subplot(1, 2, 1)
            plt.imshow(gray_1, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(gray_2, cmap='gray')
            plt.show()