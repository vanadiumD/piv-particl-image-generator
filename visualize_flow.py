import numpy as np
import matplotlib.pyplot as plt
import h5py
import config

class VisualizeFlow(object):

    def __init__(self, load_filename=None, save_dir= None, flow_data=None):
        self.load_filename = load_filename
        self.save_dir = save_dir
        self.flow_data = flow_data


        x = np.linspace(0, config.pixel_x - 1, config.pixel_x)
        y = np.linspace(0, config.pixel_y - 1, config.pixel_y)
        self.X, self.Y = np.meshgrid(x, y)
    def load_flow_data(self, flow_filename):

        self.load_filename = flow_filename
        with h5py.File(flow_filename, 'r') as f:
            u = np.array(f['u'])
            v = np.array(f['v'])  
            self.flow_data = (u, v)
        return u, v

    def streamplot(self, density=0.5):
        u, v = self.flow_data
        
        plt.figure(figsize=(8, 6))
        plt.streamplot(self.X, self.Y, u, v, density=density, color='b', linewidth=1, arrowstyle='->')
        plt.title("Streamplot")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks(np.arange(0, self.X.shape[1], step=20))
        plt.yticks(np.arange(0, self.Y.shape[0], step=20))
        plt.grid(True)
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/streamplot.png")
        plt.show()

    def quiver(self, scale=10, density=0.1):
        u, v = self.flow_data
        # 根据 density 计算下采样步长
        step = max(1, int(1 / density))
                
        # 对流场数据进行下采样
        u_sparse = u[::step, ::step]
        v_sparse = v[::step, ::step]
        # 对完整坐标网格按照相同步长下采样，保证物理位置不被压缩
        X_sparse = self.X[::step, ::step]
        Y_sparse = self.Y[::step, ::step]
        
        plt.figure(figsize=(8, 6))
        plt.quiver(X_sparse, Y_sparse, u_sparse, v_sparse, scale=scale, pivot='mid', color='r')
        plt.title("Quiver Plot")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks(np.arange(0, self.X.shape[1], step=20))
        plt.yticks(np.arange(0, self.Y.shape[0], step=20))
        plt.grid(True)
        
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/quiver_plot.png")
        plt.show()

    def contourplot(self, levels=10):
        u, v = self.flow_data
        magnitude = np.sqrt(u**2 + v**2)  # 计算速度的大小
        
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(self.X, self.Y, magnitude, levels=levels, cmap='viridis')
        plt.colorbar(contour)
        plt.title("Contour Plot")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks(np.arange(0, self.X.shape[1], step=20))
        plt.yticks(np.arange(0, self.Y.shape[0], step=20))
        plt.grid(True)
        
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/contour_plot.png")
        plt.show()

    def colormap(self):
        u, v = self.flow_data
        magnitude = np.sqrt(u**2 + v**2)  # 计算速度的大小
    
        
        plt.figure(figsize=(8, 6))
        cmap_plot = plt.pcolormesh(self.X, self.Y, magnitude, shading='auto', cmap='viridis')
        plt.colorbar(cmap_plot)
        plt.title("Flow Magnitude Colormap")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks(np.arange(0, self.X.shape[1], step=20))
        plt.yticks(np.arange(0, self.Y.shape[0], step=20))
        plt.grid(True)
        
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/colormap.png")
        plt.show()

    def sliceplot(self, slice_axis='x', slice_index=None):
        u, v = self.flow_data
        
        # 根据切片轴（x或y）提取流场数据的切片
        if slice_axis == 'x':
            slice_data = u[:, slice_index]  # 获取第 slice_index 列
            title = f'Slice Plot at X={slice_index}'
        elif slice_axis == 'y':
            slice_data = u[slice_index, :]  # 获取第 slice_index 行
            title = f'Slice Plot at Y={slice_index}'
        else:
            raise ValueError("slice_axis must be 'x' or 'y'")
        
        plt.figure(figsize=(8, 6))
        plt.plot(slice_data)
        plt.title(title)
        plt.xlabel(f'{slice_axis}-index')
        plt.ylabel('Velocity')
        
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/slice_plot_{slice_axis}_{slice_index}.png")
        plt.show()

    def save_flow(self, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('u', data=self.flow_data[0])
            f.create_dataset('v', data=self.flow_data[1])