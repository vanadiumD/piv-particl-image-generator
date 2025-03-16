import h5py
import numpy as np
import torch.utils.data as data
from os.path import splitext
import imageio.v2 as imageio

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    
    if ext in ['.png', '.jpeg', '.ppm', '.jpg']:
        im = imageio.imread(file_name)
        
        if im.ndim == 2:  # 灰度图 (H, W)
            im = np.expand_dims(im, axis=-1)  # (H, W, 1)
        elif im.ndim == 3 and im.shape[2] > 3:  
            im = im[:, :, :3]  # 取前三通道
        
        return im.astype(np.float32)  # 统一转换为 float32 以适应后续 PyTorch 处理

    elif ext in ['.bin', '.raw']:
        return np.load(file_name).astype(np.float32)

    elif ext == '.h5':
        with h5py.File(file_name, 'r') as f:
            u = f['u'][:].astype(np.float32)  # 读取 u 分量
            v = f['v'][:].astype(np.float32)  # 读取 v 分量
            return (u, v)  # (H, W, 2)

    return None


def writeFlow(filename,uv,v=None):
    """ Write optical flow to file for a 2D flow field.
    
    Parameters:
    filename: str, path to save the flow data.
    uv: 2D numpy array (height x width x 2) representing the flow field with u (horizontal) and v (vertical) components.
    v: 2D numpy array (height x width), optional. If provided, it is used as the vertical (v) component. 
       If not provided, the second channel of `uv` is assumed to be the vertical component.

    """
    nBands = 2  # Optical flow has two components: u and v.

    if v is None:
        # If 'v' is not provided, assume 'uv' contains both 'u' and 'v' in its third dimension.
        assert(uv.ndim == 3)  # uv should be 3D: height x width x 2
        assert(uv.shape[2] == 2)  # The third dimension should have a size of 2 (u and v)
        u = uv[:, :, 0]  # Extract the u component (horizontal)
        v = uv[:, :, 1]  # Extract the v component (vertical)
    else:
        # If 'v' is provided, assume 'uv' is just the 'u' component.
        assert (uv.ndim == 2)
        u = uv 

    # Ensure the dimensions of u and v match
    assert(u.shape == v.shape)

    height, width = u.shape  # Get the dimensions of the flow field
        # Convert height and width to int32 format
    height = np.array(height, dtype=np.int32)
    width = np.array(width, dtype=np.int32)

    # Open the file to write binary data
    with h5py.File(filename, 'w') as f:
        # Write the header (magic tag and dimensions)

       # Create datasets for u and v components
        f.create_dataset('u', data=u.astype(np.float32))  # Store horizontal flow (u)
        f.create_dataset('v', data=v.astype(np.float32))  # Store vertical flow (v)
        
        # Optionally store the dimensions as metadata
        f.attrs['height'] = height
        f.attrs['width'] = width
        
        f.attrs['flow_format'] = '2D Optical Flow' 
