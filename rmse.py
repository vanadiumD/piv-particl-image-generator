from dataset import read_gen
import numpy as np
import h5py
interp_method = 'nearest'

def compute_rmse(exp_file, true_file):

    if exp_file is None or true_file is None:
        raise ValueError(f"Failed to read {exp_file} or {true_file}.")
    # 读取实验和真实数据
    ue, ve = read_gen(exp_file)
    ur, vr = read_gen(true_file)

    has_nan_or_inf = False
    if np.isnan(ue).any() or np.isnan(ve).any() or np.isnan(ur).any() or np.isnan(vr).any():
        print(f"Warning: NaN detected in {exp_file} or {true_file}.")
        has_nan_or_inf = True
    
    if np.isinf(ue).any() or np.isinf(ve).any() or np.isinf(ur).any() or np.isinf(vr).any():
        print(f"Warning: Inf detected in {exp_file} or {true_file}.")
        has_nan_or_inf = True

    if has_nan_or_inf:
        valid_mask = ~np.isnan(ue) & ~np.isnan(ve) & ~np.isnan(ur) & ~np.isnan(vr)  # 排除 NaN
        valid_mask &= ~np.isinf(ue) & ~np.isinf(ve) & ~np.isinf(ur) & ~np.isinf(vr)  # 排除 Inf

        # 只保留有效的数据
        ue, ve = ue[valid_mask], ve[valid_mask]
        ur, vr = ur[valid_mask], vr[valid_mask]
    
    # 计算SEM
    squr = np.sum(np.square(ue - ur) + np.square(ve - vr))
    rmse = np.sqrt(squr / np.size(ue))
    mean = np.sqrt(np.average(squr))
    relative_rmse = rmse / mean
    return (rmse, relative_rmse)

if __name__ == "__main__":
    indices = range(540, 545)  # 这里假设有 10 组数据，可调整
    output_h5 = 'cross_correlation/rmse_results.h5'

with h5py.File(output_h5, 'w') as f:
    rmse_list = []
    r_rmse_list = []

    for idx in indices:
        exp_file = f"cross_correlation/exp1_{interp_method}_{idx}.h5"
        true_file = f"data/flow_{idx}.h5"
        
        try:
            rmse, relative_rmse = compute_rmse(exp_file, true_file)
            rmse_list.append(rmse)
            r_rmse_list.append(relative_rmse)
            print(f"idx: {idx}, rmse: {rmse}, r_rmse: {relative_rmse}")
        except ValueError as e:
            print(f"Error for idx {idx}: {e}")
            rmse_list.append(np.nan)  # 记录 NaN 以便识别错误
            r_rmse_list.append(np.nan)

    # 将数据存入 HDF5
    f.create_dataset('indices', data=np.array(list(indices)))
    f.create_dataset('rmse', data=np.array(rmse_list))
    f.create_dataset('relative_rmse', data=np.array(r_rmse_list))