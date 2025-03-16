import numpy as np
import random
import config
from concurrent.futures import ProcessPoolExecutor
import data_generator
import logging
import sys
# 日志配置
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s | %(processName)-10s | %(levelname)-7s | %(message)s'
    )

    # 清除已有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 控制台输出（所有消息）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（追加模式）
    file_handler = logging.FileHandler('batch_generator.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 子进程日志初始化函数
def init_child_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s | %(processName)-10s | %(levelname)-7s | %(message)s'
    )

    # 清除可能继承的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 添加与主进程一致的处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('simulation.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def generate_varied_parameters(base_index: int) -> dict:

    logger = logging.getLogger(__name__)
    """生成带随机波动的参数"""
    # 流场公式库
    flow_keys = [
        "default", 
        "uniform", 
        "vortex", 
        "source_sink", 
        "dipole", "shear",
        "stagnation", 
        "poiseuille",
        "power_law"
        ]
    
    flow_templates = {
        "default": [
            "{A} * cos({B} * pi * x / (250 - {C} * y)) * sin({D} * pi * y / (250 - {E} * y))",
            "{A} * sin({B} * pi * x / (250 - {C} * y)) * cos({D} * pi * y / (250 - {E} * y))"
        ],
        "uniform": [  # 均匀流
            "{U}",
            "{V}"
        ],
        "vortex": [  # 旋涡流
            "-{G} * (y - {Yc}) / ((x - {Xc})**2 + (y - {Yc})**2 + {eps})",
            "{G} * (x - {Xc}) / ((x - {Xc})**2 + (y - {Yc})**2 + {eps})"
        ],
        "source_sink": [  # 源/汇流（笛卡尔坐标形式）
        "{Q} * (x - {Xc}) / ((x - {Xc})**2 + (y - {Yc})**2 + {eps})",
        "{Q} * (y - {Yc}) / ((x - {Xc})**2 + (y - {Yc})**2 + {eps})"
        ],
        "dipole": [  # 偶极子流
            "-{K} * ((x - {Xc})**2 - (y - {Yc})**2) / ((x - {Xc})**2 + (y - {Yc})**2 + {eps})**2",
            "-2*{K} * (x - {Xc}) * (y - {Yc}) / ((x - {Xc})**2 + (y - {Yc})**2 + {eps})**2"
        ],
        "shear": [  # 线性剪切流
            "{Gamma} * (y - {Y0})",
            "0"
        ],
        "stagnation": [  # 驻点流动
            "{Alpha} * (x - {Xc})",
            "-{Alpha} * (y - {Yc})"
        ],

        "poiseuille": [  # 泊肃叶抛物线流
            "{Umax} * (1 - ((y - {Yc})/{h})**2)",
            "0"
        ],
        "power_law": [  # 幂律流体抛物线流
        "{Umax} * (1 - (abs(y - {Yc})/{h})**(({n}+1)/{n}))",
        "0"
        ],
    }
    
    # 随机选择一个流场
    flow_key = random.choice(flow_keys)
    selected_flow = flow_templates[flow_key]
    
    # 仅生成所需参数
    flow_params = {}
    if flow_key == "default":
        flow_params.update({
            'A': round(np.random.uniform(2, 5), 4),
            'B': round(np.random.uniform(1, 3.5), 4),
            'C': round(np.random.uniform(0.2, 0.4), 4),
            'D': round(np.random.uniform(1, 3.5), 4),
            'E': round(np.random.uniform(0.2, 0.4), 4)
        })

    elif flow_key == "uniform":
        flow_params.update({
            'U': round(np.random.uniform(-20, 20),4),
            'V': round(np.random.uniform(-20, 20),4)
        })

    elif flow_key == "vortex":
        flow_params.update({
            'Xc': round(np.random.uniform(0, 200),4),
            'Yc': round(np.random.uniform(0, 200),4),
            'G': round(np.random.uniform(80, 200),4),
            'eps': round(np.random.uniform(0.01, 0.1),4)
        })

    elif flow_key == "source_sink":
        flow_params.update({
            'Xc': round(np.random.uniform(0.2*config.pixel_x, 0.8*config.pixel_x), 4),  # 源位置在中心区域
            'Yc': round(np.random.uniform(0.2*config.pixel_y, 0.8*config.pixel_y), 4),
            'Q': round(np.random.uniform(-150, 150), 4),  # Q>0为源，Q<0为汇
            'eps': round(np.random.uniform(1e-3, 1.0), 4)  # 避免奇点
        })

    elif flow_key == "dipole":
        flow_params.update({
            'Xc': round(np.random.uniform(0.3*config.pixel_x, 0.7*config.pixel_x), 4),
            'Yc': round(np.random.uniform(0.3*config.pixel_y, 0.7*config.pixel_y), 4),
            'K': round(np.random.uniform(80, 2000), 4),  # 偶极子强度
            'eps': round(np.random.uniform(1e-3, 1.0), 4)
        })

    elif flow_key == "shear":
        flow_params.update({
            'Gamma': round(np.random.uniform(-0.1, 0.1), 4),  # 剪切率控制
            'Y0': round(np.random.uniform(0.4*config.pixel_y, 0.6*config.pixel_y), 4)  # 剪切中心线
        })

    elif flow_key == "stagnation":
        flow_params.update({
            'Xc': round(np.random.uniform(0.4*config.pixel_x, 0.6*config.pixel_x), 4),  # 驻点位置
            'Yc': round(np.random.uniform(0.4*config.pixel_y, 0.6*config.pixel_y), 4),
            'Alpha': round(np.random.uniform(0.01, 0.1), 4)  # 应变率参数
        })

    elif flow_key == "poiseuille":
        flow_params.update({
            'Umax': round(np.random.uniform(0.01, 5), 4),   # 中心线速度
            'Yc': round(np.random.uniform(0.4*config.pixel_y, 0.6*config.pixel_y), 4), # 通道中心
            'h': round(np.random.uniform(0.3*config.pixel_y, 0.6*config.pixel_y), 4)   # 半高度
        })

    elif flow_key == "power_law":
        flow_params.update({
        'Umax': round(np.random.uniform(0.01, 5), 4),          # 中心线最大速度
        'Yc': round(np.random.uniform(0.4*config.pixel_y, 0.6*config.pixel_y), 4),  # 流动中心位置
        'h': round(np.random.uniform(0.3*config.pixel_y, 0.6*config.pixel_y), 4),   # 半高度（确保>0）
        'n': round(np.random.uniform(0.3, 1.5), 4)             # 幂律指数（0.3假塑性，1.5膨胀性）
    })

    flow_formula = [eq.format(**flow_params) for eq in selected_flow]
    logger.info(
        "Generated flow for index %d:\n u = %s\n v = %s", 
        base_index, flow_formula[0], flow_formula[1]
    )

    # 粒子参数波动
    d = np.random.uniform(2, 6)
    rho = np.random.uniform(0.002, 0.7)

    # 噪声参数波动
    noise_type = random.choice(["none", "gaussian", "poisson", "salt_and_pepper"])
    mean = sigma = scale = prob = None  # 初始化默认值
    if noise_type == "gaussian":
        mean = np.random.uniform(-0.1, 0.1)
        sigma = np.random.uniform(0.05, 0.15)
    elif noise_type == "poisson":
        scale = np.random.uniform(0.5, 2.0)
    elif noise_type == "salt_and_pepper":
        prob = np.random.uniform(0.01, 0.1)

    # 掩膜公式库
    mask_keys = ["ellipse", "log"]
    mask_templates = {
        "ellipse": "(abs((x - {x0})/config.pixel_x * {F} * {H})) ** {n} + (abs((y - {y0})/config.pixel_y * {G} * {H})) ** {n} - 1",
        "log": "log(abs((x - {x0}) / config.pixel_x * {F} * {H}) + 1) + log(abs((y - {y0}) / config.pixel_y * {G} * {H}) + 1) - log(2)"
    }

    # 随机选择一个掩膜
    mask_key = random.choice(mask_keys)
    selected_mask = mask_templates[mask_key]
    mask_params = {} 
    # 仅生成所需参数
    if mask_key == "ellipse":
        mask_params.update({
            'F': round(np.random.uniform(0.5, 1.2),4),
            'G': round(np.random.uniform(0.5, 1.2),4),
            'x0': round(np.random.uniform(0, 250),4),
            'y0': round(np.random.uniform(0, 250),4),
            'n': random.randint(1, 3),  # n 为随机正整数
            'H': 1
        })
    elif mask_key == "log":
        mask_params.update(
            {'F': round(np.random.uniform(0.5, 1.2),4),
            'G': round(np.random.uniform(0.5, 1.2),4),
            'x0': round(np.random.uniform(0, 250),4),
            'y0': round(np.random.uniform(0, 250),4),
            'H': 1          
            })
        
    mask0 = [selected_mask.format(**mask_params)]
    logger.info("Generated mask0 for index %d: %s", base_index, mask0)
    if 0.85 < np.random.uniform(0, 1):
        mask_params.update(
            {'H': 1 + round(np.random.uniform(-0.01, 0.01),4)
            })
        mask1 = [selected_mask.format(**mask_params)]
        logger.info("Generated mask1 for index %d: %s", base_index, mask1)


    else:
        mask1 = None
    return {
        "index": base_index,
        "mask_formula0": mask0,
        "mask_formula1": mask1,
        "flow_formula": flow_formula,
        "d" : d,
        "rho" : rho,
        "noise_type": noise_type,
        "mean": mean,
        "sigma": sigma,
        "scale": scale,
        "prob": prob, 
        }

def generate_batch(start_index: int, num_images: int, parallel: bool = True, chunk_size: int = 5):
    """批量生成入口函数"""
    logger = logging.getLogger(__name__)
    logger.info("START batch: %d images from index %d (parallel=%s)", 
               num_images, start_index, parallel)
    
    try:
        if parallel:
            logger.info("Parallel mode | Chunk size: %d | Total range: %d-%d", 
                       chunk_size, start_index, start_index + num_images - 1)
            
            with ProcessPoolExecutor(initializer=init_child_logging) as executor:
                for i in range(0, num_images, chunk_size):
                    current_chunk = min(chunk_size, num_images - i)
                    start_range = start_index + i
                    end_range = start_index + i + current_chunk - 1
                    
                    logger.info("Processing batch %d/%d | Indexes %d-%d", 
                              (i//chunk_size)+1, (num_images//chunk_size)+1, 
                              start_range, end_range)
                    
                    params_list = [generate_varied_parameters(start_index + i + j) for j in range(current_chunk)]
                    executor.map(lambda p: data_generator.generate_single_image(**p), params_list)
                    
                    logger.info("Completed batch %d/%d | Indexes %d-%d", 
                              (i//chunk_size)+1, (num_images//chunk_size)+1, 
                              start_range, end_range)
        else:
            logger.info("Sequential mode started")
            for i in range(num_images):
                current_index = start_index + i
                logger.info("Processing image %d/%d (index %d)", 
                           i+1, num_images, current_index)
                try:
                    params = generate_varied_parameters(current_index)
                    data_generator.generate_single_image(**params)
                    logger.info("Success: index %d", current_index)
                except Exception as e:
                    logger.error("FAILED index %d | Error: %s", current_index, str(e), exc_info=True)
        
        logger.info("FINISHED batch: %d-%d", 
                   start_index, start_index + num_images - 1)
    except Exception as e:
        logger.error("CRITICAL ERROR: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    setup_logging()
    try:
        generate_batch(start_index=0, num_images=1000, parallel=False)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Process interrupted by user")
        sys.exit(1)