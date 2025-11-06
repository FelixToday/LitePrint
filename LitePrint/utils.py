# -*- coding: utf-8 -*-

# @Author : 李先军

# @Time : 2025/8/30 下午4:37
from torch.utils.data import DataLoader

from .baseline import *
from .model_explore import get_model
from .dataset import CountDataset as EDdataset
import numpy as np
def aultdict(**kwargs):
    return kwargs

def get_model_and_dataloader(X1,y1, X2,y2, num_classes, config:dict, args:dict, num_workers=8):
    """
    根据配置创建模型和数据加载器

    参数:
        X1: 第一个数据集的特征数据
        y1: 第一个数据集的标签数据
        X2: 第二个数据集的特征数据
        y2: 第二个数据集的标签数据
        num_classes: 分类任务的类别数量
        config: 配置字典，包含以下键:
            - model: 模型名称，支持["AWF", "DF", "TF", "TMWF", "ARES", "TikTok", "VarCNN", "RF", "MultiTabRF", "CountMamba", "ExploreModel"]
            - seq_len: 序列长度
            - batch_size: 批次大小
            - maximum_cell_number: 最大单元格数量（用于非传统模型）
            - max_matrix_len: 最大矩阵长度（用于非传统模型）
            - log_transform: 是否进行对数变换（用于非传统模型）
            - time_interval_threshold: 时间间隔阈值（用于非传统模型）
            - drop_extra_time: 是否丢弃额外时间（用于非传统模型）
            - minimum_packet_number: 最小数据包数量（用于非传统模型）
            - drop_path_rate: Mamba模型的drop路径率
            - depth: Mamba模型的深度
            - embed_dim: Mamba模型的嵌入维度
            - early_stage: Mamba模型的早期阶段设置
            - fine_predict: Mamba模型的精细预测设置
        args: 参数字典，包含以下键:
            - load_ratio: 数据加载比例（用于传统模型）
            - TAM_type: TAM类型（用于非传统模型）
            - maximum_load_time: 最大加载时间（用于非传统模型）
            - drop_extra_time: 是否丢弃额外时间（用于非传统模型）
            - num_tabs: 标签数量（用于Mamba和ExploreModel）
            - Model_name: Explore模型的名称
        num_workers: 数据加载的工作进程数，默认为8

    返回:
        tuple: (model, loader1, loader2)
            model: 根据配置创建的模型实例
            loader1: 第一个数据集的数据加载器
            loader2: 第二个数据集的数据加载器
    """
    dataset_config = aultdict(
         loaded_ratio=args['load_ratio'],
         seq_len=config['seq_len'],
         is_idx=False,

         TAM_type=args['TAM_type'],
         BAPM=None,
         maximum_cell_number=config['maximum_cell_number'],
         max_matrix_len=config['max_matrix_len'],
         log_transform=config['log_transform'],
         maximum_load_time=args['maximum_load_time'],
         time_interval_threshold=config['time_interval_threshold'],
         drop_extra_time = args['drop_extra_time'],)
    set1 = EDdataset(X1,y1, **dataset_config)
    set2 = EDdataset(X2,y2, **dataset_config)

    patch_size = next(iter(set1))[0].shape[1]
    # 使用DataLoader加载数据集
    loader1 = DataLoader(set1, batch_size=int(config['batch_size']),
                            shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    loader2 = DataLoader(set2, batch_size=int(config['batch_size']),
                             shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    if config['model'] == 'LitePrint':
        model = get_model(patch_size=patch_size, num_classes=num_classes, num_tab=args['num_tabs'], model_name=args['Model_name'],
                          # 用于敏感度测试的参数
                          max_matrix_len=config['max_matrix_len'],
                          embed_dim=config['embed_dim'],
                          num_heads=config['num_heads'],
                          r_of_lina=config['r_of_lina'],
                          atten_type=config['atten_type'],
                          )
    else:
        # 如果模型名称错误，则抛出异常
        raise Exception('模型名称错误')
    return model, loader1, loader2


def get_model_and_dataloader1(X1,y1, X2,y2, num_classes, config:dict, args:dict, num_workers=8):
    """
    根据配置创建模型和数据加载器

    参数:
        X1: 第一个数据集的特征数据
        y1: 第一个数据集的标签数据
        X2: 第二个数据集的特征数据
        y2: 第二个数据集的标签数据
        num_classes: 分类任务的类别数量
        config: 配置字典，包含以下键:
            - model: 模型名称，支持["AWF", "DF", "TF", "TMWF", "ARES", "TikTok", "VarCNN", "RF", "MultiTabRF", "CountMamba", "ExploreModel"]
            - seq_len: 序列长度
            - batch_size: 批次大小
            - maximum_cell_number: 最大单元格数量（用于非传统模型）
            - max_matrix_len: 最大矩阵长度（用于非传统模型）
            - log_transform: 是否进行对数变换（用于非传统模型）
            - time_interval_threshold: 时间间隔阈值（用于非传统模型）
            - drop_extra_time: 是否丢弃额外时间（用于非传统模型）
            - minimum_packet_number: 最小数据包数量（用于非传统模型）
            - drop_path_rate: Mamba模型的drop路径率
            - depth: Mamba模型的深度
            - embed_dim: Mamba模型的嵌入维度
            - early_stage: Mamba模型的早期阶段设置
            - fine_predict: Mamba模型的精细预测设置
        args: 参数字典，包含以下键:
            - load_ratio: 数据加载比例（用于传统模型）
            - TAM_type: TAM类型（用于非传统模型）
            - maximum_load_time: 最大加载时间（用于非传统模型）
            - drop_extra_time: 是否丢弃额外时间（用于非传统模型）
            - num_tabs: 标签数量（用于Mamba和ExploreModel）
            - Model_name: Explore模型的名称
        num_workers: 数据加载的工作进程数，默认为8

    返回:
        tuple: (model, loader1, loader2)
            model: 根据配置创建的模型实例
            loader1: 第一个数据集的数据加载器
            loader2: 第二个数据集的数据加载器
    """
    if config['model'] in ["AWF", "DF", "TF", "TMWF", "ARES", "TikTok", "VarCNN", "RF", "MultiTabRF"]:
        dataset_config = aultdict(loaded_ratio=args['load_ratio'], length=int(config['seq_len']))
        if config['model'] in ["AWF", "DF", "TF", "TMWF", "ARES"]:
            # 如果模型是AWF、DF、TF、TMWF或ARES，则使用DirectionDataset数据集
            dataset_str = 'DirectionDataset'
        elif config['model'] in ["TikTok"]:
            # 如果模型是TikTok，则使用DTDataset数据集
            dataset_str = 'DTDataset'
        elif config['model'] in ["VarCNN"]:
            # 如果模型是VarCNN，则使用DT2Dataset数据集
            dataset_str = 'DT2Dataset'
        elif config['model'] in ["RF", "MultiTabRF"]:
            # 如果模型是RF或MultiTabRF，则使用RFDataset数据集
            dataset_str = 'RFDataset'
        set1 = eval(dataset_str)(X1,y1, **dataset_config)
        set2 = eval(dataset_str)(X2,y2, **dataset_config)
    else:
        dataset_config = aultdict(
                 loaded_ratio=args['load_ratio'],
                 seq_len=config['seq_len'],
                 is_idx=False,

                 TAM_type=args['TAM_type'],
                 BAPM=None,
                 maximum_cell_number=config['maximum_cell_number'],
                 max_matrix_len=config['max_matrix_len'],
                 log_transform=config['log_transform'],
                 maximum_load_time=args['maximum_load_time'],
                 time_interval_threshold=config['time_interval_threshold'],
                 drop_extra_time = args['drop_extra_time'],)
        set1 = EDdataset(X1,y1, **dataset_config)
        set2 = EDdataset(X2,y2, **dataset_config)

    patch_size = next(iter(set1))[0].shape[1]
    # 使用DataLoader加载数据集
    loader1 = DataLoader(set1, batch_size=int(config['batch_size']),
                            shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    loader2 = DataLoader(set2, batch_size=int(config['batch_size']),
                             shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    # 根据模型名称获取模型实例
    if config['model'] == "AWF":
        # 如果模型是AWF，则使用AWF模型
        model = AWF(num_classes=num_classes)
    elif config['model'] == "DF":
        # 如果模型是DF，则使用DF模型
        model = DF(num_classes=num_classes)
    elif config['model'] == "TikTok":
        # 如果模型是TikTok，则使用TikTok模型
        model = TikTok(num_classes=num_classes)
    elif config['model'] == "VarCNN":
        # 如果模型是VarCNN，则使用VarCNN模型
        model = VarCNN(num_classes=num_classes)
    elif config['model'] == "TF":
        # 如果模型是TF，则使用TF模型
        model = TF(num_classes=num_classes)
    elif config['model'] == "TMWF":
        # 如果模型是TMWF，则使用TMWF模型
        model = TMWF(num_classes=num_classes)
    elif config['model'] == "ARES":
        # 如果模型是ARES，则使用ARES模型
        model = ARES(num_classes=num_classes)
    elif config['model'] == "RF":
        # 如果模型是RF，则使用RF模型
        model = RF(num_classes=num_classes)
    elif config['model'] == "MultiTabRF":
        # 如果模型是MultiTabRF，则使用MultiTabRF模型
        model = MultiTabRF(num_classes=num_classes)
    elif config['model'] == "CountMamba":
        model = CountMambaModel(num_classes=num_classes, drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                                embed_dim=config['embed_dim'], patch_size=patch_size, max_matrix_len=config['max_matrix_len'],
                                early_stage=config['early_stage'], num_tabs=args['num_tabs'], fine_predict=config['fine_predict'])
    elif config['model'] == 'ExploreModel':
        model = get_model(patch_size=patch_size, num_classes=num_classes, num_tab=args['num_tabs'], model_name=args['Model_name'],
                          # 用于敏感度测试的参数
                          max_matrix_len=config['max_matrix_len'],
                          embed_dim=config['embed_dim'],
                          num_heads=config['num_heads'],
                          r_of_lina=config['r_of_lina'],
                          atten_type=config['atten_type'],
                          )
    else:
        # 如果模型名称错误，则抛出异常
        raise Exception('模型名称错误')
    return model, loader1, loader2

def load_data(data_path, drop_extra_time=False, load_time=None):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    # 时间负数调整
    X[:, :, 0] = np.abs(X[:, :, 0])
    # 去除大小信息
    #X[:, :, 1] = np.sign(X[:, :, 1])
    if drop_extra_time and load_time is not None:
        print(f"丢弃额外时间，时间上限：{load_time}")
        invalid_ind = X[:, :, 0]>load_time
        X[invalid_ind, :] = 0
    return X, y

# def load_data(data_path, drop_extra_time=False, load_time=None):
#     data = np.load(data_path)
#     X = data["X"]
#     y = data["y"]
#
#     # 时间负数调整
#     X[:,:,0] = np.abs(X[:,:,0])
#     return X, y

if __name__ == '__main__':
    from torchinfo import summary
    num_classes = 95
    model, train_loader, test_loader = get_model_and_dataloader(train_X,train_y, test_X,test_y, num_classes, config, args)
