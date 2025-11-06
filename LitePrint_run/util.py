from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import numpy as np
import sys
import math
from torch.optim.lr_scheduler import LambdaLR


def gen_one_hot(arr, num_classes):
    binary = np.zeros((arr.shape[0], num_classes))
    for i in range(arr.shape[0]):
        binary[i, arr[i]] = 1

    return binary


def compute_metric(y_true_fine, y_pred_fine):
    y_true_fine = y_true_fine.reshape(-1, y_true_fine.shape[-1])
    y_pred_fine = y_pred_fine.reshape(-1, y_pred_fine.shape[-1])

    num_classes = np.max(y_true_fine) + 1
    y_true_fine = gen_one_hot(y_true_fine, num_classes)
    y_pred_fine = gen_one_hot(y_pred_fine, num_classes)

    result = measurement(y_true_fine, y_pred_fine, eval_metrics="Accuracy Precision Recall F1-score")
    return result


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def pad_along_axis(array: np.ndarray,
                               target_length: int,
                               axis: int = 0,
                               pad_value: float = 0.0) -> np.ndarray:
    """
    沿指定维度padding或截断数组

    Args:
        array: 输入numpy数组
        target_length: 目标长度
        axis: 需要操作的维度 (默认0)
        pad_value: 填充值 (默认0.0)

    Returns:
        调整长度后的数组
    """
    current_length = array.shape[axis]

    if current_length < target_length:
        # 需要padding的情况
        pad_size = target_length - current_length
        pad_width = [(0, 0)] * array.ndim  # 初始化所有维度不padding
        pad_width[axis] = (0, pad_size)  # 只padding指定维度
        return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)
    elif current_length > target_length:
        # 需要截断的情况
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_length)
        return array[tuple(slices)]
    else:
        # 长度正好，直接返回
        return array

def pad_along_axis_backup(array: np.ndarray,
                   target_length: int,
                   axis: int = 0,
                   pad_value: float = 0.0) -> np.ndarray:
    """
    沿指定维度padding数组

    Args:
        array: 输入numpy数组
        target_length: 目标长度
        axis: 需要padding的维度 (默认0)
        pad_value: 填充值 (默认0.0)

    Returns:
        填充后的数组

    Raises:
        ValueError: 当目标长度小于当前长度时
    """
    if target_length < array.shape[axis]:
        raise ValueError(f"Target length ({target_length}) must be >= current length ({array.shape[axis]})")

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    # 构造padding参数：对每个维度指定(pad_before, pad_after)
    pad_width = [(0, 0)] * array.ndim  # 初始化所有维度不padding
    pad_width[axis] = (0, pad_size)  # 只padding指定维度

    return np.pad(array, pad_width=pad_width,mode='constant',constant_values=pad_value)

def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def process_TAM(sequence, maximum_load_time, max_matrix_len):
    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    for pack in sequence:
        if pack == 0:
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[1, idx] += 1
    return feature


def measurement(y_true, y_pred, eval_metrics):
    eval_metrics = eval_metrics.split(" ")
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred) * 100, 2)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    """
    Perform k-Nearest Neighbors (kNN) monitoring.

    Parameters:
    net (nn.Module): The neural network model.
    device (torch.device): The device to run the computations on.
    memory_data_loader (DataLoader): DataLoader for the memory bank.
    test_data_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes.
    k (int): Number of nearest neighbors to use.
    t (float): Temperature parameter for scaling.

    Returns:
    tuple: True labels and predicted labels.
    """
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    return y_true, y_pred


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    Predict labels using k-Nearest Neighbors (kNN) with cosine similarity.

    Parameters:
    feature (Tensor): Feature tensor.
    feature_bank (Tensor): Feature bank tensor.
    feature_labels (Tensor): Labels corresponding to the feature bank.
    classes (int): Number of classes.
    knn_k (int): Number of nearest neighbors to use.
    knn_t (float): Temperature parameter for scaling.

    Returns:
    Tensor: Predicted labels.
    """
    feature_labels = feature_labels.long()
    
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels

def parse_value(config):
    """尝试将字符串转换成 int/float/bool，失败则保持原样"""
    def parse_value(value):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                return value

    config_dict = {k: parse_value(v) for k, v in config['config'].items()}
    return config_dict


def extract_metrics(json_path):
    """
    从字典列表中提取指定指标的值组成新列表

    参数:
    data -- 包含多个字典的列表, 每个字典包含'Accuracy','Precision','Recall','F1-score'键

    返回:
    包含四个列表的元组 (accuracy_list, precision_list, recall_list, f1_list)
    """
    import json
    import numpy as np
    with open(json_path, 'r') as f:
        result = json.load(f)
    data = result['valid']['result']
    loss = result['train']['loss']

    # 初始化四个空列表
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # 遍历每个字典并提取值
    for metrics in data:
        accuracy_list.append(metrics['Accuracy'])
        precision_list.append(metrics['Precision'])
        recall_list.append(metrics['Recall'])
        f1_list.append(metrics['F1-score'])

    return np.array([loss, accuracy_list, precision_list, recall_list, f1_list]).T

def get_model_and_dataset(X1,y1, X2,y2, num_classes, config, args):
    from torch.utils.data import DataLoader
    from dataset_util import DirectionDataset, DTDataset, DT2Dataset, RFDataset, FFTDataset
    from ExploreModel import EMDataset
    # 数据
    if config['model'] in ["AWF", "DF", "TF", "TMWF", "ARES"]:
        # 如果模型是AWF、DF、TF、TMWF或ARES，则使用DirectionDataset数据集
        set1 = DirectionDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = DirectionDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["TikTok"]:
        # 如果模型是TikTok，则使用DTDataset数据集
        set1 = DTDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = DTDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["VarCNN"]:
        # 如果模型是VarCNN，则使用DT2Dataset数据集
        set1 = DT2Dataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = DT2Dataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["RF", "MultiTabRF"]:
        # 如果模型是RF或MultiTabRF，则使用RFDataset数据集
        set1 = RFDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = RFDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["TFRF", "TFRF_fft"]:
        # 如果模型是TFRF或TFRF_fft，则使用CountDataset数据集
        set1 = FFTDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = FFTDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        # set1 = CountDataset(X1,y1, args=config, is_idx=False, TAM='TF')
        # set2 = CountDataset(X2,y2,args=config, is_idx=False, TAM='TF')
    elif config['model'] in ["CountMamba"]:
        # 如果模型是CountMamba，则使用CountDataset数据集
        set1 = EMDataset(X1,y1, loaded_ratio=args.load_ratio,
                         is_idx=False, TAM_type='Mamba',
                         seq_len=config['seq_len'],
                         maximum_cell_number=config['maximum_cell_number'],
                         max_matrix_len=config['max_matrix_len'],
                         log_transform=config['log_transform'],
                         maximum_load_time=config['maximum_load_time'],
                         time_interval_threshold=config['time_interval_threshold']
                         )
        set2 = EMDataset(X2,y2, loaded_ratio=args.load_ratio,
                         is_idx=False, TAM_type='Mamba',
                         seq_len=config['seq_len'],
                         maximum_cell_number=config['maximum_cell_number'],
                         max_matrix_len=config['max_matrix_len'],
                         log_transform=config['log_transform'],
                         maximum_load_time=config['maximum_load_time'],
                         time_interval_threshold=config['time_interval_threshold']
                         )
    elif config['model'] in ["TFCNN"]:
        # 如果模型是TFCNN，则使用CountDataset数据集
        set1 = EMDataset(X1,y1, loaded_ratio=args.load_ratio,
                         is_idx=False, TAM_type='TF',
                         seq_len=config['seq_len'],
                         maximum_cell_number=config['maximum_cell_number'],
                         max_matrix_len=config['max_matrix_len'],
                         log_transform=config['log_transform'],
                         maximum_load_time=config['maximum_load_time'],
                         time_interval_threshold=config['time_interval_threshold']
                         )
        set2 = EMDataset(X2,y2, loaded_ratio=args.load_ratio,
                         is_idx=False, TAM_type='TF',
                         seq_len=config['seq_len'],
                         maximum_cell_number=config['maximum_cell_number'],
                         max_matrix_len=config['max_matrix_len'],
                         log_transform=config['log_transform'],
                         maximum_load_time=config['maximum_load_time'],
                         time_interval_threshold=config['time_interval_threshold']
                         )
    elif config['model'] in ["ExploreModel"]:
        # 如果模型是ExploreModel，则使用EMDataset数据集
        set1 = EMDataset(X1, y1, loaded_ratio=args.load_ratio,
                BAPM=None, is_idx=False, TAM_type=config['tam_type'],
                 seq_len=config['seq_len'],
                 maximum_cell_number=config['maximum_cell_number'],
                 max_matrix_len=config['max_matrix_len'],
                 log_transform=config['log_transform'],
                 maximum_load_time=config['maximum_load_time'],
                 time_interval_threshold=config['time_interval_threshold']
                 )
        set2 = EMDataset(X2, y2,loaded_ratio=args.load_ratio,
                 BAPM=None, is_idx=False, TAM_type=config['tam_type'],
                 seq_len=config['seq_len'],
                 maximum_cell_number=config['maximum_cell_number'],
                 max_matrix_len=config['max_matrix_len'],
                 log_transform=config['log_transform'],
                 maximum_load_time=config['maximum_load_time'],
                 time_interval_threshold=config['time_interval_threshold']
                 )
    patch_size = next(iter(set1))[0].shape[1]

    # 使用DataLoader加载数据集
    loader1 = DataLoader(set1, batch_size=int(config['batch_size']),
                            shuffle=False, drop_last=False, num_workers=8)
    loader2 = DataLoader(set2, batch_size=int(config['batch_size']),
                             shuffle=False, drop_last=False, num_workers=8)

    # Model
    from model import AWF, DF, VarCNN
    from model_tiktok import TikTok
    from model_TF import TF
    from model_TMWF import TMWF
    from model_RF import RF
    from model_MultiTabRF import MultiTabRF
    from model_ARES import Trans_WF as ARES
    from model_TFRF import TFRF
    from model_TFRF_fft import TFRF_fft
    from CountMambaModel import CountMambaModel
    from MyModel import TFCNN_model
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
    elif config['model'] == "TFRF":
        # 如果模型是TFRF，则使用TFRF模型
        model = TFRF(num_classes=num_classes)
    elif config['model'] == "TFRF_fft":
        # 如果模型是TFRF_fft，则使用TFRF_fft模型
        model = TFRF_fft(num_classes=num_classes, patch_size=10)
    elif config['model'] == "CountMamba":
        # 如果模型是CountMamba，则使用CountMambaModel模型
        model = CountMambaModel(num_classes=num_classes, drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                                embed_dim=config['embed_dim'], patch_size=patch_size, max_matrix_len=config['max_matrix_len'],
                                early_stage=config['early_stage'], num_tabs=args.num_tabs, fine_predict=config['fine_predict'])
    elif config['model'] == 'TFCNN':
        # 如果模型是TFCNN，则使用TFCNN_model模型
        model = TFCNN_model(patch_size=patch_size, num_classes=num_classes, num_tab=args.num_tabs, embed_dim=config['embed_dim'])
    elif config['model'] == 'ExploreModel':
        # 如果模型是ExploreModel，则使用ExploreModel模型
        from ExploreModel import ExploreModel
        #model = ExploreModel(patch_size=patch_size, num_classes=num_classes, num_tab=args.num_tabs)
        from user_test.main import ExploreModel_base as ExploreModel
        from user_test.test import RF1 as ExploreModel
        model = ExploreModel(patch_size=patch_size, num_classes=num_classes, num_tab=args.num_tabs)
    else:
        # 如果模型名称错误，则抛出异常
        raise Exception('模型名称错误')
    return model, loader1, loader2

def get_model_and_dataset_backup(X1,y1, X2,y2, num_classes, config, args):
    from torch.utils.data import DataLoader
    from dataset_util import DirectionDataset, DTDataset, DT2Dataset, RFDataset
    from ExploreModel import ExporeDataset
    # 数据
    if config['model'] in ["AWF", "DF", "TF", "TMWF", "ARES"]:
        # 如果模型是AWF、DF、TF、TMWF或ARES，则使用DirectionDataset数据集
        set1 = DirectionDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = DirectionDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["TikTok"]:
        # 如果模型是TikTok，则使用DTDataset数据集
        set1 = DTDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = DTDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["VarCNN"]:
        # 如果模型是VarCNN，则使用DT2Dataset数据集
        set1 = DT2Dataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = DT2Dataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["RF", "MultiTabRF"]:
        # 如果模型是RF或MultiTabRF，则使用RFDataset数据集
        set1 = RFDataset(X1,y1,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
        set2 = RFDataset(X2,y2,loaded_ratio=args.load_ratio, length=int(config['seq_len']))
    elif config['model'] in ["CountMamba"]:
        # 如果模型是CountMamba，则使用CountDataset数据集
        set1 = ExporeDataset(X1,y1, loaded_ratio=args.load_ratio,
                         is_idx=False, TAM_type='Mamba',
                         seq_len=config['seq_len'],
                         maximum_cell_number=config['maximum_cell_number'],
                         max_matrix_len=config['max_matrix_len'],
                         log_transform=config['log_transform'],
                         maximum_load_time=config['maximum_load_time'],
                         time_interval_threshold=config['time_interval_threshold']
                         )
        set2 = ExporeDataset(X2,y2, loaded_ratio=args.load_ratio,
                         is_idx=False, TAM_type='Mamba',
                         seq_len=config['seq_len'],
                         maximum_cell_number=config['maximum_cell_number'],
                         max_matrix_len=config['max_matrix_len'],
                         log_transform=config['log_transform'],
                         maximum_load_time=config['maximum_load_time'],
                         time_interval_threshold=config['time_interval_threshold']
                         )
    elif config['model'] in ["ExploreModel"]:
        # 如果模型是ExploreModel，则使用EMDataset数据集
        set1 = EMDataset(X1, y1, loaded_ratio=args.load_ratio,
                BAPM=None, is_idx=False, TAM_type=config['tam_type'],
                 seq_len=config['seq_len'],
                 maximum_cell_number=config['maximum_cell_number'],
                 max_matrix_len=config['max_matrix_len'],
                 log_transform=config['log_transform'],
                 maximum_load_time=config['maximum_load_time'],
                 time_interval_threshold=config['time_interval_threshold']
                 )
        set2 = EMDataset(X2, y2,loaded_ratio=args.load_ratio,
                 BAPM=None, is_idx=False, TAM_type=config['tam_type'],
                 seq_len=config['seq_len'],
                 maximum_cell_number=config['maximum_cell_number'],
                 max_matrix_len=config['max_matrix_len'],
                 log_transform=config['log_transform'],
                 maximum_load_time=config['maximum_load_time'],
                 time_interval_threshold=config['time_interval_threshold']
                 )
    patch_size = next(iter(set1))[0].shape[1]

    # 使用DataLoader加载数据集
    loader1 = DataLoader(set1, batch_size=int(config['batch_size']),
                            shuffle=False, drop_last=False, num_workers=8)
    loader2 = DataLoader(set2, batch_size=int(config['batch_size']),
                             shuffle=False, drop_last=False, num_workers=8)

    # Model
    from ExploreModel import ExploreModel
    from CountMambaModel import CountMambaModel


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
    elif config['model'] == "TFRF":
        # 如果模型是TFRF，则使用TFRF模型
        model = TFRF(num_classes=num_classes)
    elif config['model'] == "TFRF_fft":
        # 如果模型是TFRF_fft，则使用TFRF_fft模型
        model = TFRF_fft(num_classes=num_classes, patch_size=10)
    elif config['model'] == "CountMamba":
        # 如果模型是CountMamba，则使用CountMambaModel模型
        model = CountMambaModel(num_classes=num_classes, drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                                embed_dim=config['embed_dim'], patch_size=patch_size, max_matrix_len=config['max_matrix_len'],
                                early_stage=config['early_stage'], num_tabs=args.num_tabs, fine_predict=config['fine_predict'])
    elif config['model'] == 'TFCNN':
        # 如果模型是TFCNN，则使用TFCNN_model模型
        model = TFCNN_model(patch_size=patch_size, num_classes=num_classes, num_tab=args.num_tabs, embed_dim=config['embed_dim'])
    elif config['model'] == 'ExploreModel':
        # 如果模型是ExploreModel，则使用ExploreModel模型
        from ExploreModel import ExploreModel
        #model = ExploreModel(patch_size=patch_size, num_classes=num_classes, num_tab=args.num_tabs)
        from user_test.main import ExploreModel_base as ExploreModel
        from user_test.test import RF1 as ExploreModel
        model = ExploreModel(patch_size=patch_size, num_classes=num_classes, num_tab=args.num_tabs)
    else:
        # 如果模型名称错误，则抛出异常
        raise Exception('模型名称错误')
    return model, loader1, loader2

def load_data(data_path, drop_extra_time=False, load_time=None):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    if drop_extra_time and load_time is not None:
        for i,data_i in enumerate(X):
            no_val_ind = data_i[:,0] > load_time
            X[i,no_val_ind,:] = 0
    return X, y

## d = extract_metrics('./checkpoints/CW/TFCNN/dataset_TF')
if __name__ == '__main__':
    import os
    x,y=load_data(os.path.join("../npz_dataset", "CW", f"valid.npz"),True,10)
    x1, y1 = load_data(os.path.join("../npz_dataset", "CW", f"valid.npz"), True, 80)



