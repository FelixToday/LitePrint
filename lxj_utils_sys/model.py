import torch
import numpy as np
import math
from sklearn.metrics import precision_recall_curve, auc
def calculate_conv_output_size(input_size, conv_settings):
    """
    计算卷积网络的输出尺寸

    Args:
        input_size: 输入尺寸
        conv_settings: 卷积设置，可以是：
            - 列表/元组: [kernel, stride, pad, dialate] (单层)
            - 列表的列表: [[k1,s1,p1,d1], [k2,s2,p2,d2], ...] (多层)
    """
    # 判断输入类型
    if isinstance(conv_settings[0], (list, tuple)):
        # 多层卷积：[[k1,s1,p1,d1], [k2,s2,p2,d2], ...]
        settings_list = conv_settings
    else:
        # 单层卷积：[kernel, stride, pad, dialate]
        settings_list = [conv_settings]

    current_size = input_size

    for setting in settings_list:
        if len(setting) != 4:
            raise ValueError("每个卷积设置必须包含4个参数: [kernel, stride, pad, dialate]")

        k, s, p, d = setting

        # 处理same填充
        if p == "same":
            p = k // 2

        # 确保所有参数都是数值类型
        if not all(isinstance(x, (int, float)) for x in [k, s, p, d]):
            raise TypeError("计算时所有参数必须是数值类型")

        # 计算当前层的输出尺寸
        current_size = (current_size + 2 * p - d * (k - 1) - 1) // s + 1

    return current_size

if __name__ == "__main__":
    # 单层卷积 - 列表格式
    result1 = calculate_conv_output_size(32, [3, 1, 1, 1])
    print(result1)  # 输出: 32

    # 多层卷积 - 列表的列表格式
    result2 = calculate_conv_output_size(32, [
        [3, 1, "same", 1],  # 第一层
        [5, 2, 2, 1],  # 第二层
        [3, 1, 1, 1]  # 第三层
    ])
    print(result2)  # 输出: 15 (32→32→15→15)

    # 多层卷积 - 列表的列表格式
    stride = 1
    result2 = calculate_conv_output_size(1800, [
        [8, 2, 0, 1],  # 第一层
        [8, 2, 0, 1],  # 第一层
        [8, 2, 0, 1],  # 第一层
        [8, 2, 0, 1],  # 第一层
    ])
    print(result2)  # 输出: 15 (32→32→15→15)




class LearningRateScheduler:
    def __init__(self, optimizer, lr, min_lr, warmup_epochs, total_epochs):
        """
        学习率调度器

        Args:
            optimizer: 优化器
            lr: 基础学习率
            min_lr: 最小学习率
            warmup_epochs: 预热轮数
            total_epochs: 总训练轮数
        """
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def step(self, epoch):
        """
        根据当前epoch调整学习率

        Args:
            epoch: 当前训练轮数

        Returns:
            lr: 当前学习率
        """
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.lr * epoch / self.warmup_epochs
        else:
            # 余弦退火阶段
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - self.warmup_epochs) /
                                (self.total_epochs - self.warmup_epochs)))

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

        return lr




def compute_pr_result(model, dataloader, task_type="binary", average="macro", device=None, downsample=True, num_points=100):
    """
    通用PR计算函数，支持二分类、多分类、多标签任务，以及宏/微平均。
    可选择下采样 PR 曲线点，减少保存数据量。
    """
    import torch
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            if task_type == "binary":
                probs = torch.sigmoid(outputs).cpu()
            elif task_type == "multiclass":
                probs = torch.softmax(outputs, dim=1).cpu()
            elif task_type == "multilabel":
                probs = torch.sigmoid(outputs).cpu()
            else:
                raise ValueError("Unsupported task_type")

            all_probs.append(probs)
            all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()

    result = {}

    def downsample_curve(precision, recall, thresholds=None):
        """下采样函数"""
        if not downsample:
            return precision, recall, thresholds
        # 使用插值统一长度
        all_points = num_points
        mean_recall = np.linspace(0, 1, all_points)
        mean_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        if thresholds is not None:
            # thresholds长度比 precision/recall少1，用线性插值近似
            thresholds_full = np.concatenate(([0], thresholds))
            mean_thresholds = np.interp(mean_recall, recall[::-1], thresholds_full[::-1])
            return mean_precision, mean_recall, mean_thresholds
        return mean_precision, mean_recall, thresholds

    if task_type == "binary":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
        result["precision"] = precision.tolist()
        result["recall"] = recall.tolist()
        result["thresholds"] = thresholds.tolist() if thresholds is not None else []
        result["auc"] = float(auc(recall, precision))

    elif task_type == "multiclass":
        n_classes = y_score.shape[1]
        y_onehot = np.eye(n_classes)[y_true]
        if average == "micro":
            precision, recall, thresholds = precision_recall_curve(y_onehot.ravel(), y_score.ravel())
            precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
            result["precision"] = precision.tolist()
            result["recall"] = recall.tolist()
            result["thresholds"] = thresholds.tolist() if thresholds is not None else []
            result["auc"] = float(auc(recall, precision))
        elif average == "macro":
            precisions, recalls, aucs = [], [], []
            for i in range(n_classes):
                p, r, t = precision_recall_curve(y_onehot[:, i], y_score[:, i])
                precisions.append(p)
                recalls.append(r)
                aucs.append(auc(r, p))
            # 插值平均
            mean_recall = np.linspace(0, 1, num_points)
            mean_precision = np.zeros(num_points)
            for p, r in zip(precisions, recalls):
                mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
            mean_precision /= n_classes
            result["precision"] = mean_precision.tolist()
            result["recall"] = mean_recall.tolist()
            result["thresholds"] = []  # macro模式下无统一threshold
            result["auc"] = float(np.mean(aucs))
        else:
            raise ValueError("average must be 'micro' or 'macro'")

    elif task_type == "multilabel":
        n_labels = y_score.shape[1]
        if average == "micro":
            precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_score.ravel())
            precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
            result["precision"] = precision.tolist()
            result["recall"] = recall.tolist()
            result["thresholds"] = thresholds.tolist() if thresholds is not None else []
            result["auc"] = float(auc(recall, precision))
        elif average == "macro":
            precisions, recalls, aucs = [], [], []
            for i in range(n_labels):
                p, r, t = precision_recall_curve(y_true[:, i], y_score[:, i])
                precisions.append(p)
                recalls.append(r)
                aucs.append(auc(r, p))
            # 插值平均
            mean_recall = np.linspace(0, 1, num_points)
            mean_precision = np.zeros(num_points)
            for p, r in zip(precisions, recalls):
                mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
            mean_precision /= n_labels
            result["precision"] = mean_precision.tolist()
            result["recall"] = mean_recall.tolist()
            result["thresholds"] = []
            result["auc"] = float(np.mean(aucs))
        else:
            raise ValueError("average must be 'micro' or 'macro'")

    return result
