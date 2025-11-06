class IncrementalMeanCalculator:
    def __init__(self):
        self.total = 0
        self.count = 0

    def add(self, new_value):
        self.count += 1
        self.total += new_value

    def get(self):
        if self.count == 0:
            return 0
        else:
            return self.total / self.count

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