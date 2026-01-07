from torch.utils.data import Dataset
import math
import numpy as np


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)
    return sequence


class CountDataset(Dataset):
    def __init__(self, X, labels, loaded_ratio=100, BAPM=None, is_idx=False, TAM_type='Mamba',
                 seq_len=5000,
                 maximum_cell_number=2,
                 max_matrix_len=1800,
                 log_transform=False,
                 maximum_load_time=80,
                 time_interval_threshold=0.1,
                 drop_extra_time = False,
                 ):
        self.X = X
        self.labels = labels
        self.loaded_ratio = loaded_ratio
        self.BAPM = BAPM
        self.is_idx = is_idx
        self.TAM = TAM_type
        self.drop_extra_time = drop_extra_time

        self.args = {
            "seq_len" : seq_len,
            "maximum_cell_number" : maximum_cell_number,
            "max_matrix_len" : max_matrix_len,
            "log_transform" : log_transform,
            "maximum_load_time" : maximum_load_time,
            "time_interval_threshold" : time_interval_threshold,
            "minimum_packet_number" : 10,
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.labels[index]

        timestamp = data[:, 0]
        loading_time = timestamp.max()
        if self.drop_extra_time:
            threshold = min(loading_time * self.loaded_ratio / 100,self.args['maximum_load_time'])
        else:
            threshold = loading_time * self.loaded_ratio / 100
        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)[0]
        data = data[valid_index,:]

        if self.BAPM is not None:
            bapm = self.BAPM[index]
            return self.process_data(data, bapm=bapm), label
        else:
            return self.process_data(data), label

    def process_data(self, data, bapm=None):
        time = data[:, 0]
        packet_length = data[:, 1]

        packet_length = pad_sequence(packet_length, self.args["seq_len"])
        time = pad_sequence(time, self.args["seq_len"])
        # get_TAM_Mamba get_TAM_TF
        TAM, current_index, bapm_labels = eval(f"get_TAM_{self.TAM}")(packet_length, time, args=self.args, bapm=bapm)
        TAM = TAM.reshape((1, -1, self.args["max_matrix_len"]))
        if self.args["log_transform"]:
            TAM = np.log1p(TAM)
        if self.is_idx:
            if bapm is not None:
                return TAM.astype(np.float32), current_index, bapm_labels
            else:
                return TAM.astype(np.float32), current_index
        else:
            if bapm is not None:
                return TAM.astype(np.float32), bapm_labels
            else:
                return TAM.astype(np.float32)



def get_TAM_RF(packet_length, time, args, bapm):
    max_matrix_len = args["max_matrix_len"]
    sequence = np.sign(packet_length) * time
    maximum_load_time = args["maximum_load_time"]

    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    count = 0
    for pack in sequence:
        count += 1
        if pack == 0:
            count -= 1
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
    return feature, count, None


def get_TAM_Mamba(packet_length, time, args, bapm):
    # 统计窗口长度 返回datalen
    args['feature_dim'] = 2 * (args["maximum_cell_number"] + 2)
    feature = np.zeros((args['feature_dim'], args["max_matrix_len"]))
    w = args["maximum_load_time"] / args["max_matrix_len"]
    time_interval = w * args["time_interval_threshold"]

    if bapm is not None:
        bapm_time = np.trim_zeros(time, "b")

        bapm_first_start = min(len(bapm_time) - 1, bapm[0])
        bapm_first_end = min(len(bapm_time) - 1, bapm[0] + bapm[1])
        bapm_second_start = min(len(bapm_time) - 1, bapm[2])
        bapm_second_end = min(len(bapm_time) - 1, bapm[2] + bapm[3])

        bapm_first_start_time = bapm_time[bapm_first_start]
        bapm_first_end_time = bapm_time[bapm_first_end]
        bapm_second_start_time = bapm_time[bapm_second_start]
        bapm_second_end_time = bapm_time[bapm_second_end]

        bapm_first_start_position = min(math.floor(bapm_first_start_time / w), args["max_matrix_len"] - 1)
        bapm_first_end_position = min(math.floor(bapm_first_end_time / w), args["max_matrix_len"] - 1)
        bapm_second_start_position = min(math.floor(bapm_second_start_time / w), args["max_matrix_len"] - 1)
        bapm_second_end_position = min(math.floor(bapm_second_end_time / w), args["max_matrix_len"] - 1)

        bapm_labels = np.full((2, args["max_matrix_len"]), -1)
        bapm_labels[0, bapm_first_start_position:bapm_first_end_position + 1] = bapm[-2]
        bapm_labels[1, bapm_second_start_position:bapm_second_end_position + 1] = bapm[-1]
    else:
        bapm_labels = None

    current_index = 0
    current_timestamps = []
    data_len = []
    for l_k, t_k in zip(packet_length, time):
        if t_k == 0 and l_k == 0:
            break  # End of sequence

        d_k = int(np.sign(l_k))
        c_k = min(int(np.abs(l_k) // 512), args["maximum_cell_number"])  # [0, C]

        fragment = 0 if d_k < 0 else 1
        i = 2 * c_k + fragment  # [0, 2C + 1]
        j = min(math.floor(t_k / w), args["max_matrix_len"] - 1)
        j = max(j, 0)
        feature[i, j] += 1

        if j != current_index:
            feature[2 * args["maximum_cell_number"] + 2, j] = max(j - current_index, 0)
            data_len.append(len(current_timestamps))
            delta_t = np.diff(current_timestamps)
            cluster_count = np.sum(delta_t > time_interval) + 1
            feature[2 * args["maximum_cell_number"] + 3, current_index] = cluster_count



            current_index = j
            current_timestamps = [t_k]
        else:
            current_timestamps.append(t_k)

    delta_t = np.diff(current_timestamps)
    data_len.append(len(current_timestamps))
    cluster_count = np.sum(delta_t > time_interval) + 1
    feature[2 * args["maximum_cell_number"] + 3, current_index] = cluster_count

    return feature, data_len, bapm_labels
def get_TAM_SWIFT(packet_length, time, args, bapm):
    # 加上窗口内包大小特征共5个
    w = args["maximum_load_time"] / args["max_matrix_len"]
    time_interval = w * args["time_interval_threshold"]

    if bapm is not None:
        bapm_time = np.trim_zeros(time, "b")

        bapm_first_start = min(len(bapm_time) - 1, bapm[0])
        bapm_first_end = min(len(bapm_time) - 1, bapm[0] + bapm[1])
        bapm_second_start = min(len(bapm_time) - 1, bapm[2])
        bapm_second_end = min(len(bapm_time) - 1, bapm[2] + bapm[3])

        bapm_first_start_time = bapm_time[bapm_first_start]
        bapm_first_end_time = bapm_time[bapm_first_end]
        bapm_second_start_time = bapm_time[bapm_second_start]
        bapm_second_end_time = bapm_time[bapm_second_end]

        bapm_first_start_position = min(math.floor(bapm_first_start_time / w), args["max_matrix_len"] - 1)
        bapm_first_end_position = min(math.floor(bapm_first_end_time / w), args["max_matrix_len"] - 1)
        bapm_second_start_position = min(math.floor(bapm_second_start_time / w), args["max_matrix_len"] - 1)
        bapm_second_end_position = min(math.floor(bapm_second_end_time / w), args["max_matrix_len"] - 1)

        bapm_labels = np.full((2, args["max_matrix_len"]), -1)
        bapm_labels[0, bapm_first_start_position:bapm_first_end_position + 1] = bapm[-2]
        bapm_labels[1, bapm_second_start_position:bapm_second_end_position + 1] = bapm[-1]
    else:
        bapm_labels = None
    feature = extract_features_swift(time, packet_length, args["maximum_load_time"], args["max_matrix_len"])
    #data_len = np.where(feature[2,:]!=0)[0][-1]+1
    data_len = -1
    return feature,data_len, bapm_labels
def extract_features_swift(T, L, Tmax, max_column):
    """
    优化版本特征提取函数，保留五个特征：
    0: 上行包数量
    1: 下行包数量
    2: 包间隔（窗口索引差）
    3: 上行包大小和（除以512）
    4: 下行包大小和（除以512）
    """
    # 初始化特征矩阵 (5行 x max_column列)
    feature = np.zeros((5, max_column))

    # 截断T==0之后的数据
    indices = np.flatnonzero(T)
    ind = indices[-1]+1 if indices.size > 0 else len(T)
    T = T[:ind]
    L = L[:ind]

    # 无有效数据时返回空特征
    if len(T) == 0:
        return feature

    # 计算所有窗口索引
    all_windows = np.floor(T / Tmax * (max_column - 1)).astype(int)
    all_windows = np.clip(all_windows, 0, max_column - 1)

    # 特征0&1: 分别计算上行/下行包数量
    up_mask = L > 0
    down_mask = L < 0
    if np.any(up_mask):
        feature[0] = np.bincount(all_windows[up_mask], minlength=max_column)
    if np.any(down_mask):
        feature[1] = np.bincount(all_windows[down_mask], minlength=max_column)

    # 特征3&4: 分别计算上行/下行包大小和（除以512）
    if np.any(up_mask):
        # 上行包大小和：直接使用L[up_mask]（正数），然后除以512
        up_size_sum = np.bincount(all_windows[up_mask], weights=L[up_mask], minlength=max_column)
        feature[3] = up_size_sum / 512.0  # 使用浮点除法避免整数截断
    if np.any(down_mask):
        # 下行包大小和：取-L[down_mask]（绝对值），然后除以512
        down_size_sum = np.bincount(all_windows[down_mask], weights=-L[down_mask], minlength=max_column)
        feature[4] = down_size_sum / 512.0

    # 特征2: 包间隔（窗口索引差）
    unique_windows = np.unique(all_windows)
    if unique_windows.size > 1:
        window_gaps = np.diff(unique_windows)
        feature[2, unique_windows[1:]] = window_gaps

    return np.log(1+feature)



if __name__ == "__main__":
    import os
    import time
    # 定义数据集路径
    from const import dataset_lib
    def load_data(data_path, drop_extra_time=False, load_time=None):
        data = np.load(data_path)
        X = data["X"]
        y = data["y"]
        # 时间负数调整
        X[:, :, 0] = np.abs(X[:, :, 0])
        if drop_extra_time and load_time is not None:
            print(f"丢弃额外时间，时间上限：{load_time}")
            invalid_ind = X[:, :, 0] > load_time
            X[invalid_ind, :] = 0
        return X, y
    def test_process(TAM_type, load_ratio=100, N=500, max_matrix_len=1800, seq_len=5000):
        dataset = CountDataset(X, y, loaded_ratio=load_ratio, TAM_type=TAM_type,
                               is_idx=True,
                               maximum_load_time=120,
                               drop_extra_time=True,
                               max_matrix_len=max_matrix_len,
                               seq_len=seq_len)
        tic = time.time()
        if N == -1:
            N = len(dataset)
        for i in range(N):
            dataset[i]
        toc = time.time()
        time_duration = round(toc - tic,2)
        print(f'dataset shape : {dataset[i][0][0].shape}')
        print(f"{TAM_type}({load_ratio}%) - 运行时间:{time_duration:.2f}s\n")
        return dataset, time_duration


    for dataset in ['Closed_2tab', 'Closed_3tab', 'Closed_4tab', 'Closed_5tab']:
        base_dir = f"../npz_dataset/{dataset}"  # CW trafficsilver_bwr_CW Closed_5tab regulator_Closed_2tab
        data_path = os.path.join(base_dir, "valid.npz")
        X, y = load_data(data_path,drop_extra_time=False,load_time=dataset_lib[os.path.basename(base_dir)]["maximum_load_time"])
        # 时间负数调整
        X[:, :, 0] = np.abs(X[:, :, 0])
        n = -1
        seq_len = 10000
        test_process(TAM_type='SWIFT', load_ratio=100, N=n, seq_len=seq_len)
        test_process(TAM_type='RF', load_ratio=100, N=n, seq_len=seq_len)
        test_process(TAM_type='Mamba', load_ratio=100, N=n, seq_len=seq_len)

