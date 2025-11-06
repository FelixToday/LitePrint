import argparse
import configparser
import time

import numpy as np
import tqdm
import torch
import os

from util import measurement, compute_metric
import warnings
from lxj_utils_sys import BaseLogger, ModelCheckpoint, str_to_bool, same_seed, LearningRateScheduler, print_dict
from lxj_utils_sys import IncrementalMeanCalculator
from util import parse_value

from LitePrint import get_model_and_dataloader, load_data

from LitePrint.const import dataset_lib

warnings.filterwarnings("ignore")

# Set a fixed seed for reproducibility
fix_seed = 2025
same_seed(fix_seed)

# 公共参数
# Config
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument('--dataset', default="Closed_5tab")  # CW trafficsilver_bwr_CW Closed_2tab regulator_CW regulator_Closed_2tab
parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
parser.add_argument('--config', default="config/LitePrint.ini")
parser.add_argument("--num_tabs", type=int, default=1)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--note", type=str, default='test')
parser.add_argument("--load_ratio", type=float, default=100)
parser.add_argument("--test_flag", type=str_to_bool, default=True)
parser.add_argument("--load_checkpoint", type=str_to_bool, default=False)
parser.add_argument("--TAM_type", type=str, default='SWIFT')
parser.add_argument("--Model_name", type=str, default='LitePrint')
parser.add_argument("--maximum_load_time", type=float, default=120)
parser.add_argument("--drop_extra_time", type=str_to_bool, default=True)
parser.add_argument("--num_workers", type=int, default=2)

parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--stag_epochs', type=int, default=20)
parser.add_argument('--optim', type=str_to_bool, default=False)


parser.add_argument('--is_Sen', type=str_to_bool, default=False)
## 第一组参数
# 1. 5k 10k 15k 20k
parser.add_argument('--seq_len', type=int, default=10000)
# 2. maximum_load_time 80 120 180 240 （默认120s）
# 3. 2400 3600 4800 7200
parser.add_argument('--max_matrix_len', type=int, default=7200)
# 4. 32 64 128 256
parser.add_argument('--embed_dim', type=int, default=256)
# 5. 4 8 12 16
parser.add_argument('--num_heads', type=int, default=8)
# 6. 2 4 6 8
parser.add_argument('--r_of_lina', type=int, default=5)

## 第二组 用于不同注意力机制的参数
parser.add_argument('--atten_type', type=str, default="Linear")
args = parser.parse_args()


# ================= System 参数调整开始 ================
args.num_tabs = dataset_lib[args.dataset]['num_tabs']
args.maximum_load_time = dataset_lib[args.dataset]['maximum_load_time'] if not args.is_Sen else args.maximum_load_time
# 私用参数
config = configparser.ConfigParser()
config.read(args.config)
config = parse_value(config)
# 特殊参数调整
config_name = args.config.strip(".ini").strip("config/")
if args.num_tabs > 1:
    if config_name in ['LitePrint']:
        config['max_matrix_len'] = 7200
        config['seq_len'] = 10000
# ================= System 参数调整完毕 ================


# ================= Sensitivity Test 参数调整开始 ================
if args.is_Sen:
    config['seq_len'] = args.seq_len
    # args.maximum_load_time
    config['max_matrix_len'] = args.max_matrix_len
    config['embed_dim'] = args.embed_dim
    config['num_heads'] = args.num_heads
    config['r_of_lina'] = args.r_of_lina
    config['atten_type'] = args.atten_type
# ================= Sensitivity Test 参数调整完毕 ================

print_dict(config)
print_dict(vars(args))
# Preparation
device = torch.device(args.device)

# 创建检查点保存路径
ckp_path = os.path.join(
    "../checkpoints",
    args.dataset,
    config['model'],
    args.note
).rstrip('/')  # 移除末尾可能多余的斜杠

os.makedirs(ckp_path, exist_ok=True)
# out_file = os.path.join(ckp_path, f"max_f1.pth")

# 初始化日志记录器
logger = BaseLogger(json_save_path=os.path.join(ckp_path, "result.json"),
                    log_save_path=os.path.join(ckp_path, "log.txt"))

# 根据任务类型（单标签 vs 多标签）设置模型保存器
if args.num_tabs == 1:
    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode='max',
                                 metric_name='f1',
                                 max_stagnation_epochs=args.stag_epochs if args.optim else None)
else:
    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode='max',
                                 metric_name='ap',
                                 max_stagnation_epochs=args.stag_epochs if args.optim else None)

file_base_dir = "../npz_dataset"
# 加载训练和验证数据
if not args.test_flag:
    train_X, train_y = load_data(os.path.join(file_base_dir, args.dataset, f"train.npz"), drop_extra_time=args.drop_extra_time, load_time=args.maximum_load_time)
    valid_X, valid_y = load_data(os.path.join(file_base_dir, args.dataset, f"valid.npz"), drop_extra_time=args.drop_extra_time, load_time=args.maximum_load_time)
else:
    # 测试模式下使用验证集作为训练集，测试集作为验证集
    train_X, train_y = load_data(os.path.join(file_base_dir, args.dataset, f"valid.npz"), drop_extra_time=args.drop_extra_time, load_time=args.maximum_load_time)
    valid_X, valid_y = load_data(os.path.join(file_base_dir, args.dataset, f"test.npz"), drop_extra_time=args.drop_extra_time, load_time=args.maximum_load_time)

# 确定类别数量
if args.num_tabs == 1:
    num_classes = len(np.unique(train_y))
else:
    num_classes = train_y.shape[1]

# 获取模型和数据加载器
model, train_loader, val_loader = get_model_and_dataloader(train_X, train_y, valid_X, valid_y, num_classes, config, vars(args), num_workers=args.num_workers)

# 加载检查点或初始化训练
if args.load_checkpoint:
    model, metric_best_value, base_epoch = modelsaver.load(model)
else:
    logger.log('config.config', config, True)
    logger.log('config.args', vars(args), True)
    metric_best_value = 0
    base_epoch = 0

# 确保metric_best_value不为None
if metric_best_value is None:
    metric_best_value = 0
    base_epoch = 0

# 将模型移动到指定设备
model.to(device)
# 初始化优化器
optimizer = eval(f"torch.optim.{config['optimizer']}")(model.parameters(),
                                                       lr=float(config['learning_rate']))
print(model.__class__)
print(f'optim:{args.optim}', 'config:', args.config, 'Model name:', args.Model_name, 'TAM type:', args.TAM_type, sep="\t")
logger.log("config.model", str(model.__class__))
# 记录meta信息到console
logger.info(f"model:{model.__class__}", is_logfile=True, log_to_console=False)
logger.info(f"args:{str(args)}", is_logfile=True, log_to_console=False)
logger.info(f"config:{config}", is_logfile=True, log_to_console=False)

# 根据模型类型和任务类型选择损失函数
if args.num_tabs > 1:
    # 多标签分类任务使用多标签softmargin损失
    criterion = torch.nn.MultiLabelSoftMarginLoss()
else:
    # 单标签分类任务
    if args.optim:
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = torch.nn.CrossEntropyLoss()

# 如果启用优化调度器，则初始化学习率调度器
if args.optim:
    scheduler = LearningRateScheduler(optimizer, lr=float(config['learning_rate']),
                                      min_lr=args.min_lr, warmup_epochs=args.warmup_epochs,
                                      total_epochs=args.train_epochs)

# Train
# 初始化时间统计器
train_timmer = IncrementalMeanCalculator()
valid_timmer = IncrementalMeanCalculator()

logger.info("\n\n" + "-" * 20 + " start " + "-" * 20 + "\n", is_logfile=True)
# 训练循环
for epoch in range(args.train_epochs):
    start_time = time.time()
    model.train()
    sum_loss = 0
    sum_count = 0
    # 训练阶段
    for index, cur_data in enumerate(tqdm.tqdm(train_loader)):
        if args.optim:
            lr = scheduler.step(epoch + index / len(train_loader))
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        cur_y = cur_y.long()
        optimizer.zero_grad()
        outs = model(cur_X)

        # 根据模型类型计算损失
        loss = criterion(outs, cur_y)
        # 检查损失是否为NaN
        if torch.isnan(loss):
            print("loss is nan")
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    # 计算并记录训练损失
    train_loss = round(sum_loss / sum_count, 3)
    # print(f"epoch {epoch+1}: train_loss = {train_loss}")
    logger.log("train.loss", train_loss)
    logger.info(f"epoch {epoch + 1}: train_loss = {train_loss}", is_logfile=True)
    train_timmer.add(time.time() - start_time)

    # 验证阶段
    start_time = time.time()
    if args.num_tabs > 1:
        # 多标签分类任务的验证
        y_pred_score = np.zeros((0, num_classes))
        y_true = np.zeros((0, num_classes))
        with torch.no_grad():
            model.eval()
            for index, cur_data in enumerate(val_loader):
                cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
                cur_y = cur_y.long()
                outs = model(cur_X)
                y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
                y_true = np.append(y_true, cur_y.cpu().numpy(), axis=0)

            # 计算多标签分类的精度指标
            max_tab = 5
            tp = {}
            for tab in range(1, max_tab + 1):
                tp[tab] = 0

            for idx in range(y_pred_score.shape[0]):
                cur_pred = y_pred_score[idx]
                for tab in range(1, max_tab + 1):
                    target_webs = cur_pred.argsort()[-tab:]
                    for target_web in target_webs:
                        if y_true[idx, target_web] > 0:
                            tp[tab] += 1
            mapk = .0
            for tab in range(1, max_tab + 1):
                p_tab = tp[tab] / (y_true.shape[0] * tab)
                mapk += p_tab
                print(f"p@{tab}", round(p_tab, 4) * 100, epoch + 1)
                print(f"ap@{tab}", round(mapk / tab, 4) * 100, epoch + 1)

                if tab == args.num_tabs:
                    p_metric = round(p_tab, 4) * 100
                    ap_metric = round(mapk / tab, 4) * 100
                    valid_result = {f'p@{tab}': p_metric, f'ap@{tab}': ap_metric}
                    logger.log(f"valid.result", valid_result)
                    logger.info(f"{epoch + 1}: {round(mapk / tab, 4) * 100}", is_logfile=True)
                    should_stop = modelsaver.save(round(mapk / tab, 4) * 100, model, epoch + 1 + base_epoch, final=(epoch + 1) == args.train_epochs)

                    # 保存最佳模型
                    if ap_metric > metric_best_value:
                        metric_best_value = ap_metric
                        #torch.save(model.state_dict(), out_file)

            # 计算粗粒度指标
            y_pred_coarse = y_pred_score.argsort()[:, -2:]
            y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
            y_true_coarse = np.array(y_true_coarse)

            # valid_result metric_coarse_result
            valid_result = compute_metric(y_true_coarse, y_pred_coarse)
            print(valid_result)
    else:
        # 单标签分类任务的验证
        with torch.no_grad():
            model.eval()
            valid_pred = []
            valid_true = []

            for index, cur_data in enumerate(tqdm.tqdm(val_loader)):
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                cur_y = cur_y.long()
                outs = model(cur_X)
                if args.num_tabs == 1:
                    cur_pred = torch.argsort(outs, dim=1, descending=True)[:, 0]
                else:
                    cur_indices = torch.argmax(outs, dim=-1).cpu()
                    cur_pred = torch.zeros((cur_indices.shape[0], num_classes))
                    for cur_tab in range(cur_indices.shape[1]):
                        row_indices = torch.arange(cur_pred.shape[0])
                        cur_pred[row_indices, cur_indices[:, cur_tab]] = 1
                valid_pred.append(cur_pred.cpu().numpy())
                valid_true.append(cur_y.cpu().numpy())

            valid_pred = np.concatenate(valid_pred)
            valid_true = np.concatenate(valid_true)

        # 计算验证指标
        valid_result = measurement(valid_true, valid_pred, config['eval_metrics'])
        # print(f"{epoch+1}: {valid_result}")
        logger.log("valid.result", valid_result)
        logger.info(f"{epoch + 1}: {valid_result}", is_logfile=True)
        should_stop = modelsaver.save(valid_result["F1-score"], model, epoch + 1 + base_epoch, final=(epoch + 1) == args.train_epochs)
        # 保存最佳模型
        if valid_result["F1-score"] > metric_best_value:
            metric_best_value = valid_result["F1-score"]
            #torch.save(model.state_dict(), out_file)
    # 记录时间统计
    valid_timmer.add(time.time() - start_time)
    logger.log("time.train", train_timmer.get())
    logger.log("time.valid", valid_timmer.get())

    logger.info(f"epoch {epoch + 1}: time.train = {train_timmer.get():.2f}, time.valid = {valid_timmer.get():.2f}, F1 = {valid_result['F1-score']}", is_logfile=False)
    # 检查是否应该早停
    if should_stop:
        print("训练早停：达到最大停滞epoch次数")
        break

logger.info("\n\n" + "=" * 20 + " end " + "=" * 20 + "\n", is_logfile=True)