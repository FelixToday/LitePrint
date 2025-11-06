import argparse
import configparser
import numpy as np
import random
import tqdm
import torch
import os
import torch.backends.cudnn as cudnn
from util import parse_value
from util import measurement, compute_metric
import warnings

from lxj_utils_sys import BaseLogger, str_to_bool, print_dict,ModelCheckpoint
from LitePrint import get_model_and_dataloader, load_data

from LitePrint.const import dataset_lib

warnings.filterwarnings("ignore")

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)
rng = np.random.RandomState(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Config
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument('--dataset', default="Closed_5tab")# trafficsilvers_bwr_CW
parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
parser.add_argument('--config', default="config/LitePrint.ini")
parser.add_argument("--num_tabs", type=int, default=2)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--note", type=str, default='test')
parser.add_argument("--load_ratio", type=float, default=100)
parser.add_argument("--test_flag", type=str_to_bool, default=False)
parser.add_argument("--load_checkpoint", type=str_to_bool, default=False)
parser.add_argument("--TAM_type", type=str, default='SWIFT')
parser.add_argument("--Model_name", type=str, default='LitePrint')
parser.add_argument("--maximum_load_time", type=float, default=120)
parser.add_argument("--drop_extra_time", type=str_to_bool, default=False)

parser.add_argument("--minimum_packet_number", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--is_pr_auc", type=str_to_bool, default=False)

parser.add_argument('--is_Sen', type=str_to_bool, default=False)
# 1. 5k 10k 15k 20k
parser.add_argument('--seq_len', type=int, default=5000)
# 2. maximum_load_time 80 120 180 240
# 3. 2400 3600 4800 7200 (120s)
parser.add_argument('--max_matrix_len', type=int, default=1800)
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
print_dict(vars(args))
print_dict(config)
# Preparation
device = torch.device("cuda")

in_path = os.path.join("../npz_dataset", args.dataset)
ckp_path = os.path.join(
    "../checkpoints",
    args.dataset,
    config['model'],
    args.note
).rstrip('/')  # 移除末尾可能多余的斜杠

test_path = os.path.join(str(ckp_path), f'test_p{args.load_ratio}')

logger=BaseLogger(json_save_path=os.path.join(test_path,"result.json"),
                  log_save_path=os.path.join(test_path,"log.txt"))

valid_X, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))
if args.load_ratio != 100 and False:
    test_X, test_y = load_data(os.path.join(in_path, f"test_p{args.load_ratio}.npz"), drop_extra_time=args.drop_extra_time, load_time=args.maximum_load_time)
else:
    test_X, test_y = load_data(os.path.join(in_path, f"test.npz"), drop_extra_time=args.drop_extra_time, load_time=args.maximum_load_time)
if args.num_tabs == 1:
    num_classes = len(np.unique(test_y))
else:
    num_classes = test_y.shape[1]

model, val_loader, test_loader = get_model_and_dataloader(valid_X,valid_y,
                                                            test_X,test_y,
                                                            num_classes, config, vars(args), num_workers=args.num_workers)

modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                             mode='max',
                             metric_name='f1')
model = modelsaver.load(model, device)[0]
#model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))



if args.num_tabs > 1:
    y_pred_score = np.zeros((0, num_classes))
    y_true = np.zeros((0, num_classes))
    with torch.no_grad():
        model.eval()
        for index, cur_data in enumerate(tqdm.tqdm(test_loader)):
            cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
            outs = model(cur_X)
            y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
            y_true = np.append(y_true, cur_y.cpu().numpy(), axis=0)

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
        topk_result = {}
        for tab in range(1, max_tab + 1):
            p_tab = tp[tab] / (y_true.shape[0] * tab)
            mapk += p_tab

            topk_result[f"p@{tab}"] = round(p_tab, 4) * 100
            topk_result[f"ap@{tab}"] = round(mapk / tab, 4) * 100

            if tab == args.num_tabs:
                result = {
                    f"p@{tab}": round(p_tab, 4) * 100,
                    f"ap@{tab}": round(mapk / tab, 4) * 100
                }

        y_pred_coarse = y_pred_score.argsort()[:, -2:]
        y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
        y_true_coarse = np.array(y_true_coarse)

        metric_coarse_result = compute_metric(y_true_coarse, y_pred_coarse)
        #result.update(metric_coarse_result)
    logger.log('test.topk_metrics', topk_result)
else:
    with torch.no_grad():
        model.eval()
        valid_pred = []
        valid_true = []

        for index, cur_data in enumerate(tqdm.tqdm(test_loader)):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)

            outs = model(cur_X)
            outs = torch.argsort(outs, dim=1, descending=True)[:, 0]
            valid_pred.append(outs.cpu().numpy())
            valid_true.append(cur_y.cpu().numpy())

        valid_pred = np.concatenate(valid_pred)
        valid_true = np.concatenate(valid_true)

    result = measurement(valid_true, valid_pred, config['eval_metrics'])

#print(result)
logger.log('test.metrics', result)
logger.info("Test metrics: %s", result, is_logfile=True)

if args.is_pr_auc:# 计算AUC值 和 PR曲线的值
    from lxj_utils_sys import compute_pr_result
    pr_auc_result = compute_pr_result(dataloader=test_loader,
                              model=model,
                              task_type="multilabel",
                              average="micro")
    logger.log('test.pr_auc', pr_auc_result)
print(result)
