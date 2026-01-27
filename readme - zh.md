# LitePrint 项目文档

## 0. 环境配置

```bash
# LitePrint 安装环境
conda create -n liteprint python=3.10.18
conda activate liteprint
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# 其他依赖包
pip install tqdm
pip install pytorch_metric_learning
pip install captum
pip install pandas
pip install timm
pip install natsort
pip install noise
pip install transformers==4.53.2
pip install tabulate
pip install torchinfo

# 如果需要与 CountMamba 进行对比，请安装以下内容
# CountMamba 安装环境
# Mamba-ssm
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

---

## 1. 快速开始与数据集链接

* **Zenodo 仓库:** [https://zenodo.org/records/14195051](https://zenodo.org/records/14195051)
* **设置:** 下载后，解压并将数据集放入 `npz_dataset` 文件夹中。

### 项目文件结构

```plaintext
LitePrint
├── /dataset         # 原始流量文件 (Raw trace files)
├── /npz_dataset     # 处理后的 .npz 文件
│   ├── /CW
│   ├── /OW
│   └── ...
├── /data_process    # 转换与拆分脚本
└── /defense         # 防御模拟脚本

```

---

## 2. 数据集准备

<details>

### 2.1 下载原始数据集

在 `dataset/` 文件夹中准备以下内容：

* **DF:** 由 Tik-Tok 提供。
* **CW:** Undefended.zip
* **OW:** Undefended_OW.zip
* **Multi-Tab (ARES):** GitHub 仓库

### 2.2 处理原始数据集 (OW)

```bash
cd data_process
python check_format.py  # 手动修复非法文件的末尾: OW/5278340744671043543057

```

### 2.3 转换为 npz

<summary><b>点击查看转换脚本</b></summary>

```bash
cd data_process
# 标准数据集
python convert_to_npz.py --dataset CW
python convert_to_npz.py --dataset OW

# 多标签 (Multi-Tab) 数据集
python convert_multi_tab_npz.py --dataset Closed_2tab
python convert_multi_tab_npz.py --dataset Closed_3tab
python convert_multi_tab_npz.py --dataset Closed_4tab
python convert_multi_tab_npz.py --dataset Closed_5tab
python convert_multi_tab_npz.py --dataset Open_2tab
python convert_multi_tab_npz.py --dataset Open_3tab
python convert_multi_tab_npz.py --dataset Open_4tab
python convert_multi_tab_npz.py --dataset Open_5tab

```

### 2.4 数据集拆分

<summary><b>点击查看拆分脚本</b></summary>

```bash
cd data_process
for dataset in CW OW
do
  python dataset_split.py --dataset ${dataset}
done

for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab Open_2tab Open_3tab Open_4tab Open_5tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done

```

## </details>

## 3. 防御模拟

<details>

### 3.1 运行防御

<summary><b>点击展开各防御方法 (WTF-PAD, FRONT 等)</b></summary>

#### WTF-PAD

> 添加哑包，且不增加延迟。

```bash
cd defense/wtfpad
python main.py --traces_path "../../dataset/CW"
python main.py --traces_path "../../dataset/OW"

cd defense_npz/wtfpad
python main.py --traces_path "../../npz_dataset/Closed_2tab"
python main.py --traces_path "../../npz_dataset/Open_2tab"

```

#### FRONT

> 添加固定长度为 888 的哑包。无延迟。

```bash
cd defense/front
python main.py --p "../../dataset/CW"
python main.py --p "../../dataset/OW"

cd defense_npz/front
python main.py --p "../../npz_dataset/Closed_2tab"
python main.py --p "../../npz_dataset/Open_2tab"

```

#### RegulaTor

> 以时间敏感的方式传输数据包。

```bash
cd defense/regulartor
python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"

cd defense_npz/regulartor
python regulator_sim.py --source_path "../../npz_dataset/Closed_2tab" --output_path "../results/regulator_Closed_2tab"
python regulator_sim.py --source_path "../../npz_dataset/Open_2tab" --output_path "../results/regulator_Open_2tab"

```

#### TrafficSilver

> 使用不同的分配策略拆分流量。

```bash
# 轮询 (Round Robin)
cd defense/trafficsilver
python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_rb_CW/" --s round_robin
python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_rb_OW/" --s round_robin

# 按方向 (By Direction)
cd defense/trafficsilver
python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bd_CW/" --s in_and_out
python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bd_OW/" --s in_and_out

# 分批加权随机 (Batched Weighted Random)
cd defense/trafficsilver
python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1

```

### 3.2 防御后处理 (转换与拆分)

<summary><b>点击查看防御后数据的转换与拆分</b></summary>

```bash
cd data_process
# 转换防御后的数据
for dataset in wtfpad_CW front_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python convert_to_npz.py --dataset ${dataset}
done

# 拆分防御后的数据集
for dataset in wtfpad_CW front_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python dataset_split.py --dataset ${dataset}
done

for dataset in wtfpad_Closed_2tab wtfpad_Open_2tab front_Closed_2tab front_Open_2tab regulator_Closed_2tab regulator_Open_2tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done

```

## </details>

## 4. 模型训练与执行

### 4.1 设置工作目录

```bash
screen -S runner
conda activate liteprint
cd /root/autodl-tmp/lixianjun/LitePrint/LitePrint_run
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/lixianjun/LitePrint

```

### 4.2 模型运行脚本

<summary><b>点击查看执行脚本 (单流与多流)</b></summary>

#### 单流执行 - CW

```bash
for dataset in CW trafficsilver_bwr_CW trafficsilver_rb_CW trafficsilver_bd_CW wtfpad_CW front_CW regulator_CW
do
  for method in LitePrint
  do
    python main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
    python test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16
  done
done

```

#### 多流执行 - CW

```bash
for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab wtfpad_Closed_2tab front_Closed_2tab regulator_Closed_2tab Open_2tab Open_3tab Open_4tab Open_5tab wtfpad_Open_2tab front_Open_2tab regulator_Open_2tab
do
  for method in LitePrint
  do
    python main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
    python test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16
  done
done

```
