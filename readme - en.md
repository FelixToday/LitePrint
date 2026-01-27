# LitePrint Project Documentation

## 0. Environment Setup


``` bash
# LitePrint installation environment
conda create -n liteprint python=3.10.18
conda activate liteprint
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# other packages
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

# If you need to compare with CountMamba, install the following
# CountMamba installation environment
# Mamba-ssm
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

------

## 1. Quick Start & Dataset Links

- **Zenodo Repository:** https://zenodo.org/records/14195051
- **Setup:** After downloading, extract and place the datasets into the `npz_dataset` folder.

### Project File Structure

```Plaintext
LitePrint
├── /dataset           # Raw trace files
├── /npz_dataset       # Processed .npz files
│   ├── /CW
│   ├── /OW
│   └── ...
├── /data_process      # Scripts for conversion and splitting
└── /defense           # Scripts for defense simulation
```

------

## 2. Dataset Preparation

<details>

### 2.1 Download Raw Datasets

Prepare the following in the `dataset/` folder:

- **DF:** Provided by Tik-Tok.
- **CW:** Undefended.zip
- **OW:** Undefended_OW.zip
- **Multi-Tab (ARES):** GitHub Repository

### 2.2 Process Raw Dataset (OW)

```Bash
cd data_process
python check_format.py  # Manually fix the tail of the illegal file: OW/5278340744671043543057
```

### 2.3 Convert to npz



<summary><b>Click to view conversion scripts</b></summary>

```Bash
cd data_process
# Standard Datasets
python convert_to_npz.py --dataset CW
python convert_to_npz.py --dataset OW

# Multi-Tab Datasets
python convert_multi_tab_npz.py --dataset Closed_2tab
python convert_multi_tab_npz.py --dataset Closed_3tab
python convert_multi_tab_npz.py --dataset Closed_4tab
python convert_multi_tab_npz.py --dataset Closed_5tab
python convert_multi_tab_npz.py --dataset Open_2tab
python convert_multi_tab_npz.py --dataset Open_3tab
python convert_multi_tab_npz.py --dataset Open_4tab
python convert_multi_tab_npz.py --dataset Open_5tab
```


### 2.4 Dataset Split


<summary><b>Click to view splitting scripts</b></summary>

```Bash
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

</details>
------

## 3. Defense Simulation

<details>

### 3.1 Run Defenses

<summary><b>Click to expand individual defense methods (WTF-PAD, FRONT, etc.)</b></summary>

#### WTF-PAD

> Adds dummy packets with no added latency.

```Bash
cd defense/wtfpad
python main.py --traces_path "../../dataset/CW"
python main.py --traces_path "../../dataset/OW"

cd defense_npz/wtfpad
python main.py --traces_path "../../npz_dataset/Closed_2tab"
python main.py --traces_path "../../npz_dataset/Open_2tab"
```

#### FRONT

> Adds dummy packets with a fixed length of 888. No latency.

```Bash
cd defense/front
python main.py --p "../../dataset/CW"
python main.py --p "../../dataset/OW"

cd defense_npz/front
python main.py --p "../../npz_dataset/Closed_2tab"
python main.py --p "../../npz_dataset/Open_2tab"
```

#### RegulaTor

> Transmits packets in a time-sensitive manner.

```Bash
cd defense/regulartor
python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"

cd defense_npz/regulartor
python regulator_sim.py --source_path "../../npz_dataset/Closed_2tab" --output_path "../results/regulator_Closed_2tab"
python regulator_sim.py --source_path "../../npz_dataset/Open_2tab" --output_path "../results/regulator_Open_2tab"
```

#### TrafficSilver

> Split traffic using different distribution strategies.

```Bash
# Round Robin
cd defense/trafficsilver
python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_rb_CW/" --s round_robin
python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_rb_OW/" --s round_robin

# By Direction
cd defense/trafficsilver
python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bd_CW/" --s in_and_out
python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bd_OW/" --s in_and_out

# Batched Weighted Random
cd defense/trafficsilver
python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1
```

### 3.2 Post-Defense Processing (Convert & Split)

<summary><b>Click to view conversion and splitting for defended data</b></summary>

```Bash
cd data_process
# Convert Defended Data
for dataset in wtfpad_CW front_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python convert_to_npz.py --dataset ${dataset}
done

# Split Defended Datasets
for dataset in wtfpad_CW front_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python dataset_split.py --dataset ${dataset}
done

for dataset in wtfpad_Closed_2tab wtfpad_Open_2tab front_Closed_2tab front_Open_2tab regulator_Closed_2tab regulator_Open_2tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done
```

</details>
------

## 4. Model Training & Execution

### 4.1 Set Up Working Directory

```Bash
screen -S runner
conda activate liteprint
cd /root/autodl-tmp/lixianjun/LitePrint/LitePrint_run
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/lixianjun/LitePrint
```

### 4.2 Model Running Scripts


<summary><b>Click to view Execution Scripts (Single-stream & Multi-stream)</b></summary>

#### Single-stream Execution - CW

```Bash
for dataset in CW trafficsilver_bwr_CW trafficsilver_rb_CW trafficsilver_bd_CW wtfpad_CW front_CW regulator_CW
do
  for method in LitePrint
  do
    python main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
    python test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16
  done
done
```

#### Multi-stream Execution - CW

```Bash
for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab wtfpad_Closed_2tab front_Closed_2tab regulator_Closed_2tab Open_2tab Open_3tab Open_4tab Open_5tab wtfpad_Open_2tab front_Open_2tab regulator_Open_2tab
do
  for method in LitePrint
  do
    python main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
    python test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16
  done
done
```
