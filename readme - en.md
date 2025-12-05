

# Environment Setup

``` bash
# LitePrint installation environment
conda create -n liteprint python=3.10.18
conda activate liteprint
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/


# If you need to compare with CountMamba, install the following
# CountMamba installation environment
# Mamba-ssm
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

# Dataset

All datasets can be downloaded at https://zenodo.org/records/14195051
Then drop the dataset into the npz_dataset project folder below
File Structure:
LitePrint
    /npz_dataset
        /CW
        /OW
        ...
        
# Single-stream Execution - CW

``` bash
for dataset in  CW trafficsilver_bwr_CW trafficsilver_rb_CW trafficsilver_bd_CW wtfpad_CW front_CW regulator_CW
do
for method in LitePrint
do
python  main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
python  test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16
done
done
```

# Multi-stream Execution - CW

``` bash
for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab wtfpad_Closed_2tab front_Closed_2tab regulator_Closed_2tab Open_2tab Open_3tab Open_4tab Open_5tab wtfpad_Open_2tab front_Open_2tab regulator_Open_2tab
do
for method in LitePrint
do
python  main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
python  test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim --drop_extra_time True --TAM_type SWIFT --Model_name LitePrint --num_workers 16
done
done
```