# 配置环境

``` bash	
# liteprint的安装环境
conda create -n liteprint python=3.10.18
conda activate liteprint
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/



# CountMamba的安装环境
# Mamba-ssm
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


```


# 单流运行 - CW

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

# 多流运行 - CW

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

# 消融实验

## No No 

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
python  main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim_ablation_no_no --drop_extra_time TRUE --TAM_type RF --Model_name LitePrint_no_LINA --num_workers 16 --optim True --train_epochs 100
python  test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim_ablation_no_no --drop_extra_time TRUE --TAM_type RF --Model_name LitePrint_no_LINA --num_workers 16
done
done
```
## SWIFT No 
```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
python  main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim_ablation_SWIFT_no --drop_extra_time TRUE --TAM_type SWIFT --Model_name LitePrint_no_LINA --num_workers 16 --optim True --train_epochs 100
python  test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim_ablation_SWIFT_no --drop_extra_time TRUE --TAM_type SWIFT --Model_name LitePrint_no_LINA --num_workers 16
done
done
```
## No LINA 
```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
python  main.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim_ablation_no_LINA --drop_extra_time TRUE --TAM_type RF --Model_name LitePrint --num_workers 16 --optim True --train_epochs 100
python  test.py --dataset ${dataset} --config config/${method}.ini --note baseline_optim_ablation_no_LINA --drop_extra_time TRUE --TAM_type RF --Model_name LitePrint --num_workers 16
done
done

```
## SWIFT LINA  

前面的实验已经包含，就是optim里面的内容

# 参数敏感度实验
## 不同模型参数效果
### 序列长度  

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for seq_len in 5000 10000 15000 20000
do
python  main.py  --optim TRUE --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_seq_len_${seq_len} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len ${seq_len} --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type Linear
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_seq_len_${seq_len} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len ${seq_len} --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type Linear
done
done
done
```

### 加载时长  

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for maximum_load_time in 60 120 180 240
do
python  main.py --optim True --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_maximum_load_time_${maximum_load_time} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time ${maximum_load_time} --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type Linear
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_maximum_load_time_${maximum_load_time} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time ${maximum_load_time} --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type Linear
done
done
done
```

### 时间窗口大小  

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for max_matrix_len in 2400 3600 7200 14400
do
python  main.py  --optim True --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_max_matrix_len_${max_matrix_len} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len ${max_matrix_len} --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type Linear
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_max_matrix_len_${max_matrix_len} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len ${max_matrix_len} --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type Linear
done
done
done
```

### 嵌入层大小  

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for embed_dim in 32 64 128 256
do
python  main.py  --optim True --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_embed_dim_${embed_dim} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim ${embed_dim} --num_heads 8 --r_of_lina 5 --atten_type Linear
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_embed_dim_${embed_dim} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim ${embed_dim} --num_heads 8 --r_of_lina 5 --atten_type Linear
done
done
done
```

### 注意力头数量 

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for num_heads in 4 8 12 16
do
python  main.py  --optim True --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_num_heads_${num_heads} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads ${num_heads} --r_of_lina 5 --atten_type Linear
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_num_heads_${num_heads} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads ${num_heads} --r_of_lina 5 --atten_type Linear
done
done
done
```

### LINA核大小 

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for r_of_lina in 5 10 15 20
do
python  main.py  --optim True --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_r_of_lina_${r_of_lina} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina ${r_of_lina} --atten_type Linear
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_r_of_lina_${r_of_lina} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen1 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina ${r_of_lina} --atten_type Linear
done
done
done
```

## 不同自注意力机制的效果
### 不同自注意力机制 

```bash
for dataset in Closed_5tab Open_5tab
do
for method in LitePrint
do
for atten_type in base topm Linear
do
python  main.py  --optim True --train_epochs 100 --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_atten_${atten_type} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen2 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type ${atten_type}
python  test.py --is_pr_auc TRUE --dataset ${dataset} --config config/${method}.ini --note baseline_optim_Sen_atten_${atten_type} --TAM_type SWIFT  --num_workers 16 --drop_extra_time TRUE --is_Sen TRUE --Model_name LitePrint_Sen2 --seq_len 10000 --maximum_load_time 120 --max_matrix_len 1800 --embed_dim 256 --num_heads 8 --r_of_lina 5 --atten_type ${atten_type}
done
done
done
```
