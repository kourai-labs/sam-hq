# Training instruction for HQ-SAM

> [**Segment Anything in High Quality**](https://arxiv.org/abs/2306.01567)           
> Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu \
> ETH Zurich & HKUST 

We organize the training folder as follows.
```
train
|____data
|____pretrained_checkpoint
|____train.py
|____utils
| |____dataloader.py
| |____misc.py
| |____loss_mask.py
|____segment_anything_training
|____work_dirs
```

## 1. Data Preparation

HQSeg-44K can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)

Run the `download_dataset.sh` to download all datasets.
```
sh download_dataset.sh
```

### Expected dataset structure for HQSeg-44K

```
data
|____DIS5K
|____cascade_psp
| |____DUTS-TE
| |____DUTS-TR
| |____ecssd
| |____fss_all
| |____MSRA_10K
|____thin_object_detection
| |____COIFT
| |____HRSOD
| |____ThinObject5K
|____TestSAM (symlink from sample_data/)

```

## 2. Init Checkpoint
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

Run the `download_pretrained.sh` to download all pretrained checkpoints
```
sh download_pretrained.sh
```

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b_01ec64.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h_4b8939.pth

```

## 3. Training
### Modify Training Script

#### 1. Add train/validation dataset
At the bottom of `main.py`, create a `dict` to define the dataset configuration and append it to `train_datasets` or `valid_datasets` list.
```python
dataset_hoarder = {"name": "TestSAM",
            "im_dir": "./data/TestSAM",
            "gt_dir": "",
            "im_ext": ".jpg",
            "gt_ext": ".png"}

train_datasets = [dataset_train1, dataset_train2,  dataset_hoarder]
```

To finetune without the original datasets, just change the `train_datasets` or `valid_datasets` list.
```python
train_datasets = [dataset_hoarder_train1, dataset_hoarder_train2]
valid_datasets = [dataset_hoarder_train1, dataset_hoarder_train2]
```

#### 2. Modify Dataloader
The current implementation doesn't support other hoarder datasets, only filter by specific name (see utils/dataloader.py#L47, utils/dataloader.py#L78). Need to change or modify the function to support generic hoarder dataset.

Change `TestSAM` in `utils/dataloader.py#L47` and `utils/dataloader.py#L78` to other dataset name.

To train HQ-SAM on HQSeg-44K dataset

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>

# PREFERRABLE DUE TO UNKNOWN ERROR IN PYTORCH DISTRIBUTED
torchrun --nproc_per_node=<num_gpus> train.py --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type vit_h --output work_dirs/hq_sam_h
```

### Example HQ-SAM-L training script
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_sam_l
```

### Example HQ-SAM-B training script
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b
```

### Example HQ-SAM-H training script
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type vit_h --output work_dirs/hq_sam_h

(optional for single gpu) WORLD_SIZE='' torchrun --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type vit_h --output work_dirs/hq_sam_h

```

## 4. Evaluation
To evaluate on 4 HQ-datasets

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```

### Example HQ-SAM-L evaluation script
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_sam_l --eval --restore-model work_dirs/hq_sam_l/epoch_11.pth
```

### Example HQ-SAM-L visualization script
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_sam_l --eval --restore-model work_dirs/hq_sam_l/epoch_11.pth --visualize
```