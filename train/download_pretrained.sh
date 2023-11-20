!#bin/bash

mkdir pretrained_checkpoint/
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_b_01ec64.pth?download=true -O pretrained_checkpoint/sam_vit_b_01ec64.pth
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_b_maskdecoder.pth?download=true -O pretrained_checkpoint/sam_vit_b_maskdecoder.pth
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_4b8939.pth?download=true -O pretrained_checkpoint/sam_vit_h_4b8939.pth
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_maskdecoder.pth?download=true -O pretrained_checkpoint/sam_vit_h_maskdecoder.pth
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_l_0b3195.pth?download=true -O pretrained_checkpoint/sam_vit_l_0b3195.pth
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_l_maskdecoder.pth?download=true -O pretrained_checkpoint/sam_vit_l_maskdecoder.pth
