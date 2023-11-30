!#bin/bash

mkdir -p data/ data/DIS5K data/cascade_psp data/thin_object_detection

wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/data/DIS5K.zip?download=true -O data/DIS5K.zip
unzip data/DIS5K.zip data/
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/data/cascade_psp.zip?download=true -O data/cascade_psp.zip
unzip data/cascade_psp.zip -d data/cascade_psp/
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/data/thin_object_detection.zip?download=true -O data/thin_object_detection.zip
unzip data/thin_object_detection.zip -d data/thin_object_detection/
