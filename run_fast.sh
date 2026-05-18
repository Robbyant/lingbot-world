
set -x 

WEIGHT_DIR=$1
FRAME=$2

# case1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; torchrun --nproc_per_node=8 generate_fast.py \
                           --task i2v-A14B \
                           --size 480*832 \
                           --ckpt_dir ${WEIGHT_DIR} \
                           --image examples/03/image.jpg \
                           --action_path examples/03 \
                           --dit_fsdp \
                           --t5_fsdp \
                           --ulysses_size 8 \
                           --frame_num ${FRAME} \
                           --prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere."


# case2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; torchrun --nproc_per_node=8 generate_fast.py \
                           --task i2v-A14B \
                           --size 480*832 \
                           --ckpt_dir ${WEIGHT_DIR} \
                           --image examples/04/image.jpg \
                           --action_path examples/04 \
                           --dit_fsdp \
                           --t5_fsdp \
                           --ulysses_size 8 \
                           --frame_num ${FRAME} \
                           --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder."
