torchrun \
--nproc_per_node=8 generate.py \
--task i2v-A14B --size 480*832 \
--ckpt_dir lingbot-world-base-cam \
--image examples/05/image.jpg \
--action_path examples/05 \
--allow_act2cam \
--sample_steps 20 \
--dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 121 \
--prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."