version="512" #1024, 512, 256
seed=123
name=dynamicrafter_512_seed${seed}

ckpt=/kaggle/working/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt
config=/kaggle/working/DynamiCrafter/configs/inference_512_v1.0.yaml

input_name=$(basename /kaggle/input/*)

prompt_dir=/kaggle/input/$input_name
res_dir=/kaggle/working/DynamiCrafter/results

H=320
FS=24 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)

CUDA_VISIBLE_DEVICES=0 python3 /kaggle/working/DynamiCrafter/scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
