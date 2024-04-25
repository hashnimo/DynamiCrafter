version="1024" ##1024, 512, 256
seed=123
name=dynamicrafter_1024_seed${seed}

ckpt=/content/DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt
config=/content/DynamiCrafter/configs/inference_1024_v1.0.yaml

prompt_dir=/content/DynamiCrafter/prompts/1024
res_dir=/content/DynamiCrafter/results

H=576
FS=10 ## This model adopts FPS=10, range recommended: 15-5 (smaller value -> larger motion)

CUDA_VISIBLE_DEVICES=0 python3 /content/DynamiCrafter/scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width 1024 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae

## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop