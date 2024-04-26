version=1024 #1024, ${version}, 256
seed=123
name=dynamicrafter_${version}_seed${seed}

ckpt=/kaggle/working/DynamiCrafter/checkpoints/dynamicrafter_${version}_v1/model.ckpt
config=/kaggle/working/DynamiCrafter/configs/inference_${version}_v1.0.yaml

prompt_dir=/kaggle/working/DynamiCrafter/prompts/${version}
res_dir=/kaggle/working/DynamiCrafter/results

H=576
FS=10 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)

CUDA_VISIBLE_DEVICES=0 python3 /kaggle/working/DynamiCrafter/scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width ${version} \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae True
