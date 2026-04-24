### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

### LOG INFO ###
#SBATCH --job-name=ebt-4xs-bs_256_s2_lr_
#SBATCH --output=logs/slurm/nlp/ebt-4xs-bs_256_s2_lr_%A-%a.log
export RUN_NAME="ebt-small-bs_256_s2_lr_0.0006"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/nlp/
module purge

lr=(0.0006)
alpha=(5)
alpha_random_scale=(2)
randomize_mcmc_num_steps=(2)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--tokenizer "EleutherAI/gpt-neox-20b" \
\
--no_mcmc_detach \
--mcmc_replay_buffer \
--mcmc_replay_buffer_size 16 \
--truncate_mcmc \
--langevin_dynamics_noise 3.0 \
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--randomize_mcmc_step_size_scale ${alpha_random_scale[$SLURM_ARRAY_TASK_ID]} \
--randomize_mcmc_num_steps ${randomize_mcmc_num_steps[${SLURM_ARRAY_TASK_ID}]} \
--randomize_mcmc_num_steps_min 2 \
--mcmc_num_steps 1 \
\
--context_length 128 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--prefetch_factor 8 \
--accumulate_grad_batches 4 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--max_epochs 6 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "lambada" \
--num_workers 12 \
--validation_split_pct 0.0005 \
--val_check_interval 1.0 \
\
--wandb_project 'nlp_pretrain' \
\
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--wandb_watch \
--no_wandb \
--compile_model \
\
--execution_mode "finetune" \
--finetuning_model_ckpt "/content/drive/MyDrive/EBT_models/EBT_small.ckpt" \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}