CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=100 \
    --group='r_digits' \
    > out/logs/r_digits_100.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=1234 \
    --group='r_digits' \
    > out/logs/r_digits_1234.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=12345 \
    --group='r_digits' \
    > out/logs/r_digits_12345.txt 2>&1 &
    
CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=100 \
    --group='n_digits' \
    > out/logs/n_digits_100.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=1234 \
    --group='n_digits' \
    > out/logs/n_digits_1234.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=12345 \
    --group='n_digits' \
    > out/logs/n_digits_12345.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=100 \
    --group='pacs_fts' \
    > out/logs/pacs_fts_100.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=1234 \
    --group='pacs_fts' \
    > out/logs/pacs_fts_1234.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=12345 \
    --group='pacs_fts' \
    > out/logs/pacs_fts_12345.txt 2>&1 &