CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=100 \
    --group='digits' \
    > out/logs/digits_100.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=1234 \
    --group='digits' \
    > out/logs/digits_1234.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=12345 \
    --group='digits' \
    > out/logs/digits_12345.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=100 \
    --group='f_digits' \
    > out/logs/f_digits_100.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=1234 \
    --group='f_digits' \
    > out/logs/f_digits_1234.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=12345 \
    --group='f_digits' \
    > out/logs/f_digits_12345.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=100 \
    --group='officehome_fts' \
    > out/logs/officehome_fts_100.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=1234 \
    --group='officehome_fts' \
    > out/logs/officehome_fts_1234.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m experiments.run \
    --device='cuda:0' \
    --dataset_seed=0 \
    --experiment_seed=12345 \
    --group='officehome_fts' \
    > out/logs/officehome_fts_12345.txt 2>&1 &