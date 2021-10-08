from pathlib import Path

from .run import GROUPS

def script(group, gpu, stochastic=False, baseline=False, lambda_baseline=False):

    if stochastic:
        add_arg = '--stochastic '
        add_log = '_s'
    elif baseline:
        add_arg = '--baseline '
        add_log = '_b'
    elif lambda_baseline:
        add_arg = '--lambda_baseline'
        add_log = '_l'
    else:
        add_arg = ''
        add_log = ''

    return f"""CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=0 \\
    --experiment_seed=100 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_0_100.txt 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=0 \\
    --experiment_seed=1234 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_0_1234.txt 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=0 \\
    --experiment_seed=12345 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_0_12345.txt 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=1 \\
    --experiment_seed=100 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_1_100.txt 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=1 \\
    --experiment_seed=1234 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_1_1234.txt 2>&1 &
PID4=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=1 \\
    --experiment_seed=12345 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_1_12345.txt 2>&1 &
PID5=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=100 \\
    --experiment_seed=100 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_100_100.txt 2>&1 &
PID6=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=100 \\
    --experiment_seed=1234 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_100_1234.txt 2>&1 &
PID7=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=100 \\
    --experiment_seed=12345 \\
    --group='{group}' {add_arg}\\
    > out/logs/{group}{add_log}_100_12345.txt 2>&1 &
PID8=$!

wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8
"""

if __name__ == '__main__':
    par = 'scripts'
    loc = f'{par}/groups'
    Path(loc).mkdir(parents=True, exist_ok=True)
    for k in GROUPS.keys():
        for gpu in [0, 1]:
            with open(f'{loc}/{k}-gpu={gpu}.sh', 'w') as out:
                out.write(script(k, gpu))
    for k in GROUPS.keys():
        for gpu in [0, 1]:
            with open(f'{loc}/{k}-b-gpu={gpu}.sh', 'w') as out:
                out.write(script(k, gpu, baseline=True))
    for k in GROUPS.keys():
        for gpu in [0, 1]:
            with open(f'{loc}/{k}-s-gpu={gpu}.sh', 'w') as out:
                out.write(script(k, gpu, stochastic=True))
    for k in GROUPS.keys():
        for gpu in [0, 1]:
            with open(f'{loc}/{k}-l-gpu={gpu}.sh', 'w') as out:
                out.write(script(k, gpu, lambda_baseline=True))
    
    Path(f'{par}/run').mkdir(parents=True, exist_ok=True)
    with open(f'{par}/run/example.sh', 'w') as out:
        out.write(
"""sh scripts/groups/digits-gpu=0.sh
sh scripts/groups/digits_m-gpu=0.sh
"""
        )
    # make some directories that experiments expects to be there
    Path(f'out/logs').mkdir(parents=True, exist_ok=True)
    Path(f'out/results').mkdir(parents=True, exist_ok=True)
