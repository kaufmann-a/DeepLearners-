#!/bin/bash

ln -s /cluster/project/infk/hilliges/lectures/mp21/project2/data ../data

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy

if [ -d "$HOME/mp-env" ]; then
    echo "Using cached venv \"$HOME/mp-env\""
else
    python -m venv "$HOME/mp-env"
    "$HOME/mp-env/bin/pip" install -r requirements.txt
fi

source "$HOME/mp-env/bin/activate"

bsub -n 4 -W $1 -J "mperc-job" -B -N \
    -R "rusage[mem=2048, ngpus_excl_p=1]" \
    -R "select[gpu_mtotal0>=10240]" \
    'python train.py --configuration ./configurations/default.jsonc'
bbjobs
