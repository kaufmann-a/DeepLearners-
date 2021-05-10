## Initial setup on leonhard cluster
- Clone git repo
- Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5`
- Create virtual env and install packages:   
    - `cd ./DeepLearners/`
    - `python -m venv mp_env`
    - `source ./mp_env/bin/activate`
    - `pip install -r ./3dhumanpose_main/requirements.txt`
- Setup data folder path
    - `cd ./DeepLearners/`
    - delete data folder `rm ./data -r`
    - create symbolic link to data `ln -s /cluster/project/infk/hilliges/lectures/mp21/project2/data`

## General: loading environment
1. `cd ./DeepLearners/`
2. Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6`
3. Load python env: `source ./mp_env/bin/activate`

## Run training
1. Load environment
2. Navigate to the road segmentation folder `cd 3dhumanpose_main/`
3. Edit the configuration file to your needs
    - `vim ./configurations/default.jsonc`
4. Run job on GPU
  - `bsub -n 4 -W 6:00 -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python train.py --configuration ./configurations/default.jsonc'`
  - check job status `bbjobs` 
  - peek stdout log `bpeek`
    - `bpeek -f` to actively peek
5. Find your training results with `ls ./trainings/`
