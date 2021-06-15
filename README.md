# Reproduction of the results

## 1. Initial setup on leonhard cluster
- Clone git repo
- Load the software modules: `module load python_gpu/3.8.5`
- Create virtual env and install packages:   
    - `cd ./DeepLearners/`
    - `python -m venv mp_env`
    - `source ./mp_env/bin/activate`
    - `pip install -r ./3dhumanpose_main/requirements.txt`


## 2. Adjust environment variables
- In the `./DeepLearners/3dhumanpose_main/.env` file the environment variables can be adjusted if needed
- By default, the environment variables are set to the correct paths on the leonhard cluster
  ```
  OUTPUT_PATH=./trainings
  DATA_COLLECTION=/cluster/project/infk/hilliges/lectures/mp21/project2/data
  VOC_DATASET=../voc_dataset
  ```

## 3. Loading environment
1. `cd ./DeepLearners/`
2. Load software modules: `module load python_gpu/3.8.5 tmux/2.6 eth_proxy`
3. Load python env: `source ./mp_env/bin/activate`

## 4. Run training
1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the human pose main folder `cd 3dhumanpose_main/`
3. Run a training job on the GPU using the python script `train.py`
   - All configuration files can be found in the folder `.configurations/`
   - Example to run a configuration: `python train.py --configuration ./configurations/<path-to-config>/<configuration-file>`
   - Leonhard commands to run different config files:
    
| Description | Datasets | Config file | Submission score | Command |
| ----------- | -------- |----------- | ---------------- | ------- |
| Best submission | h36m trainval, mpii trainval | `./configurations/...` | ... | `bsub -n 5 -W 120:00 -J "trainval" -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/..'` |
| Best trained only on training data set | h36m train, mpii train | `./configurations/...` | ... | `bsub -n 5 -W 120:00 -J "train" -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/..'` |

4. The result of the trainings can be found by default (see [2. Adjust environment variables](#2-adjust-environment-variables)) in the folder `./trainings`
   - The folders have following naming convention: `<start-datetime>-<configfile_name>`

## 5. Run inference
1. Load environment
2. Navigate to the main project folder `cd ./DeepLearners/3dhumanpose_main/` 
3. Run an inference job on the GPU using the python script `inference.py`
   - The command line argument `--run_folder` of the inference script `inference.py` takes the path to the trainings' folder created during training, for example: `--run_folder ./trainings/<start-datetime>-<configfile_name>`
   - Leonhard command to run an inference job:
     
     `bsub -n 5 -J "inference" -W 0:10 -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/<start-datetime>-<configfile_name>'`

4. During the inference job a folder called `prediction-<start-datetime>` is created inside the `run folder`. This folder will contain the submission file `test_pred.csv`.

## Training/Run folder structure
```
+-- trainings
    +-- <start-datetime>-<configfile_name>
        +-- prediction-<start-datetime>
        |   +-- <configfile>
        |   +-- test_pred.csv   
        +-- tensorboard
        |   +-- events.out.tfevents.*
        +-- weights_checkpoint
        |   +-- <epoch>_*.pth
        |   +-- ...
        +-- <configfile>
        +-- logs.txt
```