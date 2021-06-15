# Reproduction of the results

## 1. Initial setup and installation
- Clone this git repository `git clone https://gitlab.inf.ethz.ch/COURSE-MP2021/DeepLearners.git`
- Load the **leonhard** software modules: `module load gcc/6.3.0 python_gpu/3.8.5`
- Create a virtual environment and install the required python packages:   
    - `cd ./DeepLearners/`
    - `python -m venv mp_env`
    - `source ./mp_env/bin/activate`
    - `pip install -r ./3dhumanpose_main/requirements.txt`

## 2. Adjust environment variables
- In the `./DeepLearners/3dhumanpose_main/.env` file the environment variables can be adjusted if needed
- By default, the environment variables are set to the correct paths on the **leonhard** cluster
  ```
  OUTPUT_PATH=./trainings
  DATA_COLLECTION=/cluster/project/infk/hilliges/lectures/mp21/project2/data
  VOC_DATASET=../voc_dataset
  ```

## 3. Loading environment
1. `cd ./DeepLearners/`
2. Load the **leonhard** software modules: `module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy`
3. Load the python environment: `source ./mp_env/bin/activate`

## 4. Run training
1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the main project folder `./DeepLearners/3dhumanpose_main/`
3. Run a training job on the GPU using the python script `train.py`
   - All configuration files can be found in the folder `.configurations/`
   - Example to run a configuration: `python train.py --configuration ./configurations/<path-to-config>/<configuration-file>`
   - **Leonhard** commands to run different config files:
    
| Description | Datasets | Config file | Validation score | Submission score | Command |
| ----------- | -------- |------------ | ---------------- | ---------------- | ------- |
| Best submission | h36m trainval,</br>mpii trainval | `./configurations/best/regression_exp_findings_trainval.jsonc` | - | ... | `bsub -n 5 -W 120:00 -J "trainval" -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/best/regression_exp_findings_trainval.jsonc'` |
| Best trained only on training data set | h36m train,</br>mpii train | `./configurations/best/regression_exp_findings.jsonc` |  ... | ... | `bsub -n 5 -W 120:00 -J "train" -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/best/regression_exp_findings.jsonc'` |

4. The result of the trainings can be found by default (see [2. Adjust environment variables](#2-adjust-environment-variables)) in the folder `./trainings`
   - The folders have following naming convention: `<datetime>-<configfile_name>` (see [Training folder structure](#training-folder-structure))

## 5. Run inference
1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the main project folder `./DeepLearners/3dhumanpose_main/`
3. Run an inference job on the GPU using the python script `inference.py`
   - The command line argument `--run_folder` of the inference script `inference.py` takes the path to the trainings' folder created during training, for example: `--run_folder ./trainings/<datetime>-<configfile_name>`
   - **Leonhard** command to run an inference job:
     `bsub -n 5 -J "inference" -W 0:10 -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/<datetime>-<configfile_name>'`
4. During the inference job a folder called `prediction-<datetime>` is created inside the `run folder`. This folder will contain the submission file `test_pred.csv` (see [Training folder structure](#training-folder-structure)).

## Training folder structure
```
+-- trainings
    +-- <datetime>-<configfile_name>
        +-- prediction-<datetime>
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
