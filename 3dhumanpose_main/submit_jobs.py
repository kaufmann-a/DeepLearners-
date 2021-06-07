import os
import subprocess
import sys

if __name__ == '__main__':

    directory = "configurations/bottleneck"
    GPU_2080TI = False
    TIME = '19:00'
    DEBUG = False
    NR_CPUS = 5
    MEM_PER_CPU = 4096

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_path = os.path.join(directory, filename)
        if filename.endswith(".jsonc"):
            command = ''
            if DEBUG:
                command += 'echo '  # test output

            # use 4 cpus
            command += 'bsub -n ' + str(NR_CPUS) + ' -J "' + filename[:-6] + '"'
            # job time
            command += ' -W ' + TIME
            # memory per cpu and select one gpu
            command += ' -R "rusage[mem=' + str(MEM_PER_CPU) + ', ngpus_excl_p=1]"'
            if GPU_2080TI:
                command += ' -R "select[gpu_model0==GeForceRTX2080Ti]"'
            else:
                command += ' -R "select[gpu_mtotal0>=10240]"'  # GPU memory more then 10GB

            command += " 'python train.py --configuration " + file_path + "'"

            print(command)

            # new method
            # process = subprocess.run(command.split(), stdout=sys.stdout, stderr=sys.stderr, shell=True)

            # old method
            os.system(command)
