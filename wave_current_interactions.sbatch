#!/bin/bash
#SBATCH --job-name=test_python
#SBATCH --error=err.err   
#SBATCH --output=log.log 
#SBATCH --ntasks=1                                      
#SBATCH --nodes=1                                               
#SBATCH --cpus-per-task=1                                 

shopt -s expand_aliases
source /home/hse/ivointsev/.bashrc

path_to_project="${HOME}/wave_current_interactions/wave-current-interaction"
export PYTHONPATH=$path_to_project:$PYTHONPATH

activate_main
srun python ${path_to_project}/wave_current_interactions.py numerical -p ./numerical_problem_ho_1.00e-02_epsilon_5.00e-02.pkl -c ./wave_current_interactions_conf.py --show_progress
deactivate


