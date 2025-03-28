#!/bin/sh
# embedded options to bsub - start with #BSUB
### -- set the job Name AND the job array --
#BSUB -J job_name[1-10]
### –- specify queue -- 
#BSUB -q hpc 
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- set walltime limit: hh:mm --
#BSUB -W 2:00 
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id %I is the job-array index --
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
### BSUB -o ./log/Output_%J_%I.out
### BSUB -e ./log/Output_%J_%I.err 
# BSUB -o ./log/Output_%I.out
# BSUB -e ./log/Output_%I.err 
# here follow the commands you want to execute 
# Program_name_and_options
source /zhome/5f/1/167776/playground/bin/activate

python ensemble_vae.py trainEnsamble --num-decoders 1 --model-nr $LSB_JOBINDEX --device cpu --latent-dim 2 --epochs 100 --batch-size 64
python ensemble_vae.py trainEnsamble --num-decoders 2 --model-nr $LSB_JOBINDEX --device cpu --latent-dim 2 --epochs 100 --batch-size 64
python ensemble_vae.py trainEnsamble --num-decoders 3 --model-nr $LSB_JOBINDEX --device cpu --latent-dim 2 --epochs 100 --batch-size 64
python ensemble_vae.py trainEnsamble --num-decoders 4 --model-nr $LSB_JOBINDEX --device cpu --latent-dim 2 --epochs 100 --batch-size 64





