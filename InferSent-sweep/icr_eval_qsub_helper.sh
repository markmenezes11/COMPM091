#$ -S /bin/bash
#$ -l h_rt=23:30:0 # Time (h:m:s)
#$ -l tmem=47.1G # Mem (xG or xM)
#$ -l gpu=1
#$ -P gpu

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
$@ --gpu_id $(nvidia-smi dmon -c 1 | grep -v "#" | awk '{ print $4 " " $1 }' | sort -n | awk '{ print $2 }' | tr -d "\n" | head -c 1)

# TODO: Set GPU ID correctly based on CUDA_VISIBLE_DEVICES (even though it seems to be broken...)
