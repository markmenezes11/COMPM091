#$ -S /bin/bash
#$ -l h_rt=11:45:0 # Time (h:m:s)
#$ -l tmem=7.8G # Mem (xG or xM)
#$ -l gpu=1
#$ -P gpu

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# $@ --gpu_id $(nvidia-smi dmon -c 1 | grep -v "#" | awk '{ print $4 " " $1 }' | sort -n | awk '{ print $2 }' | tr -d "\n" | head -c 1)
$@ --gpu_id $CUDA_VISIBLE_DEVICES

