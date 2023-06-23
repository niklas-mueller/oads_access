batch_size=64
n_processes=8

# model_type='resnet18'
model_type=$1
CURRENTDATE=`date +"%Y-%m-%d-%T"`
# CURRENTDATE='2023-06-08-11:54:21'

jpeg=''
path=/home/nmuller/projects/oads_results/"${model_type}"/rgb/"${jpeg}""${CURRENTDATE}"
mkdir -p $path
python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 60 --image_representation RGB --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes



path=/home/nmuller/projects/oads_results/"${model_type}"/coc/"${jpeg}""${CURRENTDATE}"
mkdir -p $path
python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 60 --image_representation COC --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes



jpeg='jpeg/'
path=/home/nmuller/projects/oads_results/"${model_type}"/rgb/"${jpeg}""${CURRENTDATE}"
mkdir -p $path
python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 60 --image_representation RGB --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes -use_jpeg



path=/home/nmuller/projects/oads_results/"${model_type}"/coc/"${jpeg}""${CURRENTDATE}"
mkdir -p $path
python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 60 --image_representation COC --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes -use_jpeg