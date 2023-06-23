batch_size=32
n_processes=8

# model_type='resnet18'
model_type=$1


jpeg=''
# CURRENTDATE=`date +"%Y-%m-%d-%T"`
CURRENTDATE='2023-06-06-18:04:08'
for VARIABLE in 1
do
    random_state=$RANDOM
    path=/home/nmuller/projects/oads_results/"${model_type}"/rgb/"${jpeg}""${CURRENTDATE}_CV/${random_state}"
    mkdir -p $path
    python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 50 --random_state $random_state --image_representation RGB --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes
done

for VARIABLE in {1..10}
do
    random_state=$RANDOM
    path=/home/nmuller/projects/oads_results/"${model_type}"/coc/"${jpeg}""${CURRENTDATE}_CV/${random_state}"
    mkdir -p $path
    python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 50 --random_state $random_state --image_representation COC --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes
done


jpeg='jpeg/'
for VARIABLE in {1..10}
do
    random_state=$RANDOM
    path=/home/nmuller/projects/oads_results/"${model_type}"/rgb/"${jpeg}""${CURRENTDATE}_CV/${random_state}"
    mkdir -p $path
    python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 50 --random_state $random_state --image_representation RGB --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes -use_jpeg
done

for VARIABLE in {1..10}
do
    random_state=$RANDOM
    path=/home/nmuller/projects/oads_results/"${model_type}"/coc/"${jpeg}""${CURRENTDATE}_CV/${random_state}"
    mkdir -p $path
    python train_dnn.py --input_dir /home/nmuller/projects/data/oads --output_dir $path --n_epochs 50 --random_state $random_state --image_representation COC --model_type $model_type --dataloader_path /home/nmuller/projects/oads_access/dataloader -new_dataloader --batch_size $batch_size --n_processes $n_processes -use_jpeg
done