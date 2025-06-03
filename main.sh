GPUS=$1
python3 -m torch.distributed.launch --nproc_per_node=$GPUS /kaggle/working/YOLOv11-pt/main.py ${@:2}
