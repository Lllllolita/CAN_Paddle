CAN_DIR=$(cd $(dirname $0);cd ..; pwd)
echo $CAN_DIR
export PYTHONPATH=$PYTHONPATH:$CAN_DIR
#export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch tools/train.py --dist