CAN_DIR=$(cd $(dirname $0);cd ..; pwd)
echo $CAN_DIR
export PYTHONPATH=$PYTHONPATH:$CAN_DIR
python tools/infer.py --use_gpu True