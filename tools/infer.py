import os
import numpy as np
import pickle as pkl

# from PIL import Image
import cv2
from paddle import inference
from utils.util import load_config
from utils.util_infer import Words
# from utils.process_ops import Words
# from dataset import   Words
class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args
    
        # init inference engine
        self.predictor, self.config= self.load_predictor(
            os.path.join(args.model_dir, "inference_faster.pdmodel"),
            os.path.join(args.model_dir, "inference_faster.pdiparams"))
      
    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        return predictor, config

    def preprocess(self, img_path):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        img=np.ones([1])
        if img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
              

        elif img_path.endswith('.pkl'):
            with open(img_path, "rb") as f:
                img = pkl.load(f)
                
        img = np.array(img/255).astype("float32")
     
        img = np.expand_dims(img, axis=0)
       
        return img

    def postprocess(self, x , Words):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after argmax.
        """

        result_seq = Words.decode(x)

        return result_seq

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        outputs=[]
        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])
        
        input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        for i in range(96):
            output_tensor = self.predictor.get_output_handle(output_names[i])
            output = output_tensor.copy_to_cpu()
            
            if output[0]==0:
                break
            outputs.append(output)
        return outputs

def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model_dir", default="./test_model/", help="inference model dir")
    parser.add_argument(
        "--use_gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--max_batch_size", default=16, type=int, help="max_batch_size")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")

    parser.add_argument(
        "--resize_size", default=256, type=int, help="resize_size")
    parser.add_argument("--img_path", default="./test_images/test_example/test_01.jpeg")

    parser.add_argument( 
        "--benchmark", default=False, type=str2bool, help="benchmark")
     
    # parser.add_argument("--word_path",default="../test_images/words_dict.txt",type=str,help="word_dict")
    parser.add_argument('--config_file', default="./config.yaml", help='config_file')

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="can_ocr",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = inference_engine.preprocess(args.img_path)
    img = img.reshape([args.batch_size , img.shape[0] ,img.shape[1] ,img.shape[2]])
    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()
    params=load_config(args.config_file)
    # result_seq = inference_engine.postprocess(output , Words(args.word_path))
    result_seq = inference_engine.postprocess(output , Words(params['word_path']))

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.img_path}, result_seq: {result_seq}")
    return result_seq


if __name__ == "__main__":
    args = get_args()
    result_seq = infer_main(args)