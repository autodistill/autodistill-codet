import os
from dataclasses import dataclass

import subprocess
import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from .predictor import VisualizationDemo

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_dependencies():
    # Create the ~/.cache/autodistill directory if it doesn't exist
    autodistill_dir = os.path.expanduser("~/.cache/autodistill")
    os.makedirs(autodistill_dir, exist_ok=True)
    
    os.chdir(autodistill_dir)
    
    # Check if Detic is installed
    codet_path = os.path.join(autodistill_dir, "CoDet")

    if not os.path.isdir(codet_path):
        subprocess.run(["git", "clone", "https://github.com/CVMI-Lab/CoDet.git"])
        
        os.chdir(codet_path)

        os.chdir("third_party/detectron2")

        subprocess.run(["pip", "install", "-e", "."])

        os.chdir(codet_path)

        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        
        models_dir = os.path.join(codet_path, "checkpoints")
        os.makedirs(models_dir, exist_ok=True)
        # https://github.com/CVMI-Lab/CoDet/blob/main/configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml

        model_config_path = os.path.join(codet_path, "configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml")
        subprocess.run(["wget", "-O", model_config_path, "https://raw.githubusercontent.com/CVMI-Lab/CoDet/main/configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml"])

        model_weights_path = os.path.join(models_dir, "CoDet_OVLVIS_R5021k_4x_ft4x.pth")
        subprocess.run(["wget", "-O", model_weights_path, "https://doc-04-84-docs.googleusercontent.com/docs/securesc/9p5sospc1t58dc80kjk0m67ae8mqfmul/9gsnqb5a2ll7tukmch6h12c135k9vk27/1698875025000/07118889864120988423/03796184128890941253/1-chsmrh5fahOOSa4G2o5Mi6W2mGuMtG-?e=download&ax=AI0foUrKxjaTjBnAfniAli2Bh0S_2EGVJCIN6Kn0Wmrv1DDZ6ZPvoyzCby9rNytltWUzvXHVsH8oNCOwa0CgrfV-IQchAsGLOORoVyZC3BjEkWzUoGX5Tb205lTaEhJWHI0aq_rCpBryCMu1RUnKF-FhU6Yvv07QYvI0voCfZDrETLwwDOy3wQQzyZPMwPsCgsjxPqSE6-erbA3rXcRW9MhCImcLJ3URzJ8taiKHXnYPaxzRsPn5R8CJed3r6-Mh73gLUj542AlAlcDYmriqo5RG8GnxsVrUfoQQfA2kqZzJxajAnZG8EGhNgNDqrzm72V816PS-Ne2TMYwQnS0iu8xucq3eEz_eqmFSFo7l5WteK6SYYVd7uS2d1Z5uM31s0c9Z7czEnsxSP5pXi5XDCwCHtciru5KXZydjOXyaCYUM5mriaq1LxQXJCOn5v6SRCghz_E4K4iBjhBLEi7DtmsD9t2YOpI1fEquFYnBR3wwdFxJcP-ok-jdb9Ooh7K8s7ZmvFtO4snkBrKI2FZRi9z2THX-h-u1idHv3bri6ALxrjEtB0AuYy7qlEfjOvE83C_1EpYZ8z9yso8SVC8Aej0BlqbNVX7j-VK-Zkzu1z_lWDwN6hA2ngTy_G-kVc7BVZ8HcM3t1jQpPqBohx05obJX2vVwiFnXHoeTR8_tMzzQw3qM8VUgxifqQU6dnlfoq8R-oEeZMUL1U2bFFgk3Cll7v1Towewh2PV2RonCDeTcNrPGRKiyGFmJlFV5Zvb33-ADUJozaKbuS04bx69vXQgQeMJM7Lz195lr3yxWY4Hz1zKmr-tYWivWXpzTYqIHa9MZw9kFWrvZh9h9RfFdW9UjGGIA9eqyYVNFndrIv3ooBDzS9iy6No4k-r2akybrlcTg2Rw&uuid=21678dfe-9d64-46ff-b2d0-3de940ccb328&authuser=0"])

        # mkdor atasets/coco/
        os.makedirs("datasets/coco", exist_ok=True)
        # datasets/coco/annotations/
        os.makedirs("datasets/coco/annotations", exist_ok=True)
        # run python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
        # in correct path
        # print dir
        
        subprocess.run(["python3", "tools/get_coco_zeroshot_oriorder.py", "--data_path", "datasets/coco/zero-shot/instances_train2017_seen_2.json"])
        #python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
        subprocess.run(["python3", "tools/get_coco_zeroshot_oriorder.py", "--data_path", "datasets/coco/zero-shot/instances_val2017_all_2.json"])

        # python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
        subprocess.run([ "python3", "tools/remove_lvis_rare.py", "--ann", "datasets/lvis/lvis_v1_train.json"])

# check_dependencies()

import logging
import os
import sys
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

autodistill_dir = os.path.expanduser("~/.cache/autodistill")
codet_path = os.path.join(autodistill_dir, "CoDet")
center_net_path = os.path.join(codet_path, "third_party/CenterNet2")

sys.path.insert(0, codet_path)
sys.path.insert(0, center_net_path)

from centernet.config import add_centernet_config

deformation_path = os.path.join(codet_path, "third_party/Deformable-DETR")

sys.path.insert(0, deformation_path)

from codet.config import add_codet_config

logger = logging.getLogger("detectron2")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_codet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.MODEL.WEIGHTS = os.path.join(codet_path, "checkpoints/CoDet_OVLVIS_SwinB_4x_ft4x.pth")
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="codet")
    return cfg

@dataclass
class CoDet(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        args = default_argument_parser()
        args = args.parse_args()

        # set config_filename
        args.eval_only = True
        args.config_file = os.path.join(codet_path, "configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml")
        
        cfg = setup(args)
        # set path to codet path
        os.chdir(codet_path)

        # set to cpu
        
        self.model = build_model(cfg)

        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        self.cfg = cfg

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        from detectron2.data.detection_utils import read_image

        demo = VisualizationDemo(self.cfg)
        img = read_image(input, format="BGR")

        predictions, _ = demo.run_on_image(img)

        # open autodistill/CoDet/datasets/lvis/lvis_v1_train_norare_cat_info.json
        # map class names to ontology
        class_names = []

        import json

        with open(os.path.join(codet_path, "datasets/lvis/lvis_v1_train_norare_cat_info.json")) as f:
            data = json.load(f)
            for item in data:
                class_names.append(item["name"])

        predictions = sv.Detections.from_detectron2(predictions)
        predictions = predictions[predictions.confidence > confidence]

        return predictions, class_names