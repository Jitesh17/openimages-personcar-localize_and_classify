# %%

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from utils import CustomTrainer

img_dir = "trainval/images"
json_path = "trainval/annotations/bbox-annotations.json"
json_path_split = json_path.split(".")
train_json_path = ".".join(json_path_split[:-1]) + "_train.json"
val_json_path = ".".join(json_path_split[:-1]) + "_val.json"

class_names = ["person", "car"]
register_coco_instances("my_dataset_train", {}, train_json_path, img_dir)
register_coco_instances("my_dataset_val", {}, val_json_path, img_dir)
MetadataCatalog.get("my_dataset_train").thing_classes = class_names
MetadataCatalog.get("my_dataset_val").thing_classes = class_names

# %%

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0001  
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = [10001]        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 


cfg.TEST.EVAL_PERIOD = 500
cfg.VIS_PERIOD = 500
cfg.SOLVER.CHECKPOINT_PERIOD = 500

cfg.OUTPUT_DIR = "weights/2_aug0"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg, "src/aug1.json") 
trainer.resume_or_load(resume=True)
trainer.scheduler.milestones = cfg.SOLVER.STEPS
trainer.train()
# %%

im = cv2.imread("trainval/images/image_000000021.jpg")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
# # %%
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# %%
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
output = out.get_image()[:, :, ::-1]
cv2.imwrite("output.png", output)
# cv2.imshow(out.get_image()[:, :, ::-1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# %%
# import matplotlib.pyplot as plt
# imgplot = plt.imshow(output[:, :, ::-1])
# # %%
