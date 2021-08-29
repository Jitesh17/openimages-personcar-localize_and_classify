
import os
from datetime import datetime

from jaitool.inference import D2Inferer # pip install jaitool

def infer(dir_path: str, infer_dump_dir: str, weights_path: str, threshold: float = 0.1, size: int = 1024, gt_path=None):
    inferer_seg = D2Inferer(
        weights_path=weights_path,
        confidence_threshold=threshold,
        class_names=["person", "car"],
        model="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        # model="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
        # size_min=size, size_max=size,
        detectron2_dir_path="/home/jitesh/lib/detectron2",
        
    )
    inferer_seg.infer(
        input_type="image_directory",
        # input_type="image",
        # output_type="show_image",
        output_type="write_image",
        input_path=dir_path,
        output_path=infer_dump_dir,
        text_size=0.7,
        show_class_label=True,
        # show_class_label_score_only=True,
        show_max_score_only=False,
        # color_bbox=[20, 200, 20],
        gt_path=gt_path,
        show_segmentation=False,
        thickness=2

    )


if __name__ == "__main__":
    now = datetime.now()
    # dt_string3 = now.strftime("%Y_%m_%d_%H_%M_%S")
    dt_string3 = now.strftime("%m_%d_%H")

    threshold = 0.7
    json_path = "trainval/annotations/bbox-annotations.json"
    dir_path = "/home/jitesh/jg/openimages-personcar-localize_and_classify/trainval/images_val"
    # dir_path = "/home/jitesh/jg/openimages-personcar-localize_and_classify/trainval/test"
    # weights_path = 'weights/cascade_rcnn/model_0004999.pth'
    # weights_path = 'weights/cascade_rcnn_aug/model_final.pth'
    weights_path = 'weights/faster_rcnn/model_final.pth'
    # _weights_path = weights_path.split('_')[-1].split('.')[0] + '_'
    _weights_path = dir_path.split('/')[-1] + '_'+ weights_path.split('/')[-2] + '_'\
        + weights_path.split('_')[-1].split('.')[0] 
    

    infer_dump_dir = f"{os.path.abspath(f'{dir_path}/..')}/inference_results"
    if not os.path.exists(infer_dump_dir):
        os.mkdir(infer_dump_dir)
    infer_dump_dir = f"{infer_dump_dir}/{_weights_path}_thres{threshold}_{dt_string3}"
    infer(dir_path=dir_path,
          infer_dump_dir=infer_dump_dir,
          weights_path=weights_path,
          threshold=threshold,
          gt_path=json_path,
          )
