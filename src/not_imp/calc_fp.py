import os
from pycocotools.coco import COCO
from torch import mean
from cocoeval import COCOeval
# from pycocotools.cocoeval import COCOeval
import numpy as np
import json


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def run(path, gt_path=None, thres=0):
    with open(path) as f:
        pred_anns_raw = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    pred_anns = []
    for ann in pred_anns_raw:
        if ann["score"] >= thres:
            pred_anns.append(ann)
    total_predictions = len(pred_anns)

    tp = {i:0 for i in [1, 2]}
    gt = {i:0 for i in [1, 2]}
    pd = {i:0 for i in [1, 2]}
    fp = {i:0 for i in [1, 2]}
    for gt_anns in gt_data["annotations"]:
        gt_cat_id = gt_anns["category_id"]
        gt_img_id = gt_anns["image_id"]
        gt[gt_cat_id] += 1
        max_iou = -1
        for pred_ann in pred_anns:
            if gt[gt_cat_id] < 2:
                pd[gt_cat_id] += 1
            pred_cat_id = pred_ann["category_id"]
            pred_img_id = pred_ann["image_id"]
            if pred_cat_id == gt_cat_id and pred_img_id == gt_img_id:
                bbox = gt_anns["bbox"]
                gt_bbox = [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]
                bbox = pred_ann["bbox"]
                pred_bbox = [bbox[0], bbox[1],
                             bbox[2]+bbox[0], bbox[3]+bbox[1]]
                iou = get_iou(
                    {key:gt_bbox[i] for i, key in enumerate(['x1', 'y1', 'x2', 'y2'])},
                    {key:pred_bbox[i] for i, key in enumerate(['x1', 'y1', 'x2', 'y2'])})
                if max_iou < iou:
                    max_iou = iou
        if max_iou < 0.5:
            tp[gt_cat_id] += 1
    
    fn = {i:gt[i] - tp[i] for i in [1, 2]}
    print(gt)
    print(tp)
    print(fp)
    print(fn)



def main():
    # path = '/home/jitesh/3d/data/coco_data/hook_real2/img_07_09_13_Detection_h6_500_0099999_0.1_s1500_vis_infer_output_50_1x/infered_hook.json'
    # path = '/home/jitesh/3d/data/coco_data/hook_real2/img_07_12_23_Keypoints_h8_500_0099999_0.1_s1500_vis_infer_output_50_1x/infered_hook.json'
    # path = '/home/jitesh/3d/data/coco_data/hook_real2/img_07_15_10_Keypoints_h8_500_0049999_0.1_s1500_vis_infer_output_50_1x/infered_hook.json'
    path = '/home/jitesh/jg/openimages-personcar-localize_and_classify/trainval/inference_results/images_val_cascade_rcnn_aug_final_thres-1_08_29_11/result.json'
    # path = '/home/jitesh/jg/openimages-personcar-localize_and_classify/trainval/inference_results/cascade_rcnn_aug_final_thres0.7_08_29_00/result.json'
    # path = 'cascade_rcnn_aug_final_thres0.7_08_29_00'
    gt_path = "trainval/annotations/bbox-annotations_val.json"
    run(path, gt_path)


if __name__ == "__main__":
    main()
