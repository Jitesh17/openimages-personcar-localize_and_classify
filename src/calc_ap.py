import os
from pycocotools.coco import COCO
from torch import mean
from cocoeval import COCOeval
# from pycocotools.cocoeval import COCOeval
# import numpy as np
# import json

def run(path, gt_path=None):
    # gt_path = os.path.abspath(f'{path}/../../json/gt.json')
    gt_dataset = COCO(annotation_file=gt_path)
    # best_metric = best_thres = 0
    # bb=None
    # li = []
    for thres in [i*0.1 for i in range(10)]:
        thres=-1
        dt_dataset = gt_dataset.loadRes(path, thres)
        # eval_output = os.path.abspath(f'{path}/../eval.txt')
        # evaluator = COCOeval(cocoGt=gt_dataset, cocoDt=dt_dataset, iouType='keypoints')
        # evaluator = COCOeval(cocoGt=gt_dataset, cocoDt=dt_dataset, iouType='keypoints')
        # evaluator.params.useSegm = None
        # evaluator.params.kpt_oks_sigmas = np.array([1.0]*12)/12
        
        evaluator = COCOeval(cocoGt=gt_dataset, cocoDt=dt_dataset, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        # with open(path) as f:
        #     anns_raw = json.load(f)
        
        # anns = []
        # for ann in anns_raw:
        #     if ann["score"] >= thres:
        #         anns.append(ann)
        # total_predictions = len(anns)
        # keys = list(evaluator.ious.keys())
        # ious = evaluator.ious
        # # iou_list = [iou_i for key in keys for iou in evaluator.ious[key] for iou_i in iou]
        # iou_list = [max(iou) for key in keys for iou in evaluator.ious[key]]
        # print(evaluator.ious)
        # print(len(iou_list))
        # print(len(gt_dataset.dataset["annotations"]))
        # print(total_predictions)
        # tp = sum(1 for iou in iou_list if iou > 0.5)
        # print(tp)

        # print(evaluator.ious[list(evaluator.ious.keys())[0]][0].shape)
        # print(evaluator.ious[list(evaluator.ious.keys())[0]].shape)
        # print(len(evaluator.ious))
        # print(evaluator.eval["fp_sum"])
        # print(evaluator.eval["fp_sum"].shape)
    #     print(evaluator.stats[1])
    #     p = evaluator.stats[0]
    #     r = evaluator.stats[8]
    #     f1=p*r/(p+r)
    #     li.append(p)
    #     if best_metric<f1:
    #         best_metric=f1
    #         best_thres = thres
    #         bb=best_thres, best_metric, p, r
    # print(bb)
    # print(li)
def main():
    path = 'trainval/inference_results/images_val_faster_rcnn_final_thres0.7_08_29_13/result.json'
    path = 'trainval/inference_results/images_val_cascade_rcnn_aug_final_thres0.7_08_29_13/result.json'
    gt_path = "trainval/annotations/bbox-annotations_val.json"
    run(path, gt_path)

 
if __name__ == "__main__":
    main()