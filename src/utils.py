
# import copy
import itertools
import logging
import os
import weakref

# import albumentations as A
# import cv2
import numpy as np
# import torch
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
from detectron2.data import (DatasetMapper, build_detection_test_loader,
                             build_detection_train_loader)
# from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import (DefaultPredictor, DefaultTrainer,
                               create_ddp_model, hooks)
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import create_small_table, setup_logger
from jaitool.evaluation import LossEvalHook
from tabulate import tabulate


class CustomCOCOEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        super(CustomCOCOEvaluator, self).__init__(dataset_name,
                                                  tasks,
                                                  distributed,
                                                  output_dir,
                                                  use_fast_impl=use_fast_impl,
                                                  kpt_oks_sigmas=kpt_oks_sigmas)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR", "AR50", "AR75", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR", "AR50", "AR75", "ARs", "ARm", "ARl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(
                iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info(
                "Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        # recalls ---------------------------------
        recalls = coco_eval.eval["recall"]
        # recall has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == recalls.shape[1]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")
            results_per_category.append(("{}".format(name), float(ar * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AR: \n".format(iou_type) + table)

        results.update({"AR-" + name: ar for name, ar in results_per_category})
        return results


class CustomTrainer(DefaultTrainer, TrainerBase):
    def __init__(self, cfg, aug_settings_file_path=None):
        """
        Args:
            cfg (CfgNode):
        """
        augment = None
        if aug_settings_file_path is not None:
            # augment = A.load(aug_settings_file_path)
            # AugmentationList
            augment = [
                T.RandomBrightness(0.9, 1.1),
                T.RandomFlip(prob=0.5),
                # T.RandomCrop("absolute", (640, 640))
            ]  # type: T.Augmentation

        TrainerBase.__init__(self)

        logger = logging.getLogger("detectron2")
        # setup_logger is not called for d2
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(
            cfg=cfg,
            augment=augment,)
        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, eval_tasks=None, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CustomCOCOEvaluator(dataset_name, eval_tasks, True, output_folder, use_fast_impl=False)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

    @classmethod
    def build_train_loader(
        cls, cfg,
        augment=None,
    ):
        if augment is not None:
            # aug_loader = AugmentedLoader(cfg=cfg, augment=augment)
            aug_loader = DatasetMapper(
                is_train=True,
                augmentations=augment,
                image_format=cfg.INPUT.FORMAT,
                use_instance_mask = cfg.MODEL.MASK_ON,
                use_keypoint = False,
                instance_mask_format=cfg.INPUT.MASK_FORMAT,
                keypoint_hflip_indices=cfg.MODEL.KEYPOINT_ON,
                precomputed_proposal_topk=cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN,
                recompute_boxes=cfg.MODEL.MASK_ON,)
            return build_detection_train_loader(cfg, mapper=aug_loader)
        else:
            return build_detection_train_loader(cfg)


# class AugmentedLoader(DatasetMapper):
#     def __init__(self, cfg, augment=None):
#         super().__init__(cfg)
#         self.cfg = cfg
#         self.augment = augment

#     def __call__(self, dataset_dict) -> dict:


#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         # USER: Write your own image loading if it's not from a file
#         image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
#         utils.check_image_size(dataset_dict, image)

#         # USER: Remove if you don't do semantic/panoptic segmentation.
#         if "sem_seg_file_name" in dataset_dict:
#             sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
#         else:
#             sem_seg_gt = None

#         # aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
#         # transforms = self.augmentations(aug_input)
#         # image, sem_seg_gt = aug_input.image, aug_input.sem_seg

#         # from pyjeasy.image_utils import show_image
#         # from jaitool.draw import draw_bbox
#         bboxes = []
#         class_labels = []
#         # image2=image.copy()
#         for ann in dataset_dict["annotations"]:
#             bboxes.append(ann['bbox'])
#             class_labels.append(ann['category_id'])
#     #         b=ann['bbox']
#     #         image2= draw_bbox(
#     #         image2, [b[0], b[1], b[2]+b[0], b[3]+b[1]],
#     #         # color=None, font_face=0,thickness: int = 2, text: str = None, label_thickness: int = 0,
#     #         # label_color: list = None, show_bbox: bool = True, show_label: bool = True,
#     #         # label_orientation: str = 'top', text_size: int = None
#     # )
#         # print(dataset_dict["annotations"])
#         # print()

#         transforms = self.augment(image=image, bboxes=bboxes, class_labels=class_labels)
#         transformed_image = transforms['image']
#         transformed_bboxes = transforms['bboxes']
#         transformed_class_labels = transforms['class_labels']
#         for i, ann in enumerate(dataset_dict["annotations"]):
#             dataset_dict["annotations"][i]['bbox'] = transformed_bboxes[i]
#             dataset_dict["annotations"][i]['category_id'] = transformed_class_labels[i]

#             b=transformed_bboxes[i]
#     #         print(b)
#             print([b[0], b[1], b[2]+b[0], b[3]+b[1]])
#     #         transformed_image= draw_bbox(
#     #         transformed_image, [b[0], b[1], b[2]+b[0], b[3]+b[1]],
#     #         # color=None, font_face=0,thickness: int = 2, text: str = None, label_thickness: int = 0,
#     #         # label_color: list = None, show_bbox: bool = True, show_label: bool = True,
#     #         # label_orientation: str = 'top', text_size: int = None
#     # )
#     #     im = cv2.hconcat([image2, transformed_image])
#     #     show_image(im)
#         image_shape = transformed_image.shape[:2]  # h, w
#         # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#         # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#         # Therefore it's important to use torch.Tensor.
#         dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(transformed_image.transpose(2, 0, 1)))
#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

#         # USER: Remove if you don't use pre-computed proposals.
#         # Most users would not need this feature.
#         if self.proposal_topk is not None:
#             utils.transform_proposals(
#                 dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
#             )

#         if not self.is_train:
#             # USER: Modify this if you want to keep them for some reason.
#             dataset_dict.pop("annotations", None)
#             dataset_dict.pop("sem_seg_file_name", None)
#             return dataset_dict

#         if "annotations" in dataset_dict:
#             # USER: Modify this if you want to keep them for some reason.
#             for anno in dataset_dict["annotations"]:
#                 if not self.use_instance_mask:
#                     anno.pop("segmentation", None)
#                 if not self.use_keypoint:
#                     anno.pop("keypoints", None)

#             # USER: Implement additional transformations if you have other types of data
#             annos = [
#                 obj
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             # annos = [
#             #     utils.transform_instance_annotations(
#             #         obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
#             #     )
#             #     for obj in dataset_dict.pop("annotations")
#             #     if obj.get("iscrowd", 0) == 0
#             # ]
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=self.instance_mask_format
#             )

#             # After transforms such as cropping are applied, the bounding box may no longer
#             # tightly bound the object. As an example, imagine a triangle object
#             # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
#             # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
#             # the intersection of original bounding box and the cropping box.
#             if self.recompute_boxes:
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
#             dataset_dict["instances"] = utils.filter_empty_instances(instances)



#         # image = cv2.imread(dataset_dict["file_name"])
#         # bboxes = []
#         # class_labels = []
#         # for ann in dataset_dict["annotations"]:
#         #     bboxes.append(ann['bbox'])
#         #     class_labels.append(ann['category_id'])
#         # # print(dataset_dict["annotations"])
#         # # print()

#         # transformed = self.augment(image=image, bboxes=bboxes, class_labels=class_labels)
#         # transformed_image = transformed['image']
#         # transformed_bboxes = transformed['bboxes']
#         # transformed_class_labels = transformed['class_labels']
#         # for i, ann in enumerate(dataset_dict["annotations"]):
#         #     transformed_dataset_dict["annotations"][i]['bbox'] = transformed_bboxes[i]
#         #     transformed_dataset_dict["annotations"][i]['category_id'] = transformed_class_labels[i]
#         # num_kpts = 0
#         # annots = []
#         # for item in transformed_dataset_dict["annotations"]:
#         #     if 'keypoints' in item and num_kpts == 0:
#         #         del item['keypoints']
#         #     elif 'keypoints' in item:
#         #         item['keypoints'] = np.array(
#         #             item['keypoints']).reshape(-1, 3).tolist()
#         #     annots.append(item)
#         # transformed_dataset_dict["image"] = torch.as_tensor(
#         #     transformed_image.transpose(2, 0, 1).astype("float32"))
#         # instances = utils.annotations_to_instances(annots, transformed_image.shape[:2])
#         # dataset_dict["instances"] = utils.filter_empty_instances(
#         #     instances, by_box=True, by_mask=False)
#         # print(dataset_dict["annotations"])
#         # print()
#         # print()
#         # print()
#         return dataset_dict
