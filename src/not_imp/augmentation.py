import albumentations as A
import cv2
aug_seq1 = A.OneOf([
    # A.Rotate(limit=(-90, 90), p=1.0),
    # A.Flip(p=1.0),
    A.HorizontalFlip(p=1.0),
    # A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.3, 0.3), 
    #                     shift_limit=(-0.05, 0.05), interpolation=3, 
    #                     border_mode=3, value=(0, 0, 0), mask_value=None),
    # A.CropNonEmptyMaskIfExists(
    #     height=1000, width=750, 
    #     p=1),
    # A.CropNonEmptyMaskIfExists(
    #     height=750, width=1000, 
    #     p=1),
    # A.RandomSizedCrop(
    #     height=750, width=1000, 
    #     erosion_rate=0, interpolation=3, always_apply=False, 
    #     p=1),
    # A.RandomSizedCrop(
    #     min_max_height=(700, 1024), 
    #     height = 1024, 
    #     width = 1024, 
    #     w2h_ratio=0.75, 
    #     interpolation=3, 
    #     always_apply=False, 
    #     p=1
    # ),
    # A.RandomSizedCrop(
    #     min_max_height=(700, 1024), 
    #     height = 1024, 
    #     width = 1024, 
    #     w2h_ratio=1.33, 
    #     interpolation=3, 
    #     always_apply=False, 
    #     p=1
    # ),
], p=0.5)
aug_seq2 = A.OneOf([
    # A.ChannelDropout(always_apply=False, p=1.0, channel_drop_range=(1, 1), fill_value=0),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                b_shift_limit=15, p=1.0),
    A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(
        -0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True)
], p=0.5)
# aug_seq3 = A.OneOf([
#     A.GaussNoise(always_apply=False, p=1.0, var_limit=(10, 50)),
#     A.ISONoise(always_apply=False, p=1.0, intensity=(
#         0.1, 0.5), color_shift=(0.01, 0.05)),
#     A.MultiplicativeNoise(always_apply=False, p=1.0, multiplier=(
#         0.9, 1.1), per_channel=True, elementwise=True),
# ], p=0.5)
# aug_seq4 = A.OneOf([
#     A.Equalize(always_apply=False, p=1.0,
#                 mode='pil', by_channels=True),
#     A.InvertImg(always_apply=False, p=1.0),
#     A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 7)),
#     # A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.3, 0.3), 
#     #                     shift_limit=(-0.05, 0.05), interpolation=0, 
#     #                     border_mode=0, value=(0, 0, 0), mask_value=None),
#     A.RandomFog(always_apply=False, p=1.0, fog_coef_lower=0.01,
#                 fog_coef_upper=0.2, alpha_coef=0.1)
# ], p=0.5)
# aug_seq5 = A.OneOf([
#     A.Compose([
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15,
#                 b_shift_limit=15, p=0.2),
#         A.RandomBrightnessContrast(always_apply=False, p=0.2, brightness_limit=(
#             -0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
#         A.MultiplicativeNoise(always_apply=False, p=0.2, multiplier=(
#             0.9, 1.1), per_channel=True, elementwise=True),
#     ]),
#     A.InvertImg(always_apply=False, p=0.4),
# ], p=0.5)
# aug_seq6 = A.OneOf([
    
#     # A.RGBShift(r_shift_limit=15, g_shift_limit=15,
#     A.GaussNoise(always_apply=False, p=0.8, var_limit=(10, 50)),
#     #         b_shift_limit=15, p=0.2),
#     A.RandomBrightnessContrast(always_apply=False, p=0.8, brightness_limit=(
#         -0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
#     A.MultiplicativeNoise(always_apply=False, p=0.8, multiplier=(
#         0.9, 1.1), per_channel=False, elementwise=True),
#     A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), always_apply=False, p=0.8),
    
    
# ], p=0.8)
transform = A.Compose([
    aug_seq1,
    aug_seq2
], bbox_params=A.BboxParams(format='coco',min_area=4, min_visibility=0.1, label_fields=['class_labels']))

A.save(transform, "src/aug1.json")
# image = cv2.imread("trainval/images/image_000000001.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
# transformed_image = transformed['image']
# transformed_bboxes = transformed['bboxes']
# transformed_class_labels = transformed['class_labels']