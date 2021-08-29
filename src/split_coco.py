
import json

import funcy
from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=False)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def run(json_path, train_json_path, test_json_path, split_ratio, annotations_required: bool = True):
    with open(json_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info')
        licenses = coco.get('licenses')
        images = coco.get('images')
        annotations = coco.get('annotations')
        categories = coco.get('categories')

        images_with_annotations = funcy.lmap(
            lambda a: int(a['image_id']), annotations)
        print("Before: ", len(images))
        if annotations_required:
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations, images)
        print("After: ", len(images))
        x, y = train_test_split(images, train_size=split_ratio)

        save_coco(train_json_path, info, licenses, x,
                  filter_annotations(annotations, x), categories)
        save_coco(test_json_path, info, licenses, y,
                  filter_annotations(annotations, y), categories)

        print(
            f'Splitted from {len(images)} entries in {json_path} \nto {len(x)} in {train_json_path} and {len(y)} in {test_json_path}')


if __name__ == "__main__":
    json_path = "/home/jitesh/jg/eagleview/trainval/annotations/bbox-annotations.json"
    json_path_split = json_path.split(".")
    train_json_path = ".".join(json_path_split[:-1]) + "_train.json"
    val_json_path = ".".join(json_path_split[:-1]) + "_val.json"
    print(train_json_path)
    run(json_path, train_json_path, val_json_path,
        split_ratio=0.95, annotations_required=False)
