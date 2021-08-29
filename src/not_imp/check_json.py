
import json, shutil


def run(json_path, train_json_path, test_json_path):
    with open("/home/jitesh/jg/openimages-personcar-localize_and_classify/trainval/annotations/bbox-annotations_val.json", 'rt', encoding='UTF-8') as annotations:
        data = json.load(annotations)
        count = 0
    for image in data["images"]:
        file_name = image["file_name"]
        print(image["id"], file_name)
        shutil.copyfile(f'trainval/images/{file_name}', f'trainval/images_val/{file_name}')
        count +=1
    print(count)
if __name__ == "__main__":
    json_path = "/home/jitesh/jg/eagleview/trainval/annotations/bbox-annotations.json"
    json_path_split = json_path.split(".")
    train_json_path = ".".join(json_path_split[:-1]) + "_train.json"
    val_json_path = ".".join(json_path_split[:-1]) + "_val.json"
    print(train_json_path)
    run(json_path, train_json_path, val_json_path)

    import os
    l = os.listdir("/home/jitesh/jg/openimages-personcar-localize_and_classify/trainval/images_val")
    print(len(l))