from jaitool.annotation.COCO import visualize_coco_ann # pip install jaitool

if __name__ == "__main__":
    img_dir = "trainval/images"
    json_path = "trainval/annotations/bbox-annotations.json"
    
    visualize_coco_ann(path=img_dir, json_path=json_path)
