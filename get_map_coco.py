import os
from utils.utils_map import get_coco_map

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

if __name__ == "__main__":

    classes_path = 'model_data/helmet_detection.txt'

    VOCdevkit_path = 'dataset'

    map_out_path = 'input'

    image_ids = open(os.path.join(VOCdevkit_path, "SCAUHDM2022/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    print("Get map.")
    get_coco_map(class_names=class_names, path=map_out_path)
    print("Get map done.")
