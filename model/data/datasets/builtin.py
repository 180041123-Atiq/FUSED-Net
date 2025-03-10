import os

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import json
from detectron2.data.datasets import register_coco_instances


_PREDEFINED = [
  ("bdtsd_1shot","BDTSD/train_img","BDTSD/split/seedfewx1/full_box_1shot_trainval.json"),
  ("bdtsd_3shot","BDTSD/train_img","BDTSD/split/seedfewx1/full_box_3shot_trainval.json"),
  ("bdtsd_5shot","BDTSD/train_img","BDTSD/split/seedfewx1/full_box_5shot_trainval.json"),
  ("bdtsd_10shot","BDTSD/train_img","BDTSD/split/seedfewx1/full_box_10shot_trainval.json"),
  ("bdtsd_val","BDTSD/val","BDTSD/bdtsd_val.json")
]  

def register_data(root):
    for name, image_dir, json_file in _PREDEFINED:
        with open(os.path.join(root, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = [i["name"] for i in data["categories"]]
        register_coco_instances(name, {}, os.path.join(root, json_file), os.path.join(root, image_dir))
        MetadataCatalog.get(name).set(thing_classes=classes)

# Register them all under "./datasets"
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
_root = './datasets'
register_data(_root)
