import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2 import model_zoo

import model.data.datasets.builtin
from model.engine.trainer import SIStrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.cpu : cfg.MODEL.DEVICE = 'cpu'
    else : cfg.MODEL.DEVICE = 'cuda:0'

    # cfg.DATASETS.TRAIN = ("Merge",)
    # cfg.DATASETS.TEST = ('BDTSD_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = './outputBDTSDD'

    # cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")

    if args.mtsdd :
        print("LOADING MTSDD BASE WEIGHT")
        cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth") 
    else :
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") 
    
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 288

    # We have a single GPU environment with low VRAM
    # That's why we need to lower the learning rate
    # with lower batch size
    cfg.SOLVER.IMS_PER_BATCH = (2 * args.num_gpus)
    cfg.SOLVER.BASE_LR = (cfg.SOLVER.BASE_LR * cfg.SOLVER.IMS_PER_BATCH)/16

    # cfg.SOLVER.MAX_ITER = 110000
    # cfg.SOLVER.MAX_ITER = 1
    # cfg.SOLVER.STEPS = [70000,90000]    
    # cfg.SOLVER.STEPS = []    
    # cfg.SOLVER.CHECKPOINT_PERIOD = 20000

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    Trainer = SIStrainer
  

    if args.eval_only:
   
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--cpu", action="store_true", help="for using cpu instead of gpu")
    parser.add_argument("--mtsdd", action="store_true", help="for loading finetune weight")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )