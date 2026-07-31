import json
from ultralytics import YOLO
import torch
import time

if __name__ == '__main__':

    model = YOLO("yolov8s.pt")
    # model.info(verbose=True)
    # exit()
    
    with open("datasets/BDTSD/bdtsd_val.json", "r") as f:
        test_set_json = json.load(f)

    # print(type(test_set_json['images'][0]['file_name']))
    all_results = []

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    preprocess_ms = 0
    inf_ms = 0
    postprocess_ms = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    for ii in range(len(test_set_json['images'])):
        results = model(f"datasets/BDTSD/val/{test_set_json['images'][ii]['file_name']}")
        # print(results[0].speed)
        preprocess_ms += results[0].speed['preprocess']
        inf_ms += results[0].speed['inference']
        postprocess_ms += results[0].speed['postprocess']
        
    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    # peak_mem = torch.cuda.max_memory_allocated()
    print(f"Peak allocated : {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    print(f"Peak reserved  : {torch.cuda.max_memory_reserved()/1024**2:.2f} MB")
    print(f"Current alloc  : {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Current reserve: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    print('Per Sample Inference Time according to Ultralytics: ',(preprocess_ms+inf_ms+postprocess_ms)/len(test_set_json['images']))
    