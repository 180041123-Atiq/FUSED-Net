import sys
import cv2
import os
import xml.etree.ElementTree as ET

def do_visualization(inputs,outputs):
    # print("Inside inference_on_dataset")
    # print("printing inputs")
    img_file_path = inputs[0]['file_name']
    # print(img_file_path)
    ann_file_path = '.'+img_file_path.split('.')[1]+'.xml'

    img = cv2.imread(img_file_path)

    tree = ET.parse(ann_file_path)
    root = tree.getroot()

    cls_dict = {}

    for idx,line in enumerate(open('./datasets/BDTSD/classNames.txt','r').readlines()):
        # print(line)
        cls_dict[line.split('\n')[0]] = idx
        # print(cls_lst)
    
    for obj in root.findall('./object'):
        # print(obj.find('name').text)
        ts = ''
        for sr in obj.find('name').text.split(' '):
            ts += sr
        if(ts.find('/') >= 0):
            ts = ts.split('/')[0] + ts.split('/')[1]

        # print(cls_dict[ts])
        # print(int(obj.find('./bndbox/xmin').text))
        # print(int(obj.find('./bndbox/ymin').text))
        # print(int(obj.find('./bndbox/ymax').text))
        # print(int(obj.find('./bndbox/ymin').text))

        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (int(obj.find('./bndbox/xmin').text)+5, int(obj.find('./bndbox/ymin').text)-5) 
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        img = cv2.putText(img, str(cls_dict[ts]), org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

        start_point = (int(obj.find('./bndbox/xmin').text),int(obj.find('./bndbox/ymin').text))
        end_point = (int(obj.find('./bndbox/xmax').text),int(obj.find('./bndbox/ymax').text))
        color = (255, 0, 0)
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness) 

    pbs = [box.cpu().numpy() for box in outputs[0]['instances'].get('pred_boxes').__iter__()]
    pcs = outputs[0]['instances'].get('pred_classes').cpu().numpy()
    pss = outputs[0]['instances'].get('scores').cpu().numpy()

    for idx in range(len(pbs)):
        bbox = pbs[idx]
        scr = pss[idx]
        cn= pcs[idx]

        if scr >= 0.5:
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (int(bbox[2])+5, int(bbox[1])-5) 
            fontScale = 1
            color = (0, 255, 0) 
            thickness = 2
            img = cv2.putText(img, str(cn), org, font,  
                            fontScale, color, thickness, cv2.LINE_AA) 

            start_point = (int(bbox[0]),int(bbox[1]))
            end_point = (int(bbox[2]),int(bbox[3]))
            color = (0, 255, 0)
            thickness = 2
            img = cv2.rectangle(img, start_point, end_point, color, thickness) 

    outFilename = os.path.join('./visualInspection',img_file_path.split('/')[-1])

    cv2.imwrite(outFilename, img)

    # print("printing outputs")
    # print(outputs)
    # sys.exit(-1)
