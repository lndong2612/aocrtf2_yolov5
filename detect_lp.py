import cv2
import torch

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/yolo/LP_detector.pt', force_reload=True, source='local')

def detect(img_path):
    classified = []
    img = cv2.imread(img_path)
    plates = yolo_LP_detect(img, size=640)

    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    count = 0

    if len(list_plates) == 0:
        lp = 'No detect'
        cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        list_read_plates.add(lp)
    else:
        for plate in list_plates:
            if float(plate[4]) >= 0.5:
                crop_img = img[int(plate[1]):int(plate[3]), int(plate[0]):int(plate[2])]
                cv2.imwrite("./results/crop_{}.jpg".format(count), crop_img)
                
                info = {
                    'xmin' : int(plate[0]),
                    'ymin' : int(plate[1]),
                    'xmax' : int(plate[2]),
                    'ymax' : int(plate[3]),
                    'path' : './results/crop_{}.jpg'.format(count)
                }
                classified.append(info)
                count += 1
            else:
                pass
    
    return classified