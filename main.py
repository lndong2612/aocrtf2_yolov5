import os
import re
import cv2
import common_flags

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import flags

from detect_lp import detect
from predict_lp import predict

FLAGS = flags.FLAGS
common_flags.define()

tf.disable_eager_execution()
tf.compat.v1.reset_default_graph()
image_path = './test/full_image/3.jpg'
classified = detect(image_path)


print("Predicted strings:")


print('[INFO] Result...')
fontScale = 0.5
image_full = cv2.imread(image_path)
image_h, image_w, _ = image_full.shape
bbox_thick = int(0.6 * (image_h + image_w) / 600)
bbox_color = (0, 255, 0)
for info in classified:
    image_path_pattern = info['path']
    predictions = predict(FLAGS.checkpoint, 
                        FLAGS.batch_size, 
                        FLAGS.dataset_name,
                        image_path_pattern)
    lp_number = predictions[-1]
    lp_number_process = re.sub(r"[^a-zA-Z0-9.\-\s]", "", lp_number)
    t_size = cv2.getTextSize(lp_number_process, 0, fontScale, thickness=bbox_thick // 2)[0]
    c3 = (info['xmin'] + t_size[0], info['ymin'] - t_size[1] - 7)
    cv2.rectangle(image_full, (info['xmin'], info['ymin']), (info['xmax'], info['ymax']), color = bbox_color, thickness = 2)
    cv2.rectangle(image_full, (info['xmin'], info['ymin']), c3, bbox_color, -1) #filled
    cv2.putText(image_full, lp_number_process, (info['xmin'], info['ymin']), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), bbox_thick // 2, lineType=cv2.LINE_AA)

cv2.imshow('image_full', image_full)
cv2.waitKey()
cv2.destroyAllWindows()