import os
import shutil
import cv2
import common_flags

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import flags
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_eager_execution()
tf.compat.v1.reset_default_graph()

from detect_lp import detect
from predict_lp import predict, draw_bbox

FLAGS = flags.FLAGS
common_flags.define()

if os.path.exists('./results'):
    print('[INFO] Folder already exist ..')
    for file_name in os.listdir('./results'):
        os.remove(f'./results/{file_name}')
else:
    print('[INFO] Create folder ..')
    os.makedirs('./results')


image_path = './test/full_image/117.jpg'
classified = detect(image_path)

print("Predicted strings:")
results = predict(FLAGS.checkpoint, FLAGS.batch_size, FLAGS.dataset_name, classified)

print('[INFO] Result...')
print(results)
image_show = draw_bbox(image_path, results)

cv2.imshow('image_full', image_show)
cv2.waitKey()
cv2.destroyAllWindows()