# -*- coding: UTF-8 -*-
import os
import platform
import sys
import time
import tensorflow as tf

sys.path.append(os.path.split(os.path.abspath(__file__))[0].split('python_package')[0])

from python_package.nets import inception_resnet_v2
from python_package.utils import DatasetHandler, TrainingCallback, save_model, convert_timestamp_to_timestr

if __name__ == '__main__':
    session = tf.compat.v1.Session()

    print('program start execute at {}'.format(convert_timestamp_to_timestr(timestamp=time.time())))

    dir_path = os.path.split(os.path.abspath(__file__))[0]
    class_index_path = os.path.join(dir_path, 'class_index.txt')
    model_weights_path = os.path.join(dir_path, 'saved_model/inception-resnet-v2-tf.weights.h5')
    input_shape = (224, 224, 3)

    if platform.system() != 'Windows':
        image_path = os.environ['IMAGE_TRAIN_INPUT_PATH']
        model_path = os.path.join(os.environ['MODEL_INFERENCE_PATH'], 'SavedModel')
    else:
        image_path = 'd:/data/garbage/garbage-official'
        model_path = os.path.join(dir_path, 'saved_model/inception-resnet-v2-tf')

    print('class_index_path : {}'.format(class_index_path))
    print('model_weights_path : {}'.format(model_weights_path))
    print('image_path : {}'.format(image_path))
    print('model_path : {}'.format(model_path))

    dataset_handler = DatasetHandler(root_dir_path=image_path,
                                     class_index_path=class_index_path,
                                     input_shape=input_shape,
                                     split=[0.8, 0.2],
                                     category=['train_1', 'validate'],
                                     repeat=1,
                                     shuffle=True,
                                     batch_size=32,
                                     session=session).get_dataset()

    train_data_x, train_data_y = dataset_handler[0][0], dataset_handler[0][1]
    validate_data_x, validate_data_y = dataset_handler[1][0], dataset_handler[1][1]

    model = inception_resnet_v2(input_shape=input_shape, weights=None, learning_rate=0.00005, frozen=False)
    model.load_weights(model_weights_path)
    model.summary()

    print('program start training at {}'.format(convert_timestamp_to_timestr(timestamp=time.time())))

    model.fit(x=train_data_x,
              y=train_data_y,
              epochs=60,
              batch_size=32,
              initial_epoch=3,
              verbose=2,
              shuffle=True,
              validation_data=(validate_data_x, validate_data_y),
              callbacks=[TrainingCallback(acc=0.90, val_acc=0.77, training_time=7200)])

    if platform.system() != 'Windows':
        save_model(model=model, model_path=model_path)
