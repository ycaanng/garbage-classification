# -*- coding: UTF-8 -*-
import os
import time
import tensorflow as tf
import numpy as np

from python_package.nets import inception_resnet_v2
from python_package.utils import DatasetHandler, TrainingCallback, convert_timestamp_to_timestr, save_model, save_model_weights

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    print('program start execute at {}'.format(convert_timestamp_to_timestr(timestamp=time.time())))

    image_origin_path = 'd:/data/garbage/garbage-origin'
    image_official_path = 'd:/data/garbage/garbage-official'
    model_weights_path = './saved_model/inception-resnet-v2-tf.weights.h5'
    model_path = './saved_model/inception-resnet-v2-tf'

    input_shape = (224, 224, 3)
    class_index_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'class_index.txt')

    origin_dataset_handler = DatasetHandler(root_dir_path=image_origin_path,
                                            class_index_path=class_index_path,
                                            input_shape=input_shape,
                                            split=[0.8, 0.2],
                                            category=['train_1', 'validate'],
                                            repeat=1,
                                            shuffle=True,
                                            batch_size=32,
                                            session=session)

    origin_dataset = origin_dataset_handler.get_dataset()
    train_data_x, train_data_y = origin_dataset[0][0], origin_dataset[0][1]
    validate_data_x, validate_data_y = origin_dataset[1][0], origin_dataset[1][1]

    official_dataset_handler = DatasetHandler(root_dir_path=image_official_path,
                                              class_index_path=class_index_path,
                                              input_shape=input_shape,
                                              split=[1],
                                              category=['official'],
                                              repeat=1,
                                              shuffle=True,
                                              batch_size=32,
                                              session=session)
    official_dataset = official_dataset_handler.get_dataset()
    official_data_x, official_data_y = official_dataset[0][0], official_dataset[0][1]

    train_data_x = np.vstack((train_data_x, official_data_x))
    train_data_y = np.hstack((train_data_y, official_data_y))

    model = inception_resnet_v2(input_shape=input_shape, learning_rate=0.00006, frozen=False)
    # if os.path.exists(model_weights_path):
    #     model.load_weights(model_weights_path)
    model.summary()

    print('program start training at {}'.format(convert_timestamp_to_timestr(timestamp=time.time())))

    model.fit(x=train_data_x,
              y=train_data_y,
              batch_size=32,
              epochs=30,
              verbose=2,
              shuffle=True,
              validation_data=(validate_data_x, validate_data_y),
              callbacks=[TrainingCallback(acc=0.90, val_acc=0.81, training_time=7200)])

    print('evaluate model at official data')
    model.evaluate(x=official_data_x, y=official_data_y, verbose=2)

    save_model_weights(model=model, weights_path=model_weights_path)
    save_model(model=model, model_path=model_path)
