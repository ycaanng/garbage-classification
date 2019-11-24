# -*- coding: UTF-8 -*-
import os
import random
import shutil
import sys
import time
import glob
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.python.ops.image_ops_impl import ResizeMethodV1

seed = 16


# tf.compat.v1.random.set_random_seed(seed=seed)
# np.random.seed(seed=seed)
# random.seed(seed)


class TrainingCallback(tf.compat.v1.keras.callbacks.Callback):
    def __init__(self,
                 acc=0.96,
                 val_acc=0.65,
                 training_time=9000,
                 start_time=time.time()):
        super(TrainingCallback, self).__init__()
        self.acc = acc
        self.val_acc = val_acc
        self.training_time = training_time
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if logs.get('acc') >= self.acc and logs.get('val_acc') >= self.val_acc:
            print('finally training logs : {}'.format(logs))
            self.model.stop_training = True
        if time.time() - self.start_time >= self.training_time:
            self.model.stop_training = True

        print('start_time : {}, current_time : {}'.format(
            convert_timestamp_to_timestr(self.start_time),
            convert_timestamp_to_timestr(time.time())))


class DatasetHandler(object):
    def __init__(self,
                 root_dir_path,
                 class_index_path,
                 input_shape,
                 split,
                 category,
                 repeat,
                 shuffle,
                 batch_size,
                 session):
        self.root_dir_path = root_dir_path
        self.class_index_path = class_index_path
        self.input_shape = input_shape
        self.split = split
        self.category = category
        self.repeat = repeat
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.session = session

    __python_version = int(sys.version_info.major)
    __autotune = tf.compat.v1.data.experimental.AUTOTUNE

    def __get_class_index_dicts(self):
        class_index_dicts = {}

        if self.__python_version == 2:
            with open(self.class_index_path) as file:
                for line in file:
                    split = line.strip().split(' ')
                    class_index_dicts[split[0].decode('utf-8')] = int(split[1])

        if self.__python_version == 3:
            with open(self.class_index_path, encoding='utf-8') as file:
                for line in file:
                    split = line.split(' ')
                    class_index_dicts[split[0]] = int(split[1])

        return class_index_dicts

    def __get_file_class(self, file_path):
        if self.__python_version == 2:
            return file_path.split('/')[-2].decode('utf-8')
        else:
            return file_path.split('/')[-2]

    # reference url : https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/preprocessing_factory.py#L48
    def __process_train_image(self, path, label):
        image_decoded = tf.compat.v1.cast(tf.compat.v1.image.decode_jpeg(
            tf.compat.v1.io.read_file(path), channels=self.input_shape[2]), dtype=tf.compat.v1.float16)
        resized_height, resized_width = self.input_shape[0], self.input_shape[1]
        image_resized = tf.compat.v1.image.resize(image_decoded, [resized_height, resized_width], method=ResizeMethodV1.BILINEAR)
        return image_resized, label

    def __process_eval_image(self, path, label):
        image_decoded = tf.compat.v1.cast(tf.compat.v1.image.decode_jpeg(
            tf.compat.v1.io.read_file(path), channels=self.input_shape[2]), dtype=tf.compat.v1.float16)
        resized_height, resized_width = self.input_shape[0], self.input_shape[1]
        image_resized = tf.compat.v1.image.resize(image_decoded, [resized_height, resized_width], method=ResizeMethodV1.BILINEAR)
        return image_resized, label

    def __split_dataset_v1(self):
        class_index_dicts = self.__get_class_index_dicts()

        file_paths_array = [[] for i in range(len(self.split))]
        file_labels_array = [[] for i in range(len(self.split))]

        for sub_dir_path in glob.glob(os.path.join(self.root_dir_path, '*')):
            paths = [path.replace('\\', '/') for path in glob.glob(os.path.join(sub_dir_path, '*'))]
            if self.shuffle:
                random.shuffle(paths)
            else:
                paths.sort()

            per_group_numbers = [0] + [int(round(len(paths) * sum(self.split[:i + 1]))) for i in range(len(self.split))]
            paths_array = [paths[per_group_numbers[i]:per_group_numbers[i + 1]] for i in range(len(per_group_numbers) - 1)]
            labels_array = [[class_index_dicts[self.__get_file_class(path)] for path in paths] for paths in paths_array]

            [file_paths_array[i].extend(paths_array[i]) for i in range(len(self.split))]
            [file_labels_array[i].extend(labels_array[i]) for i in range(len(self.split))]

        return file_paths_array, file_labels_array

    def __split_dataset_v2(self):
        class_index_dicts = self.__get_class_index_dicts()

        dataset_pd = []

        for sub_dir_path in glob.glob(os.path.join(self.root_dir_path, '*')):
            paths = [path.replace('\\', '/') for path in glob.glob(os.path.join(sub_dir_path, '*'))]
            if self.shuffle:
                random.shuffle(paths)
            else:
                paths.sort()

            categroy = []
            per_group_numbers = [int(round(len(paths) * self.split[i])) for i in range(len(self.split))]
            [categroy.extend([self.category[i]] * per_group_numbers[i]) for i in range(len(per_group_numbers))]

            total_size = sum(per_group_numbers)
            if total_size > len(paths):
                total_size = len(paths)

            data_pd = pd.DataFrame()
            data_pd['path'] = paths[:total_size]
            data_pd['category'] = categroy[:total_size]
            data_pd['label'] = data_pd['path'].apply(lambda x: class_index_dicts[self.__get_file_class(x)])

            dataset_pd.append(data_pd)

        return pd.concat(dataset_pd)

    def __get_dataset_v1(self):
        file_paths_arrays, file_labels_arrays = self.__split_dataset_v1()
        dataset_tf = []

        for i in range(len(self.category)):
            paths = file_paths_arrays[i]
            labels = file_labels_arrays[i]
            total_size = len(paths)

            data_tf = tf.compat.v1.data.Dataset.from_tensor_slices((paths, labels))

            if self.category[i].startswith('train'):
                total_size = total_size * self.repeat
                data_tf = data_tf.repeat(self.repeat)
                data_tf = data_tf.map(self.__process_train_image, num_parallel_calls=self.__autotune)
                data_tf = data_tf.shuffle(total_size)
            else:
                data_tf = data_tf.map(self.__process_eval_image, num_parallel_calls=self.__autotune)

            data_tf = data_tf.batch(batch_size=self.batch_size)
            data_tf = data_tf.prefetch(buffer_size=self.__autotune)
            dataset_tf.append(data_tf)

            print('{} data size : {}'.format(self.category[i], total_size))

        return dataset_tf

    def __get_dataset_v2(self):
        dataset_pd = self.__split_dataset_v2()
        dataset_tf = []

        for c in self.category:
            paths = dataset_pd[dataset_pd['category'] == c]['path']
            labels = dataset_pd[dataset_pd['category'] == c]['label']
            total_size = len(paths)

            data_tf = tf.data.Dataset.from_tensor_slices((paths, labels))

            if c.startswith('train'):
                total_size = total_size * self.repeat
                data_tf = data_tf.repeat(self.repeat)
                data_tf = data_tf.map(self.__process_train_image, num_parallel_calls=self.__autotune)
                data_tf = data_tf.shuffle(total_size)
            else:
                data_tf = data_tf.map(self.__process_eval_image, num_parallel_calls=self.__autotune)

            data_tf = self.session.run(tf.compat.v1.data.make_one_shot_iterator(data_tf.batch(batch_size=total_size)).get_next())
            dataset_tf.append([np.array(data_tf[0]), np.array(data_tf[1])])

            print('{} data size : {}'.format(c, total_size))

        return dataset_tf

    def get_dataset(self):
        return self.__get_dataset_v2()


def convert_timestamp_to_timestr(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


def save_model(model, model_path):
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tf.compat.v1.keras.experimental.export_saved_model(model, model_path)


def save_model_weights(model, weights_path):
    if os.path.exists(weights_path):
        os.remove(weights_path)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        os.rmdir(weights_path)

    model.save_weights(weights_path)
