# -*- coding: UTF-8 -*-
import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import ResNet50, InceptionResNetV2
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense


def resnet_50(weights="imagenet",
              input_shape=(224, 224, 3),
              classes=100,
              learning_rate=0.00005,
              frozen=False):
    model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)

    if frozen:
        model.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, name='output_1', activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)

    model.compile(optimizer=tf.compat.v1.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def inception_resnet_v2(weights="imagenet",
                        input_shape=(224, 224, 3),
                        classes=100,
                        learning_rate=0.00005,
                        frozen=False):
    model = InceptionResNetV2(include_top=False, weights=weights, input_shape=input_shape)

    if frozen:
        model.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, name='output_1', activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)

    model.compile(optimizer=tf.compat.v1.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
