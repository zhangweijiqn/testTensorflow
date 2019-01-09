# coding=utf-8
import tensorflow as tf
import os
import csv


'''
func: decode_libsvm
sep: seprator
start: start position of feature
'''


def decode_libsvm(line, sep=",", start=3):
    columns = tf.string_split([line], sep)
    labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
    splits = tf.string_split(columns.values[start:], ':')
    id_vals = tf.reshape(splits.values, splits.dense_shape)
    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(10000)  # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels

def print_dataset(dataset):
    iterator = dataset.make_one_shot_iterator()  # one shot iterator
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")



# 要保存后csv格式的文件名，生产环境建议使用tfRecord
filenames = ["data.csv"]
features, labels = input_fn(filenames)

