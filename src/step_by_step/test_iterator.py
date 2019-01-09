import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

# test iterator: 1, one-shot iterator; 2, iterator in eager mode

# one-shot
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))       #create Dataset
iterator = dataset.make_one_shot_iterator()     #one shot iterator
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

# eager mode
tfe.enable_eager_execution()                    #Eager mode
for one_element in tfe.Iterator(dataset):       #create Iterator directly
    print(one_element)