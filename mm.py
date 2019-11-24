import tensorflow as tf
from mm_utils import *
from tensorflow.keras import layers

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tif files (or other data files) for the task."
)

flags.DEFINE_integer(
    "interval", 20, "The length of interval."
)

flags.DEFINE_integer(
    "num_cells", 64, "The number of cells in one single LSTM layer."
)

flag.DEFINE_integer(
    'num_epochs', 3, 'The epochs in the training'
)

flag.DEFINE_integer(
    'num_training_batch_size', 32, 'The batch size in training'
)

flag.DEFINE_integer(
    'num_eval_batch_size', 32, 'The batch size in predict (evaluation)'
)

PADDING = np.array([[0] * (128 + 128 + len(VELOCITY) + 101)])


def divide_sequences(sequences):
    input_feature = []

    for sequence in sequences:
        r = len(sequence) % FLAGS.interval
        if r != 0:
            for i in range(20-r):
                sequence = np.append(sequence, PADDING, axis=0)

        intervals = np.split(sequence, int(len(sequence)/20))

        input_feature.extend(intervals)

    return np.array(input_feature)


def main():
    tf.logging.set_verbosity = True
    data_path = FLAGS.data_dir
    sequences = read_data(data_path)

    input_feature = divide_sequences(sequences)
    print(len(input_feature))

    model = tf.keras.Sequential()
    model.add(layers.LSTM(FLAGS.num_cells, input_shape=FLAGS.interval))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(input_feature[:-1], input_feature[:1], )




if __name__ == '__main__':
    main()


