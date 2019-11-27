from mm_utils import *
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers

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
    "embedding_size", 128, "Embedding size."
)

flags.DEFINE_integer(
    "num_cells", 64, "The number of cells in one single LSTM layer."
)

flags.DEFINE_integer(
    'num_lstm_layers', 3, 'Number of LSTM layers.'
)

flags.DEFINE_integer(
    'num_epochs', 3, 'The epochs in the training'
)

flags.DEFINE_integer(
    'training_batch_size', 256, 'The batch size in training'
)

flags.DEFINE_float(
    'learning_rate', 0.01, 'Learning rate.'
)


PADDING_ID = 266


def devide_single_sequence(seq):
    seq = np.concatenate((np.array([PADDING_ID]*FLAGS.interval), seq))

    r = len(seq) % FLAGS.interval
    if r != 0:
        seq = np.concatenate((seq, np.array([PADDING_ID]*(FLAGS.interval-r))))

    input = np.array([seq[i:i + FLAGS.interval] for i in range(len(seq) - FLAGS.interval + 1)])[:-1]
    output = seq[FLAGS.interval:]

    return input, output

def build_input_feature(sequences):
    input_feature = []
    labels = []
    for seq in sequences:
        input, output = devide_single_sequence(seq)
        input_feature.extend(input)
        labels.extend(output)

    input_feature = np.array(input_feature)
    labels = np.array(labels)

    return input_feature, labels

def main():
    tf.logging.set_verbosity = True

    eventSequence = convert_files_to_eventSequence(FLAGS.data_dir)
    input_feature, labels = build_input_feature(eventSequence)

    if FLAGS.num_lstm_layers < 2:
        sys.exit("Number of LSTM layers should at least be two.")

    model = tf.keras.Sequential()
    model.add(layers.Embedding(SEQUENCE_LENGTH, FLAGS.embedding_size, input_length=FLAGS.interval))
    model.add(layers.LSTM(FLAGS.num_cells, return_sequences=True, input_shape=[FLAGS.interval, FLAGS.embedding_size]))
    for i in range(FLAGS.num_lstm_layers-1):
        model.add(layers.LSTM(FLAGS.num_cells, return_sequences=True))

    model.add(layers.LSTM(SEQUENCE_LENGTH))
    model.add(layers.Softmax())

    model.summary()

    opt = optimizers.SGD(lr=FLAGS.learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(input_feature, labels, batch_size=FLAGS.training_batch_size, epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    main()


