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

flags.DEFINE_integer(
    'num_epochs', 3, 'The epochs in the training'
)

flags.DEFINE_integer(
    'training_batch_size', 256, 'The batch size in training'
)


flags.DEFINE_integer(
    'num_lstm_layers', 3, 'Number of LSTM layers.'
)

flags.DEFINE_integer(
    'num_generate_events', 1000, 'Number of events to generate.'
)

SEQUENCE_LENGTH = 128+128+len(VELOCITY)+101
PADDING = np.array([[1] * SEQUENCE_LENGTH])


def divide_sequences(sequences):
    input = []
    output = []

    for sequence in sequences:
        for i in range(FLAGS.interval):
            sequence = np.insert(sequence, 0, PADDING, axis=0)

        r = len(sequence) % FLAGS.interval
        if r != 0:
            for i in range(FLAGS.interval-r):
                sequence = np.append(sequence, PADDING, axis=0)

        print(sequence.shape)

        #intervals = np.array(zip(*(sequence[i:] for i in range(FLAGS.interval))))
        intervals = np.array([sequence[i:i+FLAGS.interval] for i in range(len(sequence)-FLAGS.interval+1)])[:-1]
        print(intervals.shape)
        output.extend(sequence[FLAGS.interval+1:])
        output.append(PADDING[0])

        input.extend(intervals)

    input = np.array(input)
    output = np.array(output)
    print(input.shape)
    print(output.shape)
    return input, output


def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.reshape((y_true-y_pred)**2, shape=[-1]))


def main():
    tf.logging.set_verbosity = True
    data_path = FLAGS.data_dir
    sequences = read_data(data_path)

    input, output = divide_sequences(sequences)

    if FLAGS.num_lstm_layers < 2:
        sys.exit("Number of LSTM layers should at least be one.")

    model = tf.keras.Sequential()
    model.add(layers.LSTM(FLAGS.num_cells, return_sequences=True, input_shape=[FLAGS.interval, SEQUENCE_LENGTH]))
    for i in range(FLAGS.num_lstm_layers-1):
        model.add(layers.LSTM(FLAGS.num_cells, return_sequences=True))

    model.add(layers.LSTM(SEQUENCE_LENGTH))
    #model.add(layers.Dense(SEQUENCE_LENGTH))
    model.add(layers.Softmax())

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(input, output, batch_size=FLAGS.training_batch_size, epochs=FLAGS.num_epochs)

    init = np.array(list(PADDING)*FLAGS.interval)
    print(init.shape)
    generated_sequence = []
    for i in range(FLAGS.num_generate_events):
        init_temp = np.array([init])
        generated_event = model.predict(init_temp, batch_size=1)
        init = np.append(init[1:], generated_event, axis=0)
        generated_sequence.append(np.argmax(generated_event[0]))

    print(generated_sequence)

   #convert_eventSequence_to_midi(generated_sequence)






if __name__ == '__main__':
    main()


