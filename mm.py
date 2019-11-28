from mm_utils import *
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import random

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
    'learning_rate', 0.1, 'Learning rate.'
)

flags.DEFINE_integer(
    'num_generate_events', 100, 'Number of events to generate.'
)

def devide_single_sequence(seq):

    r = len(seq) % FLAGS.interval
    if r != 0:
        seq = seq[:-r]

    assert len(seq) % FLAGS.interval == 0

    input = np.array([seq[i:i + FLAGS.interval] for i in range(len(seq) - FLAGS.interval + 1)])[:-1]

    notes = seq[:,0][FLAGS.interval:]
    velocity = seq[:,1][FLAGS.interval:]-256
    time = seq[:,2][FLAGS.interval:]-(256+len(VELOCITY))

    return input, notes, velocity, time

def build_input_feature(sequences):
    input_feature = []
    notes = []
    velocity = []
    time = []
    for seq in sequences:
        input, note, v, t = devide_single_sequence(seq)
        input_feature.extend(input)

        notes.extend(note)
        velocity.extend(v)
        time.extend(t)

    input_feature = np.array(input_feature)
    notes = np.array(notes)
    velocity = np.array(velocity)
    time = np.array(time)

    return input_feature, notes, velocity, time


def create_model():
    inputs = tf.keras.Input(shape=(FLAGS.interval,3))

    embeddings =layers.Embedding(SEQUENCE_LENGTH, FLAGS.embedding_size, input_length=FLAGS.interval)(inputs)
    print(embeddings.shape)
    reshape = layers.Reshape((FLAGS.interval, 3 * FLAGS.embedding_size))(embeddings)
    lstm = layers.LSTM(FLAGS.num_cells, return_sequences=True)(reshape)

    for i in range(FLAGS.num_lstm_layers-2):
        dropout = layers.Dropout(0.2)(lstm)
        lstm = layers.LSTM(FLAGS.num_cells, return_sequences=True)(dropout)

    dropout = layers.Dropout(0.2)(lstm)
    lstm = layers.LSTM(FLAGS.num_cells, return_sequences=True)(dropout)
    notes = layers.Softmax(name='notes')(layers.Dense(256)(lstm))
    velocity = layers.Softmax(name='velocity')(layers.Dense(len(VELOCITY))(lstm))
    time = layers.Softmax(name='time')(layers.Dense(101)(lstm))

    model = tf.keras.Model(inputs=inputs, outputs=[notes, velocity, time])
    model.summary()


    return model

def main():

    tf.logging.set_verbosity = True
    eventSequence = convert_files_to_eventSequence(FLAGS.data_dir)
    input_feature, notes, velocity, time = build_input_feature(eventSequence)

    if FLAGS.num_lstm_layers < 2:
        sys.exit("Number of LSTM layers should at least be two.")

    model = create_model()
    opt = optimizers.SGD(lr=FLAGS.learning_rate)
    loss_weights = {'notes': 1, 'velocity': 0.0, 'time': 0.0}

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(input_feature, {'notes': notes, 'velocity': velocity, 'time': time}, batch_size=FLAGS.training_batch_size, epochs=FLAGS.num_epochs)

    init = np.array([[random.randrange(0, 256),
                     random.randrange(256, 256+len(VELOCITY)),
                     random.randrange(256+len(VELOCITY), SEQUENCE_LENGTH)] for i in range(FLAGS.interval)])

    generated_seq = []
    for i in range(FLAGS.num_generate_events):
        init_temp = np.array([init])
        output = (model.predict(init_temp, batch_size=1))
        note, v, t = map(np.argmax, output)
        generated_seq.append([note, v, t])

        v += 256
        t += 256+len(VELOCITY)
        generated_event = [note, v, t]
        init = np.append(init[1:], [generated_event], axis=0)

    print(generated_seq)


if __name__ == '__main__':
    main()


