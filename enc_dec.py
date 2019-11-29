import sys
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from mm_utils import *
import tensorflow as tf
import numpy as np
from keras_transformer import get_model, decode

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tif files (or other data files) for the task."
)

flags.DEFINE_string(
    "output_dir", 'results',
    "Path to output dir."
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

flags.DEFINE_integer(
    'num_generate_events', 100, 'Number of events to generate.'
)

flags.DEFINE_integer(
    "epoch_interval", 10, "Epoch interval length."
)

flags.DEFINE_bool(
    "overwritting", False, "Whether over write models"
)

flags.DEFINE_bool(
    "sum_embeddings", True, "Whether sum the embeddings."
)

def process_input_feature(input_feature):

    input_feature = input_feature.reshape((input_feature.shape[0], FLAGS.interval*3))

    encoder_input = input_feature[:-1]
    print(encoder_input.shape)
    decoder_input = input_feature[1:]

    return encoder_input, decoder_input

def create_transformer():

    model = get_model(token_num=SEQUENCE_LENGTH,
                      embed_dim=FLAGS.embedding_size,
                      encoder_num=3,
                      decoder_num=3,
                      head_num=8,
                      hidden_dim=FLAGS.num_cells,
                      attention_activation='relu',
                      feed_forward_activation='relu',
                      dropout_rate=0.05,
                      embed_weights=np.random.random((SEQUENCE_LENGTH, FLAGS.embedding_size))
                      )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
    )

    return model


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


def main():
    tf.logging.set_verbosity = True
    eventSequence = convert_files_to_eventSequence(FLAGS.data_dir)

    test_sequence = eventSequence[-1]
    if len(eventSequence) > 1: eventSequence = eventSequence[:-1]

    input_feature, notes, velocity, time = build_input_feature(eventSequence)

    encoder_input, decoder_input = process_input_feature(input_feature)

    model = create_transformer()
    model.fit(
        x=[encoder_input, decoder_input],
        y=np.reshape(decoder_input,(decoder_input.shape[0], decoder_input.shape[1], 1)),
        epochs=FLAGS.num_epochs,
        batch_size=FLAGS.training_batch_size
    )

    init = np.reshape(test_sequence[:FLAGS.interval], (FLAGS.interval*3))

    generated_seq = []
    for i in range(int(FLAGS.num_generate_events/FLAGS.interval)):
        print(init)
        result = model.predict(x=[init, init], batch_size=1)
        seq = np.reshape(np.argmax(result, axis=2), (FLAGS.interval, 3))

        print(seq)
        init = np.argmax(result, axis=2).flatten()

        generated_seq.extend(seq)

    pre = os.getcwd()
    os.chdir(FLAGS.output_dir)
    convert_eventSequence_to_midi(generated_seq, FLAGS.num_epochs)
    os.chdir(pre)




if __name__ == '__main__':
    main()

