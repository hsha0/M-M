from mm_utils import *
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import random
import time
import datetime


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
    'learning_rate', 0.001, 'Learning rate.'
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
    "sum_embeddings", False, "Whether sum the embeddings."
)

flags.DEFINE_bool(
    "use_tpu", False, "Whether use tpu."
)

flags.DEFINE_string(
    "tpu_address", None, "TPU address."
)

def divide_single_sequence(seq):
    """
    Divide single sequence into small overlapping chunks with one-element shift.
    :param seq: list. message sequence.
    :return: input, three outputs.
    """
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
    """
    Build input feature based on a list of message sequences.
    :param sequences: list. A list of message sequences.
    :return: input, three outputs.
    """
    input_feature = []
    notes = []
    velocity = []
    time = []
    for seq in sequences:
        input, note, v, t = divide_single_sequence(seq)
        input_feature.extend(input)

        notes.extend(note)
        velocity.extend(v)
        time.extend(t)

    input_feature = np.array(input_feature)
    notes = np.array(notes)
    velocity = np.array(velocity)
    time = np.array(time)

    return input_feature, notes, velocity, time


def create_lstm_model():
    """
    Create LSTM model.
    :return: model.
    """
    inputs = tf.keras.Input(shape=(FLAGS.interval,3))

    embeddings =layers.Embedding(SEQUENCE_LENGTH, FLAGS.embedding_size, input_length=FLAGS.interval)(inputs)
    print(embeddings.shape)

    if FLAGS.sum_embeddings:
        reshape = tf.math.reduce_sum(embeddings, axis=2)
    else:
        reshape = layers.Reshape((FLAGS.interval, 3 * FLAGS.embedding_size))(embeddings)
    lstm = layers.LSTM(FLAGS.num_cells, return_sequences=True)(reshape)

    for i in range(FLAGS.num_lstm_layers-2):
        dropout = layers.Dropout(0.2)(lstm)
        lstm = layers.LSTM(FLAGS.num_cells, return_sequences=True)(dropout)

    dropout = layers.Dropout(0.2)(lstm)
    lstm = layers.LSTM(FLAGS.num_cells)(dropout)
    dropout = layers.Dropout(0.2)(lstm)
    notes = layers.Softmax(name='notes')(layers.Dense(256)(dropout))
    velocity = layers.Softmax(name='velocity')(layers.Dense(len(VELOCITY))(dropout))
    time = layers.Softmax(name='time')(layers.Dense(101)(dropout))

    model = tf.keras.Model(inputs=inputs, outputs=[notes, velocity, time])

    model.summary()

    return model


def merge_init(init, init_2):
    """
    Merge two message lists.
    :param init: note on message list.
    :param init_2: note off message list.
    :return: merged list.
    """
    temp = np.insert(init_2, np.arange(len(init)), init, axis=0)
    return temp


def main():

    tf.logging.set_verbosity = True
    # Convert MIDI files to message sequences.
    eventSequence = convert_files_to_eventSequence(FLAGS.data_dir)

    random_index = random.randrange(len(eventSequence))
    # Randomly choose test sequence
    test_sequence = eventSequence[random_index]

    eventSequence = np.delete(eventSequence, random_index, axis=0)
    np.random.shuffle(eventSequence)

    # Convert data to the format of input and outputs
    input_feature, notes, velocity, times = build_input_feature(eventSequence)

    # if num layer < 2, print error msg.
    if FLAGS.num_lstm_layers < 2:
        sys.exit("Number of LSTM layers should at least be two.")

    # check whether the output dir exists
    if not os.path.exists(FLAGS.output_dir): os.mkdir(FLAGS.output_dir)
    os.chdir(FLAGS.output_dir)

    output_folder = 'results_' + FLAGS.data_dir.split('/')[-1]

    # create output folder for the specific dataset
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    os.chdir(output_folder)

    cur_epoch = 0
    # check whether we can load existing model
    if glob.glob('models') and not FLAGS.overwritting:
        pre = os.getcwd()
        os.chdir('models')
        if glob.glob('*.ckpt'):
            files = glob.glob('*.ckpt')

            for file in files:
                split_name = file[:-5].split('_')
                epoch = int(split_name[-1])
                if epoch > cur_epoch: cur_epoch = epoch

            model_name = 'model_' + str(cur_epoch) + '.ckpt'
            model = tf.keras.models.load_model(model_name)
            print("Load model:", model_name)
            if FLAGS.num_epochs > cur_epoch:
                FLAGS.num_epochs = FLAGS.num_epochs - cur_epoch
            else:
                sys.exit("Existing model's num_epochs is larger than new one. Please delete the existing folder.")
        else:
            model = create_lstm_model()

        os.chdir(pre)
    else:
        model = create_lstm_model()

    # complie the model
    lossWeights = {'notes': 1.0, 'velocity': 0.1, 'time': 0.2}
    #opt = optimizers.SGD(lr=FLAGS.learning_rate)
    opt = optimizers.RMSprop(lr=FLAGS.learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', loss_weights=lossWeights, optimizer=opt, metrics=['accuracy'])

    if FLAGS.num_epochs < FLAGS.epoch_interval:
        FLAGS.epoch_interval = FLAGS.num_epochs

    epochs = 0
    # generate music and save model after each epoch_interval
    while epochs < FLAGS.num_epochs:
        epochs += FLAGS.epoch_interval

        # TRAIN
        model.fit(input_feature,
                  {'notes': notes, 'velocity': velocity, 'time': times},
                  batch_size=FLAGS.training_batch_size,
                  epochs=FLAGS.epoch_interval)

        pre = os.getcwd()
        if not os.path.exists('models'): os.mkdir('models')
        os.chdir('models')
        model.save('model_' + str(epochs+cur_epoch) + '.ckpt')
        os.chdir(pre)

        """
        init = np.array([[random.randrange(0, 128),
                         random.randrange(256, 256+len(VELOCITY)),
                         random.randrange(256+len(VELOCITY), SEQUENCE_LENGTH)] for i in range(int(FLAGS.interval/2))])

        note_offs = init[:,0]

        init_2 = np.array([[note_offs[i]+128,
                           random.randrange(256, 256+len(VELOCITY)),
                           random.randrange(256+len(VELOCITY), SEQUENCE_LENGTH)] for i in range(int(FLAGS.interval/2))])

        init = merge_init(init, init_2)
        """

        # randomly choose initial sequence from test sequence
        start_index = random.randrange(0, len(test_sequence)-FLAGS.interval)
        init = test_sequence[start_index:start_index+FLAGS.interval]

        # generate music
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
        # convert sequence generated by model to a MIDI file.
        convert_eventSequence_to_midi(generated_seq, epochs=epochs+cur_epoch)


if __name__ == '__main__':
    main()


