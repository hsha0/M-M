import os
import glob
import numpy as np
import mido
import sys

VELOCITY = [8, 20, 31, 42, 53, 64, 80, 96, 112, 127]


def read_data(data_path):
    pre = os.getcwd()
    os.chdir(data_path)
    midi_files = glob.glob('*.MID')[1:2]
    print(midi_files)

    sequences = []
    for midi in midi_files:
        eventSequence = convert_midi_to_eventSequence(midi)
        sequences.append(eventSequence)

    sequences = np.array(sequences)

    os.chdir(pre)
    return sequences


def velocity_index(true_velocity):
    diff = [abs(VELOCITY[i] - int(true_velocity)) for i in range(len(VELOCITY))]
    return diff.index(min(diff))


def preprocess_msg(msg):
    if msg.type == 'note_on':
        return [msg.note, 256+velocity_index(msg.velocity), msg.time]
    elif msg.type == 'note_off':
        return [128+msg.note, 256+velocity_index(msg.velocity), msg.time]
    else:
        return [0, 0, msg.time]


def add_time(msgs):
    msgs = np.array(msgs)
    return np.sum(msgs, axis=0)


def msg_to_event(msg):
    note_event = int(msg[0])
    velocity_event = int(msg[1])
    time_event = int(round(msg[2], 2) * 100 + 256 + len(VELOCITY))

    return [note_event, velocity_event, time_event]


def convert_midi_to_eventSequence(midi):
    mid = mido.MidiFile(midi)

    msgs = [msg for msg in mid if msg.type in ['note_on', 'note_off', 'control_change']]
    processed_msgs = np.array(list(map(preprocess_msg, msgs)))

    note_indices = np.nonzero(processed_msgs[:,:1])[0]
    processed_msgs = processed_msgs[note_indices[0]:]
    note_indices = np.nonzero(processed_msgs[:,:1])[0]+1

    split_msgs = np.split(processed_msgs, note_indices)[:-1]
    sum_msgs = np.array(list(map(add_time, split_msgs)))

    eventSequence = np.apply_along_axis(msg_to_event, 1, sum_msgs).flatten()

    return eventSequence

