import os
import glob
import numpy as np
import mido
import sys

VELOCITY = [8, 20, 31, 42, 53, 64, 80, 96, 112, 127]
SEQUENCE_LENGTH = 128+128+len(VELOCITY)+101

def convert_files_to_eventSequence(data_path):
    pre = os.getcwd()
    os.chdir(data_path)
    midi_files = glob.glob('*.MID')[:]
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
    if time_event >= SEQUENCE_LENGTH: time_event = SEQUENCE_LENGTH-1

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

    eventSequence = np.apply_along_axis(msg_to_event, 1, sum_msgs)

    return eventSequence


def single_event_to_msg(event):

    if event[1] > 256: event[1] -= 256
    if event[2] > 256+len(VELOCITY): event[2] -= 256+len(VELOCITY)
    print('time:', event[2])
    time = int(mido.second2tick(event[2]/100 + 0.01, 480, 500000))
    if event[0] < 128:
        msg = mido.Message('note_on',
                           note=event[0],
                           velocity=VELOCITY[event[1]],
                           time=time)

    else:
        msg = mido.Message('note_off',
                           note=event[0]-128,
                           velocity=VELOCITY[event[1]],
                           time=time)
    return msg


def convert_eventSequence_to_midi(seq, epochs):
    msgs = list(map(single_event_to_msg, seq))

    index = 0
    for i in range(len(msgs)):
        if msgs[i].type == 'note_off':
            index = i
        else:
            break

    msgs = msgs[index:]
    print(msgs[0])

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.extend(msgs)

    mid.tracks.append(track)

    mid.save('new_song_epoch' +  str(epochs) + '.mid')
