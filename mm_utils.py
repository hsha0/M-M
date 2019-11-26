import os
import glob
import sys
import tensorflow as tf
import mido
import numpy as np
import copy

VELOCITY = [8, 20, 31, 42, 53, 64, 80, 96, 112, 127]


class NoteOn:
    def __init__(self, note):
        self.note = note


class NoteOff:
    def __init__(self, note):
        self.note = note


class TimeShift:
    def __init__(self, time_shift):
        self.time_shift = time_shift


class Velocity:
    def __init__(self, velocity):
        self.velocity = velocity


def read_data(data_path):
    pre = os.getcwd()
    os.chdir(data_path)

    midi_files = glob.glob('*.MID')
    sequences = []
    for midi_file in midi_files[:1]:
        eventSequence = convert_midi_to_eventSequence(midi_file)
        sequences.append(eventSequence)

    sequences = np.array(sequences)

    os.chdir(pre)
    return sequences


def single_one_hot(event):

    init_sequence = [0] * (128 + 128 + len(VELOCITY) + 101)
    sequence = []
    if isinstance(event, NoteOn):
        sequence = copy.deepcopy(init_sequence)
        sequence[int(event.note)] = 1
    elif isinstance(event, NoteOff):
        sequence = copy.deepcopy(init_sequence)
        sequence[128 + int(event.note)] = 1
    elif isinstance(event, Velocity):
        sequence = copy.deepcopy(init_sequence)
        sequence[128 + 128 + VELOCITY.index(event.velocity)] = 1
    elif isinstance(event, TimeShift):
        sequence = copy.deepcopy(init_sequence)
        sequence[128 + 128 + len(VELOCITY) + int(float(event.time_shift) * 100)] = 1
    else:
        print(event)

    return sequence


def closest_velocity(true_velocity):
    diff = [abs(VELOCITY[i] - int(true_velocity)) for i in range(len(VELOCITY))]
    return VELOCITY[diff.index(min(diff))]


def convert_midi_to_eventSequence(midi_file):
    mid = mido.MidiFile(midi_file)
    eventSequence = []

    time_shift = 0
    for msg in mid:

        msg = str(msg)
        if msg.startswith('<meta'): continue
        if msg.startswith('sysex'): continue

        msg = msg.split(' ')
        event = msg[0]
        info = {msg[i].split("=")[0]: msg[i].split("=")[1] for i in range(1, len(msg))}

        if event == 'note_on':

            if time_shift > 1: time_shift = 1.0
            time_shift = str(round(time_shift, 2))

            cloest_v = closest_velocity(info['velocity'])

            note_on = NoteOn(note=info['note'])
            velocity = Velocity(velocity=cloest_v)
            time_shift = TimeShift(time_shift=time_shift)

            note_on = single_one_hot(note_on)
            velocity = single_one_hot(velocity)
            time_shift = single_one_hot(time_shift)

            eventSequence.extend([note_on, velocity, time_shift])

            time_shift = 0
        elif event == 'note_off':
            note_off = NoteOff(note=info['note'])
            if time_shift > 1: time_shift = 1.0
            time_shift = str(round(time_shift, 2))
            time_shift = TimeShift(time_shift=time_shift)
            cloest_v = closest_velocity(info['velocity'])
            velocity = Velocity(velocity=cloest_v)

            note_off = single_one_hot(note_off)
            velocity = single_one_hot(velocity)
            time_shift = single_one_hot(time_shift)

            eventSequence.extend([note_off, velocity, time_shift])
            time_shift = 0
        else:
            time_shift += float(info['time'])

    return np.array(eventSequence)

def convert_eventSequence_to_midi(eventSequence):
    for event in eventSequence:
        print(event)
        mid = mido.MidiFile()

        cur_note_on = []
        if event <= 127:


        elif event >= 128 and event <= 255:

        elif event >= 256 and event <= 256+len(VELOCITY)-1:

        elif event >= 256+len(VELOCITY):



