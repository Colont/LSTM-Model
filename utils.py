import pretty_midi
import pandas as pd
import collections
import numpy as np
import tensorflow as tf
def convert_notes(midi):

    pm = pretty_midi.PrettyMIDI(midi)
    
    instrument = pm.instruments[0]
    #instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    #print('Instrument name:', instrument_name)
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')

    for note in sorted_notes:
        start, end = note.start, note.end
        notes['start'].append(start)
        notes['end'].append(end) 
        notes['pitch'].append(note.pitch)
        notes['duration'].append(end - start)
        notes['step'].append(start - prev_start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def convert_to_midi(notes, out_file, instrument_name, velocity = 100):
  
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def create_sequences(dataset, seq_length, key_order, vocab_size):
    seq_length = seq_length + 1
    windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x/[vocab_size,1.0,1.0]
        return x
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

        return scale_pitch(inputs), labels
    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
