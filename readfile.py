import tensorflow as tf
import pathlib
import glob
import pandas as pd
import numpy as np
from utils import convert_notes, create_sequences


dir = pathlib.Path('/Users/colinjohnson/Documents/code/GitHub/Music Project/maestro-v2.0.0')
files = glob.glob(str(dir/'**/*.mid*'))
SAMPLING_RATE = 16000
note_list = []
for file in files[:5:]:
    notes = convert_notes(file)
    note_list.append(notes)



all_notes = pd.concat(note_list)
key_order = ['pitch', 'step', 'duration']
training_notes = np.stack([all_notes[key] for key in key_order], axis=1)
notes_ds = tf.data.Dataset.from_tensor_slices(training_notes)

seq_ds = create_sequences(notes_ds, 
                          seq_length=75, 
                          key_order=key_order,
                          vocab_size=128)

batch_size = 64
seq_len = 25
buffer_size = len(all_notes) - seq_len 
train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))

print(train_ds)
