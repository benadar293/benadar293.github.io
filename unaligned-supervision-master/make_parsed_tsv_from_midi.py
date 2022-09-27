'''
Convert midi to a note-list representation in a tsv file. Each line in the tsv file will contain
information of a single note: onset time, offset time, note number, velocity, and instrument.
'''


import numpy as np
import os
import warnings
from onsets_and_frames.midi_utils import parse_midi_multi
warnings.filterwarnings("ignore")


def midi2tsv_process(midi_path, target_path, shift=0, force_instrument=None):
    midi = parse_midi_multi(midi_path, force_instrument=force_instrument)
    print(target_path)
    if shift != 0:
        midi[:, 2] += shift
    np.savetxt(target_path, midi,
               fmt='%1.6f', delimiter='\t', header='onset,offset,note,velocity,instrument')


midi_src_pth = '/path/to/midi/perfromance'
target = '/disk4/ben/UnalignedSupervision/NoteEM_tsv'


FORCE_INSTRUMENT = None
piece = midi_src_pth.split('/')[-1]
os.makedirs(target + '/' + piece, exist_ok=True)
for f in os.listdir(midi_src_pth):
    if f.endswith('.mid') or f.endswith('.MID'):
        print(f)
        midi2tsv_process(midi_src_pth + '/' + f,
                         target + '/' + piece + '/' + f.replace('.mid', '.tsv').replace('.MID', '.tsv'),
                         force_instrument=FORCE_INSTRUMENT)