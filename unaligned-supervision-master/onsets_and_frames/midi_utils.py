import numpy as np
from onsets_and_frames.constants import *
import mido
from mido import Message, MidiFile, MidiTrack
from .utils import max_inst
from datetime import datetime


def midi_to_hz(m):
    return 440. * (2. ** ((m - 69.) / 12.))

def hz_to_midi(h):
    return 12. * np.log2(h / (440.)) + 69.

def midi_to_frames(midi, instruments, conversion_map=None):
    n_keys = MAX_MIDI - MIN_MIDI + 1
    midi_length = int((max(midi[:, 1]) + 1) * SAMPLE_RATE)
    n_steps = (midi_length - 1) // HOP_LENGTH + 1
    n_channels = len(instruments) + 1
    label = torch.zeros(n_steps, n_keys * n_channels, dtype=torch.uint8)
    for onset, offset, note, vel, instrument in midi:
        f = int(note) - MIN_MIDI
        if 104 > instrument > 87 or instrument > 111:
            continue
        if f >= n_keys or f < 0:
            continue
        assert 0 < vel < 128
        instrument = int(instrument)
        if conversion_map is not None:
            if instrument not in conversion_map:
                continue
            instrument = conversion_map[instrument]
        left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
        onset_right = min(n_steps, left + HOPS_IN_ONSET)
        frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
        frame_right = min(n_steps, frame_right)
        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
        if int(instrument) not in instruments:
            continue
        chan = instruments.index(int(instrument))
        label[left:onset_right, n_keys * chan + f] = 3
        label[onset_right:frame_right, n_keys * chan + f] = 2
        label[frame_right:offset_right, n_keys * chan + f] = 1

        inv_chan = len(instruments)
        label[left:onset_right, n_keys * inv_chan + f] = 3
        label[onset_right:frame_right, n_keys * inv_chan + f] = 2
        label[frame_right:offset_right, n_keys * inv_chan + f] = 1
    return label

'''
Convert piano roll to list of notes, pitch only.
'''
def extract_notes_np_pitch(onsets, frames, velocity,
                           onset_threshold=0.5, frame_threshold=0.5):
    onsets = (onsets > onset_threshold).astype(np.uint8)
    frames = (frames > frame_threshold).astype(np.uint8)
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in np.transpose(np.nonzero(onset_diff)):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch] or frames[offset, pitch]:
            if onsets[offset, pitch]:
                velocity_samples.append(velocity[offset, pitch])
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
    return np.array(pitches), np.array(intervals), np.array(velocities)


'''
Convert piano roll to list of notes, pitch and instrument.
'''
def extract_notes_np(onsets, frames, velocity,
                  onset_threshold=0.5, frame_threshold=0.5):
    onsets = (onsets > onset_threshold).astype(np.uint8)
    frames = (frames > frame_threshold).astype(np.uint8)
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

    if onsets.shape[-1] != frames.shape[-1]:
        num_instruments = onsets.shape[1] / frames.shape[1]
        assert num_instruments.is_integer()
        num_instruments = int(num_instruments)
        frames = np.tile(frames, (1, num_instruments))

    pitches = []
    intervals = []
    velocities = []
    instruments = []

    for nonzero in np.transpose(np.nonzero(onset_diff)):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch] or frames[offset, pitch]:
            if onsets[offset, pitch]:
                velocity_samples.append(velocity[offset, pitch])
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitch, instrument = pitch % N_KEYS, pitch // N_KEYS

            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
            instruments.append(instrument)
    return np.array(pitches), np.array(intervals), np.array(velocities), np.array(instruments)


def append_track_multi(file, pitches, intervals, velocities, ins, single_ins=False):
    track = MidiTrack()
    file.tracks.append(track)
    chan = len(file.tracks) - 1
    if chan >= DRUM_CHANNEL:
        chan += 1
    track.append(Message('program_change', channel=chan, program=ins if not single_ins else 0, time=0))

    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], channel=chan, note=pitch, velocity=velocity, time=current_tick - last_tick))
        # try:
        #     track.append(Message('note_' + event['type'], channel=chan, note=pitch, velocity=velocity, time=current_tick - last_tick))
        # except Exception as e:
        #     print('Err Message', 'note_' + event['type'], pitch, velocity, current_tick - last_tick)
        #     track.append(Message('note_' + event['type'], channel=chan, note=pitch, velocity=max(0, velocity), time=current_tick - last_tick))
        #     if velocity >= 0:
        #         raise e
        last_tick = current_tick


def append_track(file, pitches, intervals, velocities):
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        try:
            track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        except Exception as e:
            print('Err Message', 'note_' + event['type'], pitch, velocity, current_tick - last_tick)
            track.append(Message('note_' + event['type'], note=pitch, velocity=max(0, velocity), time=current_tick - last_tick))
            if velocity >= 0:
                raise e
        last_tick = current_tick


def save_midi(path, pitches, intervals, velocities, insts=None):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    if isinstance(pitches, list):
        for p, i, v, ins in zip(pitches, intervals, velocities, insts):
            append_track_multi(file, p, i, v, ins)
    else:
        append_track(file, pitches, intervals, velocities)
    file.save(path)


def frames2midi(save_path, onsets, frames, vels,
                onset_threshold=0.5, frame_threshold=0.5, scaling=HOP_LENGTH / SAMPLE_RATE,
                inst_mapping=None):
    p_est, i_est, v_est, inst_est = extract_notes_np(onsets, frames, vels,
                                        onset_threshold, frame_threshold)
    i_est = (i_est * scaling).reshape(-1, 2)

    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    inst_set = set(inst_est)
    inst_set = sorted(list(inst_set))

    p_est_lst = {}
    i_est_lst = {}
    v_est_lst = {}
    assert len(p_est) == len(i_est) == len(v_est) == len(inst_est)
    for p, i, v, ins in zip(p_est, i_est, v_est, inst_est):
        if ins in p_est_lst:
            p_est_lst[ins].append(p)
        else:
            p_est_lst[ins] = [p]
        if ins in i_est_lst:
            i_est_lst[ins].append(i)
        else:
            i_est_lst[ins] = [i]
        if ins in v_est_lst:
            v_est_lst[ins].append(v)
        else:
            v_est_lst[ins] = [v]
    for elem in [p_est_lst, i_est_lst, v_est_lst]:
        for k, v in elem.items():
            elem[k] = np.array(v)
    inst_set = [e for e in inst_set if e in p_est_lst]
    # inst_set = [INSTRUMENT_MAPPING[e] for e in inst_set if e in p_est_lst]
    p_est_lst = [p_est_lst[ins] for ins in inst_set if ins in p_est_lst]
    i_est_lst = [i_est_lst[ins] for ins in inst_set if ins in i_est_lst]
    v_est_lst = [v_est_lst[ins] for ins in inst_set if ins in v_est_lst]
    assert len(p_est_lst) == len(i_est_lst) == len(v_est_lst) == len(inst_set)
    inst_set = [inst_mapping[e] for e in inst_set]
    save_midi(save_path,
              p_est_lst, i_est_lst, v_est_lst,
              inst_set)


def frames2midi_pitch(save_path, onsets, frames, vels,
                onset_threshold=0.5, frame_threshold=0.5, scaling=HOP_LENGTH / SAMPLE_RATE):
    p_est, i_est, v_est = extract_notes_np_pitch(onsets, frames, vels,
                                                 onset_threshold, frame_threshold)
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
    print('Saving midi in', save_path)
    save_midi(save_path, p_est, i_est, v_est)


def parse_midi_multi(path, force_instrument=None):
    """open midi file and return np.array of (onset, offset, note, velocity, instrument) rows"""
    try:
        midi = mido.MidiFile(path)
    except:
        print('could not open midi', path)
        return

    time = 0

    events = []

    control_changes = []
    program_changes = []

    sustain = {}

    all_channels = set()

    instruments = {}  # mapping of channel: instrument

    for message in midi:
        time += message.time
        if hasattr(message, 'channel'):
            if message.channel == DRUM_CHANNEL:
                continue

        if message.type == 'control_change' and message.control == 64 \
                and (message.value >= 64) != sustain.get(message.channel, False):
            sustain[message.channel] = message.value >= 64
            event_type = 'sustain_on' if sustain[message.channel] else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            event['channel'] = message.channel
            event['sustain'] = sustain[message.channel]
            events.append(event)

        if message.type == 'control_change' and message.control != 64:
            control_changes.append((time, message.control, message.value, message.channel))

        if message.type == 'program_change':
            program_changes.append((time, message.program, message.channel))
            instruments[message.channel] = instruments.get(message.channel, []) + [(message.program, time)]

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note,
                         velocity=velocity, sustain=sustain.get(message.channel, False))
            event['channel'] = message.channel
            events.append(event)

        if hasattr(message, 'channel'):
            all_channels.add(message.channel)

    if len(instruments) == 0:
        instruments = {c: [(0, 0)] for c in all_channels}
    if len(all_channels) > len(instruments):
        for e in all_channels - set(instruments.keys()):
            instruments[e] = [(0, 0)]

    if force_instrument is not None:
        instruments = {c: [(force_instrument, 0)] for c in all_channels}

    this_instruments = set()
    for v in instruments.values():
        this_instruments = this_instruments.union(set(x[0] for x in v))

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue
        offset = next(n for n in events[i + 1:] if (n['note'] == onset['note']
                                                        and n['channel'] == onset['channel'])
                          or n is events[-1])
        if 'sustain' not in offset:
            print('offset without sustain', offset)
        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:] if (n['type'] == 'sustain_off'
                                                                    and n['channel'] == onset['channel']
                                                                      )
                                                                    or n is events[-1])
        for k, v in instruments.items():
            if len(set(v)) == 1 and len(v) > 1:
                instruments[k] = list(set(v))
        for k, v in instruments.items():
            instruments[k] = sorted(v, key=lambda x: x[1])
        if len(instruments[onset['channel']]) == 1:
            instrument = instruments[onset['channel']][0][0]
        else:
            ind = 0
            while ind < len(instruments[onset['channel']]) and onset['time'] >= instruments[onset['channel']][ind][1]:
                ind += 1
            if ind > 0:
                ind -= 1
            instrument = instruments[onset['channel']][ind][0]
        if onset['channel'] == DRUM_CHANNEL:
            print('skipping drum note')
            continue
        note = (onset['time'], offset['time'], onset['note'], onset['velocity'], instrument)
        notes.append(note)

    res = np.array(notes)
    return res


def save_midi_alignments_and_predictions(save_path, data_path, inst_mapping,
                                         aligned_onsets, aligned_frames,
                                         onset_pred_np, frame_pred_np, prefix='', use_time=True):
    inst_only = len(inst_mapping) * N_KEYS
    time_now = datetime.now().strftime('%y%m%d-%H%M%S') if use_time else ''
    if len(prefix) > 0:
        prefix = '_{}'.format(prefix)

    # Save the aligned label. If training on a small dataset or a single performance in order to label it for later adding it
    # to a large dataset, it is recommended to use this MIDI as a label.
    frames2midi(save_path + '/' + data_path.replace('.flac', '').split('/')[-1] + prefix + '_alignment_' + time_now + '.mid',
                aligned_onsets[:, : inst_only], aligned_frames[:, : inst_only],
                64. * aligned_onsets[:, : inst_only],
                inst_mapping=inst_mapping)


    # # Aligned label, pitch-only, on the piano.
    # frames2midi_pitch(save_path + '/' + data_path.replace('.flac', '').split('/')[-1] + prefix + '_alignment_pitch_' + time_now + '.mid',
    #                   aligned_onsets[:, -N_KEYS:], aligned_frames[:, -N_KEYS:],
    #                   64. * aligned_onsets[:, -N_KEYS:])


    predicted_onsets = onset_pred_np >= 0.5
    predicted_frames = frame_pred_np >= 0.5


    # # Raw pitch with instrument prediction - will probably have lower recall, depending on the model's strength.
    # frames2midi(save_path + '/' + data_path.replace('.flac', '').split('/')[-1] + prefix + '_pred_' + time_now + '.mid',
    #             predicted_onsets[:, : inst_only], predicted_frames[:, : inst_only],
    #             64. * predicted_onsets[:, : inst_only],
    #             inst_mapping=inst_mapping)


    # Pitch prediction played on the piano - will have high recall, since it does not differentiate between instruments.
    frames2midi_pitch(save_path + '/' + data_path.replace('.flac', '').split('/')[-1] + prefix + '_pred_pitch_' + time_now + '.mid',
                      predicted_onsets[:, -N_KEYS:], predicted_frames[:, -N_KEYS:],
                      64. * predicted_onsets[:, -N_KEYS:])


    # Pitch prediction, with choice of most likely instrument for each detected note.
    if len(inst_mapping) > 1:
        max_pred_onsets = max_inst(onset_pred_np)
        frames2midi(save_path + '/' + data_path.replace('.flac', '').split('/')[-1] + prefix + '_pred_max_' + time_now + '.mid',
                    max_pred_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                    64. * max_pred_onsets[:, : inst_only],
                    inst_mapping=inst_mapping)