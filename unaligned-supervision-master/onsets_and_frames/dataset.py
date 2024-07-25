# This software is for non-commercial use only.
# Commercial use requires a separate license.
# This software is covered by US Patent Application No. 63/312,219.

from .constants import *
import numpy as np
from dtw import *
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import os
from onsets_and_frames.mel import melspectrogram
from datetime import datetime
from onsets_and_frames.midi_utils import *
from onsets_and_frames.utils import *
import time


class EMDATASET(Dataset):
    def __init__(self,
                 audio_path='NoteEM_audio',
                 labels_path='NoteEm_labels',
                 groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE,
                 instrument_map=None, update_instruments=False, transcriber=None,
                 conversion_map=None):
        self.audio_path = audio_path
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.groups = groups
        self.conversion_map = conversion_map
        self.file_list = self.files(self.groups)
        if instrument_map is None:
            self.get_instruments()
        else:
            self.instruments = instrument_map
            if update_instruments:
                self.add_instruments()
        self.transcriber = transcriber
        self.load_pts(self.file_list)
        self.data = []
        print('Reading files...')
        for input_files in tqdm(self.file_list):
            data = self.pts[input_files[0]]
            audio_len = len(data['audio'])
            minutes = audio_len // (SAMPLE_RATE * 60)
            copies = minutes
            for _ in range(copies):
                self.data.append(input_files)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def files(self, groups):
        self.path = 'NoteEM_audio'
        tsvs_path = 'NoteEM_tsv'
        res = []
        good_ids = list(range(2075, 2084))
        # good_ids += list(range(1817, 1820))
        # good_ids += list(range(2202, 2205))
        # good_ids += list(range(2415, 2418))
        # good_ids += list(range(2504, 2508))
        for group in groups:
            tsvs = os.listdir(tsvs_path + '/' + group)
            tsvs = sorted(tsvs)
            for shft in range(-5, 6):
                curr_fls_pth = self.path + '/' + group + '#{}'.format(shft)
                fls = os.listdir(curr_fls_pth)
                fls = sorted(fls)
                for f, t in zip(fls, tsvs):
                    # #### MusicNet
                    if 'MusicNet' in group:
                        if all([str(elem) not in f for elem in good_ids]):
                            continue
                    res.append((curr_fls_pth + '/' + f, tsvs_path + '/' + group + '/' + t))
        return res

    def get_instruments(self):
        instruments = set()
        for _, f in self.file_list:
            print('loading midi from', f)
            events = np.loadtxt(f, delimiter='\t', skiprows=1)
            curr_instruments = set(events[:, -1])
            instruments = instruments.union(curr_instruments)
        instruments = [int(elem) for elem in instruments if elem < 115]
        instruments = list(set(instruments))
        if 0 in instruments:
            piano_ind = instruments.index(0)
            instruments.pop(piano_ind)
            instruments.insert(0, 0)
        self.instruments = instruments
        self.instruments = list(set(self.instruments) - set(range(88, 104)) - set(range(112, 150)))
        print('Dataset instruments:', self.instruments)
        print('Total:', len(self.instruments), 'instruments')

    def add_instruments(self):
        for _, f in self.file_list:
            events = np.loadtxt(f, delimiter='\t', skiprows=1)
            curr_instruments = set(events[:, -1])
            new_instruments = curr_instruments - set(self.instruments)
            self.instruments += list(new_instruments)
        instruments = [int(elem) for elem in self.instruments if (elem < 115)]
        self.instruments = instruments

    def __getitem__(self, index):
        data = self.load(*self.data[index])
        result = dict(path=data['path'])
        midi_length = len(data['label'])
        n_steps = self.sequence_length // HOP_LENGTH
        step_begin = self.random.randint(midi_length - n_steps)
        step_end = step_begin + n_steps
        begin = step_begin * HOP_LENGTH
        end = begin + self.sequence_length
        result['audio'] = data['audio'][begin:end]
        diff = self.sequence_length - len(result['audio'])
        result['audio'] = torch.cat((result['audio'], torch.zeros(diff, dtype=result['audio'].dtype)))
        result['audio'] = result['audio'].to(self.device)
        result['label'] = data['label'][step_begin:step_end, ...]
        result['label'] = result['label'].to(self.device)
        if 'velocity' in data:
            result['velocity'] = data['velocity'][step_begin:step_end, ...].to(self.device)
            result['velocity'] = result['velocity'].float() / 128.

        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()

        if 'onset_mask' in data:
            result['onset_mask'] = data['onset_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_mask' in data:
            result['frame_mask'] = data['frame_mask'][step_begin:step_end, ...].to(self.device).float()


        shape = result['frame'].shape
        keys = N_KEYS
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        # frame and offset currently do not differentiate between instruments,
        # so we compress them across instrument and save a copy of the original,
        # as 'big_frame' and 'big_offset'
        result['big_frame'] = result['frame']
        result['frame'], _ = result['frame'].reshape(new_shape).max(axis=-2)
        result['big_offset'] = result['offset']
        result['offset'], _ = result['offset'].reshape(new_shape).max(axis=-2)
        return result

    def load(self, audio_path, tsv_path):
        data = self.pts[audio_path]
        if len(data['audio'].shape) > 1:
            data['audio'] = (data['audio'].float().mean(dim=-1)).short()
        if 'label' in data:
            return data
        else:
            piece, part = audio_path.split('/')[-2:]
            piece_split = piece.split('#')
            if len(piece_split) == 2:
                piece, shift1 = piece_split
            else:
                piece, shift1 = '#'.join(piece_split[:2]), piece_split[-1]
            part_split = part.split('#')
            if len(part_split) == 2:
                part, shift2 = part_split
            else:
                part, shift2 = '#'.join(part_split[:2]), part_split[-1]
            shift2, _ = shift2.split('.')
            assert shift1 == shift2
            shift = shift1
            assert shift != 0
            orig = audio_path.replace('#{}'.format(shift), '#0')
            res = {}
            res['label'] = shift_label(self.pts[orig]['label'], int(shift))
            res['path'] = audio_path
            res['audio'] = data['audio']
            if 'velocity' in self.pts[orig]:
                res['velocity'] = shift_label(self.pts[orig]['velocity'], int(shift))
            if 'onset_mask' in self.pts[orig]:
                res['onset_mask'] = shift_label(self.pts[orig]['onset_mask'], int(shift))
            if 'frame_mask' in self.pts[orig]:
                res['frame_mask'] = shift_label(self.pts[orig]['frame_mask'], int(shift))
            return res

    def load_pts(self, files):
        self.pts = {}
        print('loading pts...')
        for flac, tsv in tqdm(files):
            print('flac, tsv', flac, tsv)
            if os.path.isfile(self.labels_path + '/' +
                              flac.split('/')[-1].replace('.flac', '.pt')):
                self.pts[flac] = torch.load(self.labels_path + '/' +
                              flac.split('/')[-1].replace('.flac', '.pt'))
            else:
                if flac.count('#') != 2:
                    print('two #', flac)
                audio, sr = soundfile.read(flac, dtype='int16')
                if len(audio.shape) == 2:
                    audio = audio.astype(float).mean(axis=1)
                else:
                    audio = audio.astype(float)
                audio = audio.astype(np.int16)
                print('audio len', len(audio))
                assert sr == SAMPLE_RATE
                audio = torch.ShortTensor(audio)
                if '#0' not in flac:
                    assert '#' in flac
                    data = {'audio': audio}
                    self.pts[flac] = data
                    torch.save(data,
                               self.labels_path + '/' + flac.split('/')[-1]
                               .replace('.flac', '.pt').replace('.mp3', '.pt'))
                    continue
                midi = np.loadtxt(tsv, delimiter='\t', skiprows=1)
                unaligned_label = midi_to_frames(midi, self.instruments, conversion_map=self.conversion_map)
                data = dict(path=self.labels_path + '/' + flac.split('/')[-1],
                            audio=audio, unaligned_label=unaligned_label)
                torch.save(data, self.labels_path + '/' + flac.split('/')[-1]
                               .replace('.flac', '.pt').replace('.mp3', '.pt'))
                self.pts[flac] = data

    '''
    Update labels. 
    POS, NEG - pseudo labels positive and negative thresholds.
    PITCH_POS - pseudo labels positive thresholds for the pitch-only classes.
    first - is this the first labelling iteration.
    update - should the labels indeed be updated - if not, just saves the output.
    BEST_BON - if true, will update labels only if the bag of notes distance between the unaligned midi and the prediction improved.
    Bag of notes distance is computed based on pitch only.
    '''
    def update_pts(self, transcriber, POS=1.1, NEG=-0.001, FRAME_POS=0.5,
                   to_save=None, first=False, update=True, BEST_BON=False):
        print('Updating pts...')
        print('POS, NEG', POS, NEG)
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        print('there are', len(self.pts), 'pts')
        for flac, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            audio_inp = data['audio'].float() / 32768.
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
                print('long audio, splitting to {} segments'.format(n_segments))
                seg_len = audio_inp_len // n_segments
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                vel_preds = []
                for i_s in range(n_segments):
                    curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
                    curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                    vel_preds.append(curr_velocity_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
                velocity_pred = torch.cat(vel_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)
            print('done predicting.')
            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()

            peaks = get_peaks(onset_pred, 3) # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
            onset_pred[~peaks] = 0

            unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
            unaligned_frames = (data['unaligned_label'] >= 2).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()

            ####
            pred_bag_of_notes = (onset_pred_np[:, -N_KEYS:] >= 0.5).sum(axis=0)
            gt_bag_of_notes = unaligned_onsets[:, -N_KEYS:].astype(bool).sum(axis=0)
            bon_dist = (((pred_bag_of_notes - gt_bag_of_notes) ** 2).sum()) ** 0.5
            # print('pred bag of notes', pred_bag_of_notes)
            # print('gt bag of notes', gt_bag_of_notes)
            bon_dist /= gt_bag_of_notes.sum()
            print('bag of notes dist', bon_dist)
            ####

            # We align based on likelihoods regardless of the octave (chroma features)
            onset_pred_comp = compress_across_octave(onset_pred_np[:, -N_KEYS:])
            onset_label_comp = compress_across_octave(unaligned_onsets[:, -N_KEYS:])
            # We can do DTW on super-frames since anyway we search for local max afterwards
            onset_pred_comp = compress_time(onset_pred_comp, DTW_FACTOR)
            onset_label_comp = compress_time(onset_label_comp, DTW_FACTOR)
            print('dtw lengths', len(onset_pred_comp), len(onset_label_comp))
            init_time = time.time()
            alignment = dtw(onset_pred_comp, onset_label_comp, dist_method='euclidean',
                            )
            finish_time = time.time()
            print('DTW took {} seconds.'.format(finish_time - init_time))
            index1, index2 = alignment.index1, alignment.index2
            matches1, matches2 = get_matches(index1, index2), get_matches(index2, index1)

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_offsets = np.zeros(onset_pred_np.shape, dtype=bool)

            # We go over onsets (t, f) in the unaligned midi. For each onset, we find its approximate time based on DTW,
            # then find its precise time with likelihood local max
            for t, f in zip(*unaligned_onsets.nonzero()):
                t_comp = t // DTW_FACTOR
                t_src = matches2[t_comp]
                t_sources = list(range(DTW_FACTOR * min(t_src), DTW_FACTOR * max(t_src) + 1))
                # we extend the search area of local max to be ~0.5 second:
                t_sources_extended = get_margin(t_sources, len(aligned_onsets))
                # eliminate occupied positions. Allow only a single onset per 5 frames:
                existing_eliminated = [t_source for t_source in t_sources_extended if (aligned_onsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(existing_eliminated) > 0:
                    t_sources_extended = existing_eliminated

                t_src = max(t_sources_extended, key=lambda x: onset_pred_np[x, f]) # t_src is the most likely time in the local neighborhood for this note onset
                f_pitch = (len(self.instruments) * N_KEYS) + (f % N_KEYS)
                if onset_pred_np[t_src, f_pitch] < NEG: # filter negative according to pitch-only likelihood (can use f instead)
                    continue
                aligned_onsets[t_src, f] = 1 # set the label

                # Now we need to decide note duration and offset time. Find note length in unaligned midi:
                t_off = t
                while t_off < len(unaligned_frames) and unaligned_frames[t_off, f]:
                    t_off += 1
                note_len = t_off - t # this is the note length in the unaligned midi. We need note length in the audio.

                # option 1: use mapping, traverse note length in the unaligned midi, and then use the reverse mapping:
                try:
                    t_off_src1 = max(matches2[(DTW_FACTOR * max(matches1[t_src // DTW_FACTOR]) + note_len) // DTW_FACTOR]) * DTW_FACTOR
                    t_off_src1 = max(t_src + 1, t_off_src1)
                except Exception as e:
                    t_off_src1 = len(aligned_offsets)
                # option 2: use relative note length
                t_off_src2 = t_src + int(note_len * (len(aligned_onsets) / len(unaligned_onsets)))
                t_off_src2 = min(len(aligned_onsets), t_off_src2)

                t_off_src = t_off_src2 # we choose option 2
                aligned_frames[t_src: t_off_src, f] = 1

                if t_off_src < len(aligned_offsets):
                    aligned_offsets[t_off_src, f] = 1

            # eliminate instruments that do not exist in the unaligned midi
            inactive_instruments, active_instruments_list = get_inactive_instruments(unaligned_onsets, len(aligned_onsets))
            onset_pred_np[inactive_instruments] = 0

            pseudo_onsets = (onset_pred_np >= POS) & (~aligned_onsets)
            inst_only = len(self.instruments) * N_KEYS
            if first: # do not use pseudo labels for instruments in first labelling iteration since the model doesn't distinguish yet
                pseudo_onsets[:, : inst_only] = 0
            onset_label = np.maximum(pseudo_onsets, aligned_onsets)

            onsets_unknown = (onset_pred_np >= 0.5) & (~onset_label) # for mask
            if first: # do not use mask for instruments in first labelling iteration since the model doesn't distinguish yet between instruments
                onsets_unknown[:, : inst_only] = 0
            onset_mask = torch.from_numpy(~onsets_unknown).byte()
            # onset_mask = torch.ones(onset_label.shape).byte()

            pseudo_frames = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            pseudo_offsets = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            for t, f in zip(*onset_label.nonzero()):
                t_off = t
                while t_off < len(pseudo_frames) and frame_pred[t_off, f % N_KEYS] >= FRAME_POS:
                    t_off += 1
                pseudo_frames[t: t_off, f] = 1
                if t_off < len(pseudo_offsets):
                    pseudo_offsets[t_off, f] = 1
            frame_label = np.maximum(pseudo_frames, aligned_frames)
            offset_label = get_diff(frame_label, offset=True)

            frames_pitch_only = frame_label[:, -N_KEYS:]
            frames_unknown = (frame_pred_np >= 0.5) & (~frames_pitch_only)
            frame_mask = torch.from_numpy(~frames_unknown).byte()
            # frame_mask = torch.ones(frame_pred.shape).byte()

            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            if to_save is not None:
                save_midi_alignments_and_predictions(to_save, data['path'], self.instruments,
                                         aligned_onsets, aligned_frames,
                                         onset_pred_np, frame_pred_np, prefix='')
                # time_now = datetime.now().strftime('%y%m%d-%H%M%S')
                # frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_' + time_now + '.mid',
                #             aligned_onsets[:, : inst_only], aligned_frames[:, : inst_only],
                #             64. * aligned_onsets[:, : inst_only],
                #             inst_mapping=self.instruments)
                # frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_pitch_' + time_now + '.mid',
                #                 aligned_onsets[:, -N_KEYS:], aligned_frames[:, -N_KEYS:],
                #                 64. * aligned_onsets[:, -N_KEYS:])
                # predicted_onsets = onset_pred_np >= 0.5
                # predicted_frames = frame_pred_np >= 0.5
                # frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_' + time_now + '.mid',
                #             predicted_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                #             64. * predicted_onsets[:, : inst_only],
                #             inst_mapping=self.instruments)
                # frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_pitch_' + time_now + '.mid',
                #             predicted_onsets[:, -N_KEYS:], predicted_frames[:, -N_KEYS:],
                #             64. * predicted_onsets[:, -N_KEYS:])
                # if len(self.instruments) > 1:
                #     max_pred_onsets = max_inst(onset_pred_np)
                #     frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_max_' + time_now + '.mid',
                #                 max_pred_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                #                 64. * max_pred_onsets[:, : inst_only],
                #                 inst_mapping=self.instruments)
            if update:
                if not BEST_BON or bon_dist < data.get('BON', float('inf')):
                    data['label'] = torch.from_numpy(label).byte()
                    data['onset_mask'] = onset_mask
                    data['frame_mask'] = frame_mask
                if bon_dist < data.get('BON', float('inf')):
                    print('Bag of notes distance improved from {} to {}'.format(data.get('BON', float('inf')), bon_dist))
                    data['BON'] = bon_dist

                    if to_save is not None:
                        os.makedirs(to_save + '/BEST_BON', exist_ok=True)
                        save_midi_alignments_and_predictions(to_save + '/BEST_BON', data['path'], self.instruments,
                                                             aligned_onsets, aligned_frames,
                                                             onset_pred_np, frame_pred_np, prefix='BEST_BON')

            velocity_pred = velocity_pred.detach().squeeze().cpu()
            # velocity_pred = torch.from_numpy(new_vels)
            velocity_pred = (128. * velocity_pred)
            velocity_pred[velocity_pred < 0.] = 0.
            velocity_pred[velocity_pred > 127.] = 127.
            velocity_pred = velocity_pred.byte()
            if update:
                data['velocity'] = velocity_pred

            del audio_inp
            try:
                del mel
            except:
                pass
            del onset_pred
            del offset_pred
            del frame_pred
            del velocity_pred
            torch.cuda.empty_cache()

    '''
        Update labels. Use only alignment without pseudo-labels.
    '''

    def update_pts_vanilla(self, transcriber,
                   to_save=None, first=False, update=True):
        print('Updating pts...')
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        print('there are', len(self.pts), 'pts')
        for flac, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            audio_inp = data['audio'].float() / 32768.
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
                print('long audio, splitting to {} segments'.format(n_segments))
                seg_len = audio_inp_len // n_segments
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                vel_preds = []
                for i_s in range(n_segments):
                    curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
                    curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                    vel_preds.append(curr_velocity_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
                velocity_pred = torch.cat(vel_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)
            print('done predicting.')
            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()

            peaks = get_peaks(onset_pred, 3)  # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
            onset_pred[~peaks] = 0

            unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
            unaligned_frames = (data['unaligned_label'] >= 2).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()

            # We align based on likelihoods regardless of the octave (chroma features)
            onset_pred_comp = compress_across_octave(onset_pred_np[:, -N_KEYS:])
            onset_label_comp = compress_across_octave(unaligned_onsets[:, -N_KEYS:])
            # We can do DTW on super-frames since anyway we search for local max afterwards
            onset_pred_comp = compress_time(onset_pred_comp, DTW_FACTOR)
            onset_label_comp = compress_time(onset_label_comp, DTW_FACTOR)
            print('dtw lengths', len(onset_pred_comp), len(onset_label_comp))
            init_time = time.time()
            alignment = dtw(onset_pred_comp, onset_label_comp, dist_method='euclidean',
                            )
            finish_time = time.time()
            print('DTW took {} seconds.'.format(finish_time - init_time))
            index1, index2 = alignment.index1, alignment.index2
            matches1, matches2 = get_matches(index1, index2), get_matches(index2, index1)

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_offsets = np.zeros(onset_pred_np.shape, dtype=bool)

            # We go over onsets (t, f) in the unaligned midi. For each onset, we find its approximate time based on DTW,
            # then find its precise time with likelihood local max
            for t, f in zip(*unaligned_onsets.nonzero()):
                t_comp = t // DTW_FACTOR
                t_src = matches2[t_comp]
                t_sources = list(range(DTW_FACTOR * min(t_src), DTW_FACTOR * max(t_src) + 1))
                # we extend the search area of local max to be ~0.5 second:
                t_sources_extended = get_margin(t_sources, len(aligned_onsets))
                # eliminate occupied positions. Allow only a single onset per 5 frames:
                existing_eliminated = [t_source for t_source in t_sources_extended if (aligned_onsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(existing_eliminated) > 0:
                    t_sources_extended = existing_eliminated

                t_src = max(t_sources_extended, key=lambda x: onset_pred_np[x, f])  # t_src is the most likely time in the local neighborhood for this note onset
                f_pitch = (len(self.instruments) * N_KEYS) + (f % N_KEYS)
                aligned_onsets[t_src, f] = 1  # set the label
                # Now we need to decide note duration and offset time. Find note length in unaligned midi:
                t_off = t
                while t_off < len(unaligned_frames) and unaligned_frames[t_off, f]:
                    t_off += 1
                note_len = t_off - t  # this is the note length in the unaligned midi. We need note length in the audio.

                # option 1: use mapping, traverse note length in the unaligned midi, and then use the reverse mapping:
                try:
                    t_off_src1 = max(matches2[(DTW_FACTOR * max(matches1[t_src // DTW_FACTOR]) + note_len) // DTW_FACTOR]) * DTW_FACTOR
                    t_off_src1 = max(t_src + 1, t_off_src1)
                except Exception as e:
                    t_off_src1 = len(aligned_offsets)
                # option 2: use relative note length
                t_off_src2 = t_src + int(note_len * (len(aligned_onsets) / len(unaligned_onsets)))
                t_off_src2 = min(len(aligned_onsets), t_off_src2)

                t_off_src = t_off_src2  # we choose option 2
                aligned_frames[t_src: t_off_src, f] = 1

                if t_off_src < len(aligned_offsets):
                    aligned_offsets[t_off_src, f] = 1

            # eliminate instruments that do not exist in the unaligned midi
            inactive_instruments, active_instruments_list = get_inactive_instruments(unaligned_onsets, len(aligned_onsets))
            onset_pred_np[inactive_instruments] = 0

            onset_label = aligned_onsets
            frame_label = aligned_frames
            offset_label = aligned_offsets
            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            if to_save is not None:
                inst_only = len(self.instruments) * N_KEYS
                time_now = datetime.now().strftime('%y%m%d-%H%M%S')
                frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_' + time_now + '.mid',
                            aligned_onsets[:, : inst_only], aligned_frames[:, : inst_only],
                            64. * aligned_onsets[:, : inst_only],
                            inst_mapping=self.instruments)
                frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_pitch_' + time_now + '.mid',
                                  aligned_onsets[:, -N_KEYS:], aligned_frames[:, -N_KEYS:],
                                  64. * aligned_onsets[:, -N_KEYS:])
                predicted_onsets = onset_pred_np >= 0.5
                predicted_frames = frame_pred_np >= 0.5
                frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_' + time_now + '.mid',
                            predicted_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                            64. * predicted_onsets[:, : inst_only],
                            inst_mapping=self.instruments)
                frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_pitch_' + time_now + '.mid',
                                  predicted_onsets[:, -N_KEYS:], predicted_frames[:, -N_KEYS:],
                                  64. * predicted_onsets[:, -N_KEYS:])
                if len(self.instruments) > 1:
                    max_pred_onsets = max_inst(onset_pred_np)
                    frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_max_' + time_now + '.mid',
                                max_pred_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                                64. * max_pred_onsets[:, : inst_only],
                                inst_mapping=self.instruments)
            if update:
                data['label'] = torch.from_numpy(label).byte()

            velocity_pred = velocity_pred.detach().squeeze().cpu()
            # velocity_pred = torch.from_numpy(new_vels)
            velocity_pred = (128. * velocity_pred)
            velocity_pred[velocity_pred < 0.] = 0.
            velocity_pred[velocity_pred > 127.] = 127.
            velocity_pred = velocity_pred.byte()
            if update:
                data['velocity'] = velocity_pred

            del audio_inp
            try:
                del mel
            except:
                pass
            del onset_pred
            del offset_pred
            del frame_pred
            del velocity_pred
            torch.cuda.empty_cache()
