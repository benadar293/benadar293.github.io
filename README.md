# benadar293.github.io
Based on the paper ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org).

We provide here our improved labels for the [MusicNet dataset](https://arxiv.org/abs/1611.09827) (the original dataset can be found [here](https://www.kaggle.com/imsparsh/musicnet-dataset)). 

The labels are in the form of MIDI files, currently pitch-only. Pitch-with-instrument will follow. In the meantime, we provide predictions of pitch-with-instrument on the MusicNet test set. The labels/predictions were generated automatically by the EM process described in our paper ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org). 

You can train from scratch the architecture from the [MAESTRO paper](https://arxiv.org/abs/1810.12247) on MusicNet recordings with our labels and reach (without pitch-shift augmentation): 

MAPS test set: N note-level F1 and M frame-level F1 

MAESTRO test set: N note-level F1 and M frame-level F1 

Guitar-Set (entire dataset used as test): N note-level F1 and M frame-level F1. 

MusicNetEM: 91.4 note-level F1, 88.1 note-with-instrument F1, and 82.5 frame-level F1 

| test set | note F1 | note-with-instrument F1 | frame F1 | note-with-offset F1 |
| --- | --- | --- | --- | --- |
| **MusicNetEM** | 91.4 | 88.1 | 82.5 | 71.4 |
| **MusicNetEM wind** | 88.5 | 79.9 | 83.1 | 65.0 |
| **MusicNetEM strings** | 89.1 | 85.5 | 82.6 | 77.7 |
| **MusicNetEM strings** * | 85.9 | 81.1 | 79.0 | 75.1 |

| test instrument | note-with-instrument F1 |
| --- | --- |
| **Violin** | 87.3 |
| **Viola** | 61.1 |
| **Cello** | 79.9 |

| `git status` | List all *new or modified* files |

| `git diff` | Show file differences that **haven't been** staged |
| MusicNetEM test | Show file differences that **haven't been** staged |


To reproduce the results from the paper, the EM process, as described in the [paper](https://link-url-here.org), is required.
