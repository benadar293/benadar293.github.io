# benadar293.github.io
Based on the paper ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org).

We provide here our improved labels for the [MusicNet dataset](https://arxiv.org/abs/1611.09827) (the original dataset can be found [here](https://www.kaggle.com/imsparsh/musicnet-dataset)). 

| test set | note F1 | note-with-inst. F1 | frame F1 | note-with-offset F1 |
| --- | :-: | :-: | :-: | :-: |
| **Hawthorne et al., 2019** | 82.0 | 82.0 |69.1 | 37.7 |
| **Kong et al., 2021** | 85.0 | 85.0 |65.2 | 31.9 |
| **Gardner et al., 2021** | 72.8 | - | 68.4 | 30.7 |
| **MusicNetEM** | 91.4 | 88.1 | 82.5 | 71.4 |
| **MusicNetEM wind** | 88.5 | 79.9 | 83.1 | 65.0 |
| **MusicNetEM strings** | 89.1 | 85.5 | 82.6 | 77.7 |
| **MusicNetEM strings** * | 85.9 | 81.1 | 79.0 | 75.1 |

The labels are in the form of MIDI files aligned with the audio, and include instrument information. Onset timing accuracy of the labels is 32ms, which is sufficient to train a transcriber. Onset timings in the original MusicNet labels are not accurate enough for this.

The labels were generated automatically by an EM process similar to the one described in our paper ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org). We improved the alignment algorithm, and in order to get more accurate labels, we divided the datast into three groups, based on the ensembles: piano solo, strings, and wind. We performed the EM process on each group separately.

You can train from scratch the architecture from the [MAESTRO paper](https://arxiv.org/abs/1810.12247) on MusicNet recordings with our labels.

For note-with-instrument transcription, use N_KEYS * (N_INSTRUMENTS + 1) onset classes, one for each note/instrument combination, and additional N_KEYS for pitch indepenedent of instrument. 
We used 88 * 12 = 1056 classes. Training in this manner for 101K steps of batch size 8, without any augmentation, we've reached: 


| test set | note F1 | note-with-inst. F1 | frame F1 | note-with-offset F1 |
| --- | :-: | :-: | :-: | :-: |
| **MAPS** | 82.0| 82.0 |69.1 | 37.7 |
| **MAESTRO** | 85.0 | 85.0 |65.2 | 31.9 |
| **GuitarSet** | 72.8 | - | 68.4 | 30.7 |
| **MusicNetEM** | 91.4 | 88.1 | 82.5 | 71.4 |
| **MusicNetEM wind** | 88.5 | 79.9 | 83.1 | 65.0 |
| **MusicNetEM strings** | 89.1 | 85.5 | 82.6 | 77.7 |
| **MusicNetEM strings** * | 85.9 | 81.1 | 79.0 | 75.1 |

| test instrument | note-with-inst. F1 |
| --- | :-: |
| **Violin** | 87.3 |
| **Viola** | 61.1 |
| **Cello** | 79.9 |
|**Bassoon** | 78.0 |
|**Clarinet** | 86.8 |
| **Horn** | 75.0 |

| `git status` | List all *new or modified* files |

| `git diff` | Show file differences that **haven't been** staged |
| MusicNetEM test | Show file differences that **haven't been** staged |


To reproduce the results from the paper, the EM process, as described in the [paper](https://link-url-here.org), including synthetic data pre-training and pitch shift augmentation, is required.
