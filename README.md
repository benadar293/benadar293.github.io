# benadar293.github.io
Based on the paper ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org).

We provide here our improved labels for the [MusicNet dataset](https://arxiv.org/abs/1611.09827) (the original dataset can be found [here](https://www.kaggle.com/imsparsh/musicnet-dataset)). 

The labels are in the form of MIDI files, currently pitch-only. Pitch-with-instrument will follow. In the meantime, we provide predictions of pitch-with-instrument on the MusicNet test set. The labels/predictions were generated automatically by the EM process described in our paper ["Unalgined Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org). 

You can train from scratch the architecture from the [MAESTRO paper](https://arxiv.org/abs/1810.12247) on MusicNet recordings with our labels and reach (without pitch-shift augmentation): 

MAPS test set: N note-level F1 and M frame-level F1 

MAESTRO test set: N note-level F1 and M frame-level F1 

Guitar-Set (entire dataset used as test): N note-level F1 and M frame-level F1. 

To reproduce the results from the paper, the EM process, as described in the [paper](https://link-url-here.org) is required.
