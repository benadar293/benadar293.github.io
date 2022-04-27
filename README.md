Ben Maman and Amit Bermano, ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://benadar293.github.io/)

Based on the paper ["Unaligned Supervision for Automatic Music Transcription in The Wild"](https://link-url-here.org).

![alt text](teaser.png "Title")

We provide here example transcriptions done by our system of famous pieces and songs, toegether with quantitative results on various benchmarks. 

We also provide here our [improved labels]("musicnet_em.zip") for the [MusicNet dataset](https://arxiv.org/abs/1611.09827) (the original dataset can be found [here](https://www.kaggle.com/imsparsh/musicnet-dataset)). The labels were generated automatically by our method. We refer to MusicNet recordings with our labels as MusicNetEM. We provide a baseline for training from scratch on MusicNetEM, including cross-dataset evaluation. The labels are in the form of MIDI files aligned with the audio, and include instrument information. Onset timing accuracy of the labels is 32ms, which is sufficient to train a transcriber. Onset timings in the original MusicNet labels are not accurate enough for this.

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


To reproduce the results from the paper, the EM process, as described in the [paper](https://link-url-here.org), including synthetic data pre-training and pitch shift augmentation, is required.

Links to performances excerpts of which we used for demonstration:

[Myung-Whun Chung & Orchestre Philharmonique de Radio France, Bizet Carmen Overture](https://www.youtube.com/watch?v=jL-Csf1pNCI&ab_channel=FranceMusique)

[Mozart, Eine Kleine Nachtmusik](https://www.youtube.com/watch?v=oy2zDJPIgwc&ab_channel=AllClassicalMusic)

[Julien Salemkour, Mozart Symphony NO. 40](https://www.youtube.com/watch?v=wqkXqpQMk2k&ab_channel=EuroArtsChannel)

[Trevor Pinnock, Bach Harpsichord Concerto No. 1](https://www.youtube.com/watch?v=R66fz9yxzAk&ab_channel=SoliDeoGloria8550)

[Frank Monster, Bach Inverntion No. 8](https://www.youtube.com/watch?v=whbFffxr2q4&ab_channel=NetherlandsBachSociety)

[John Williams & Boston Pops Orchestra, Indiana Jones Theme Song](https://www.youtube.com/watch?v=-bTpp8PQSog&ab_channel=Vyrium)

[Hungarian Symphony Orchestra, Brahms Hungarian Dance NO. 5](https://www.youtube.com/watch?v=Nzo3atXtm54&ab_channel=MelosKonzerte)

[Rossini Barber of Seville Overture](https://www.youtube.com/watch?v=OloXRhesab0&t=2s&ab_channel=ClassicalMusicOnly)

[Nino Gvetadze, Brahms Piano Concerto No. 2](https://www.youtube.com/watch?v=YzZy1is6ZRU&ab_channel=Levan)

[Arturo Benedetti Michelangeli, Carlo Maria Giulini, & Wiener Symphoniker, Beethoven Piano Concerto No. 5](https://www.youtube.com/watch?v=TahrEIVu4nQ&ab_channel=pianoconc2)

[United States Marine Band, John Philip Sousa's "The Stars and Stripes Forever"](https://www.youtube.com/watch?v=a-7XWhyvIpE&ab_channel=UnitedStatesMarineBand)

[Desireless, Voyage](https://www.youtube.com/watch?v=NlgmH5q9uNk&ab_channel=Desireless)

[Maddona, La Isla Bonita](https://www.youtube.com/watch?v=zpzdgmqIHOQ&ab_channel=Madonna)

[Toto, Africa](https://www.youtube.com/watch?v=FTQbiNvZqaY&ab_channel=TotoVEVO)

[ABBA, Gimme](https://www.youtube.com/watch?v=JWay7CDEyAI&ab_channel=CraigGagn%C3%A9)

[Dick Dale and the Del Tones, Misirlou (Pulp Fiction Soundtrack)](https://www.youtube.com/watch?v=1hLIXrlpRe8)


