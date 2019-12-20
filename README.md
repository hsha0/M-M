# CSCI 374 Final Project: Machine and Music (M&M)
Yuanzhe Liu, Kumo (Yiyun) Shao, Han Shao

This code implements multi-layer Recurrent Neural Network for training music generation
models. In other words the model takes MIDI files as input and trains a Recurrent Neural Network that
learns to predict the next message following the previous message sequence.

## Requirements
This code is written in Python and requires Tensorflow 1.15 and Mido libraries. You can
install Tensorflow 1.15 and Mido through

```bash
$ pip3 install tensorflow==1.15
$ pip3 install mido
```

## Usage
### Data 
All input MIDI files should be stored in an input directory. You'll notice that
there is an example dataset included in the repo (`dataset`) which consisted of 15 songs from performer
Jean-Selim Abdelmoula. 

### Training
Start training the model and generating music using `mm.py`. 

```bash
$ python3 mm.py --data_dir=DATA_DIR
```

Music generated after each 10 epochs will be stored in `results/DATA_DIR`.

**Checkpoints.** While the model is training, it will periodically save models to folder
`resuls/DATA_DIR/model` for each 10 epochs.

