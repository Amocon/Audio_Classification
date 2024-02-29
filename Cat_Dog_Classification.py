import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(1,128,331,1)),
        keras.layers.Conv2D(16, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(2, activation=keras.activations.softmax)
        ])

def getMelSpecLabel(audiofile:tuple):
    y, sr = librosa.load(audiofile[0])
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                        fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    return (S_dB, audiofile[1])

def plotMel(spectogram:tuple):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectogram[0], x_axis='time',
                            y_axis='mel', 
                            fmax=8000, ax=ax)
    ax.set(title=spectogram[1])
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

def pad_resize(spectogram: np.ndarray, width:int):
    counter = 0
    if spectogram.shape[1] > width:
      return spectogram[:, :width]
    else:
      while len(spectogram[0]) < width:
        # Create the new column
        new_column = spectogram[:, counter]
        spectogram = np.insert(spectogram, spectogram[0].size, new_column, axis=1)
        counter += 1  
    return spectogram

def get_audiofiles(dir:str):
    audiofiles = []
    for root, dirs, files in os.walk(train_dir, topdown=False):
        for name in files:
            data = (os.path.join(root, name),name.split("_")[0])
            audiofiles.append(data)
    np.random.shuffle(audiofiles)
    return audiofiles

def preprocess_data(audiofiles:np.ndarray):
    sameLengthMel = []
    label = []
    for audiofile in audiofiles:
        processedSpectogram = getMelSpecLabel(audiofile=audiofile)
        resizedSpectogram = pad_resize(processedSpectogram[0], 331)
        sameLengthMel.append(resizedSpectogram.reshape((128,331,1)))
        label.append(processedSpectogram[1])
    features = np.array(sameLengthMel)
    labels =np.array(label)
    return features, labels

def init_model(features:np.ndarray):
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
   
def predict(path:str):
    predict = getMelSpecLabel(("cats_dogs\cat_1.wav", 1))
    res_predict = pad_resize(predict[0], 331)
    res_predict = res_predict.reshape((1,128,331,1))
    return model.predict(x=res_predict)