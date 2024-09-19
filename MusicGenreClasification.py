import os, sys
import librosa
import numpy as np
from keras.models import load_model
from joblib import load

sr = 22050 # Frecuencia de muestreo usada

# Revisamos si se tiene la cantidad de argumentos es la correcta.
if len(sys.argv) != 3:
    # error
    sys.exit('Argumentos no validos. Ej: python MusicGenreClasification.py [cnn|crnn] audio_path')

# Validamos que se soliciten los tipos de redes correctos.
if sys.argv[1].lower() == 'cnn': 
    model = load_model('models/CNN_MusicGenreClassificacion.keras')
elif sys.argv[1].lower() == 'crnn': 
    model = load_model('models/CRNN_MusicGenreClassificacion.keras')
else: 
    # error
    sys.exit('Modelo no valido. Solo se permiten formatos CNN y CRNN')

# Validamos si el archivo de audio existe y tiene algunas de las extensiones permitidas.
if os.path.isfile(sys.argv[2]):
    if sys.argv[2].endswith('.wav'):
        audio, _ = librosa.load(sys.argv[2], sr=sr, mono=True)
    elif sys.argv[2].endswith('.mp3'):
        audio = None
    else: 
        # error
        sys.exit('Formato de audio no valido. Solo se permiten formatos .wav y .mp3')
else:
    # error
        sys.exit('El archivo no existe.')


# Funcion definida para, dado un audio, extraer 7 de sus features relevantes x frame 
# como: MFCC (de 20), Chroma STFT, Spectral Centroid, Spectral Bandwidth,
# Spectral Contrast, Spectral Rollof, Zero Crossing Rate y RMS.
# Dando como salida un Array de 44 dimensiones conteniendo todos estos features.
def extract_features(y, sr=22050):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) # 20 dimensiones x frame
    chroma = librosa.feature.chroma_stft(y=y, sr=sr) # 12 dimensiones x frame
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr) # 1 dimension x frame
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr) # 1 dimension x frame
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr) # 7  dimensiones x frame
    spec_rollof = librosa.feature.spectral_rolloff(y=y, sr=sr) # 1 dimension x frame
    zcr = librosa.feature.zero_crossing_rate(y=y) # 1 dimension x frame
    rms = librosa.feature.rms(y=y) # 1 dimension x frame

    # Resultado Array de 44 dimensiones  x frame.
    return np.concatenate(
                    (
                        mfcc,
                        chroma,
                        spec_centroid,
                        spec_bandwidth,
                        spec_contrast,
                        spec_rollof,
                        zcr,
                        rms
                    )
                    , axis=0
                )
scaler = load('models\minmax_scaler.save') # MinMaxScaler guardado
# Se traducen los generos a numeros segun este disccionario
dict_generos = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
}


# Solo se consideraran los primeros 5seg del audio
# En caso de tener menos segundos, se rellena hasta llegar a 5seg
audio = audio[:sr*5]
padding = sr * 5 - len(audio)
audio = np.pad(audio, (0, padding), mode='constant')

# Extraemos las caractristicas del audio
audio_caract = extract_features(audio)

# Obtenemos el Genero de la cancion usando la Red entrenada .
genre = np.argmax(model.predict( scaler.transform(audio_caract.reshape(1, -1)).reshape(1, 44, 216) ), axis=1)[0]

# Mostramos en pantalla el genero predicho del audio.
print('El genero del audio suministrado es: ', dict_generos[genre])