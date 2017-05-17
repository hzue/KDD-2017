from keras.layers import Input, Dense
import numpy as np
from keras.models import Model
import keras

def generate_autoencoder(vectors, name):

  print(vectors)
  encoding_dim = 2
  input_dim = len(vectors[0])
  input_vector = Input(shape=(input_dim,))

  # define layer
  encoded = Dense(encoding_dim, activation='relu')(input_vector)
  decoded = Dense(input_dim, activation='sigmoid')(encoded)

  # overall training struture
  autoencoder = Model(input_vector, decoded)

  # encode
  encoder = Model(input_vector, encoded)

  # decode
  encoded_input = Input(shape=(encoding_dim,))
  decoder_layer = autoencoder.layers[-1]
  decoder = Model(encoded_input, decoder_layer(encoded_input))

  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  autoencoder.fit(vectors, vectors, epochs=20000)
  print(encoder.predict(vectors))
  print(autoencoder.predict(vectors))
  encoder.save('./res/autoencoder_model/{}.encoder'.format(name))
  decoder.save('./res/autoencoder_model/{}.decoder'.format(name))
  autoencoder.save('./res/autoencoder_model/{}.autoencoder'.format(name))


if __name__ == '__main__':
  # input_vector = []
  # for i in range(7):
  #   a = [0 for _ in range(7)]
  #   a[i] = 1
  #   input_vector.append(a)
  # generate_autoencoder(input_vector, 'weekday')

  # input_vector = []
  # for i in range(6):
  #   a = [0 for _ in range(6)]
  #   a[i] = 1
  #   input_vector.append(a)
  # generate_autoencoder(input_vector, 'route')

  # input_vector = []
  # for i in range(24):
  #   a = [0 for _ in range(24)]
  #   a[i] = 1
  #   input_vector.append(a)
  # generate_autoencoder(input_vector, 'link')
  #

