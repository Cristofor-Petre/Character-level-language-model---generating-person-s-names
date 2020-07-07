import keras
import numpy as np
from keras.utils import to_categorical



class DataGenerator(keras.utils.Sequence):

  def __init__(self, encoder_input, decoder_input, decoder_output, batch_size,
               num_classes, shuffle=True):
    self.encoder_input = encoder_input
    self.decoder_input = decoder_input
    self.decoder_output = decoder_output
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.shuffle = shuffle
    self.on_epoch_end()

  
  def __len__(self):
    return int(np.floor(len(self.encoder_input) / self.batch_size))
  
  def __getitem__(self, index):

    indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
    
    enc_input = self.encoder_input[indexes]
    dec_input = self.decoder_input[indexes]
    dec_output = self.decoder_output[indexes]
    
    inp = []
    inp.append(enc_input)
    inp.append(to_categorical(dec_input, num_classes=self.num_classes))
    #d_i = np.expand_dims(dec_input, axis=-1)
    #inp.append(d_i)
    
    return inp, to_categorical(dec_output, num_classes=self.num_classes)
  
  def on_epoch_end(self):

    self.indexes = np.arange(len(self.encoder_input))
    if self.shuffle:
      np.random.shuffle(self.indexes)