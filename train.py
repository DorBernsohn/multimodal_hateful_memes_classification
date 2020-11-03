  
# https://github.com/AmbiTyga/MemSem/blob/master/model.py

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG19
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten



class Classifier:

    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       image_weights_name: str = 'imagenet',
                       seq_len: int = 100) -> None:
                       
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False
        self.seq_len = seq_len
        
        self.vgg = VGG19(weights=image_weights_name, include_top=False)
        self.vgg.trainable = False
    
    def encode(self, texts):
        input_id = []
        token_type_id = []
        attention_mask = []
        for text in tqdm(texts):
            dictIn = self.tokenizer.encode_plus(text, max_length=self.seq_len, pad_to_max_length=True)
            input_id.append(dictIn['input_ids'])
            token_type_id.append(dictIn['token_type_ids'])
            attention_mask.append(dictIn['attention_mask'])
        return np.array(input_id), np.array(token_type_id), np.array(attention_mask)
    

    def build(self):
        input_id = Input(shape=(self.seq_len,), dtype=tf.int64)
        mask_id = Input(shape=(self.seq_len,), dtype=tf.int64)
        seg_id = Input(shape=(self.seq_len,), dtype=tf.int64)

        _, layer_out = self.text_model_layer([input_id, mask_id, seg_id])
        dense = Dense(768, activation='relu')(layer_out)
        dense = Dense(256, activation='relu')(dense)
        txt_repr = Dropout(0.4)(dense)
        ################################################
        img_in = Input(shape=(224, 224, 3))
        img_out = self.vgg(img_in)
        flat = Flatten()(img_out)
        dense = Dense(2742, activation='relu')(flat)
        dense = Dense(256, activation='relu')(dense)
        img_repr = Dropout(0.4)(dense)
        concat = Concatenate(axis=1)([img_repr, txt_repr])
        dense = Dense(64, activation='relu')(concat)
        out = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[input_id, mask_id, seg_id, img_in], outputs=out)

        model.compile(loss='binary_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy'])

        # plot_model(model)
        model.summary()

        return model

    def train(self, data, validation_split=0.2):

        model = self.build()
        input_id, token_type_id, attention_mask = self.encode(data['texts'])
        image_data = data['image']
        labels = np.asarray(data['label']).astype('float32').reshape((-1,1))
        self.history = model.fit([input_id, token_type_id, attention_mask, image_data],
                                 labels,
                                 validation_split=validation_split,
                                 batch_size=32,
                                 epochs=10)

        model.save_weights('./model/MemSem')