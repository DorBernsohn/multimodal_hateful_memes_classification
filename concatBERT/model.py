import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten

from typing import Tuple

class VisionBertModel(tf.keras.Model):
    """create a model that concat BERT embeddings and other vision model (VGG, ResNet..)

    Args:
        tf (tf.keras.Model): tf.keras.Model
    """
    def __init__(self, seq_len: int = 100,
                       image_embeddings: bool = True,
                       text_model_name: str = 'bert-base-uncased',
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False)) -> None:

        super(VisionBertModel, self).__init__()
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False
        self.image_embeddings = image_embeddings

        self.vision_model = vision_model
        self.vision_model.trainable = False

        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.concat = Concatenate(axis=1)

        self.global_dense1 = Dense(128, activation='relu')
        self.global_dense2 = Dense(64, activation='relu')
        self.global_dense3 = Dense(1, activation='sigmoid')
        self.dense_text1 = Dense(768, activation='relu')
        self.dense_text2 = Dense(256, activation='relu')
        self.img_dense1 = Dense(2742, activation='relu')
        self.img_dense2 = Dense(256, activation='relu')
        self.img_dense3 = Dense(512, activation='relu')

    def call(self, inputs: list) -> tf.keras.layers:
        text_inputs = inputs[:3]
        img_inputs = inputs[-1]
        _, text = self.text_model_layer(text_inputs)

        text = self.dense_text1(text)
        text = self.dense_text2(text)
        text = self.dropout(text)
        
        img = img_inputs
        if not self.image_embeddings:
            img_out = self.vision_model(img)
            flatten = self.flatten(img_out)
            dense = self.img_dense1(flatten)
            dense = self.img_dense2(dense)
            img = self.dropout(dense)

        concat = self.concat([text, img])
        concat = self.global_dense1(concat)
        concat = self.dropout(concat)
        concat = self.global_dense2(concat)
        return self.global_dense3(concat)

class VisionBertClassifier:
    """build the model and run training
    """
    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       seq_len: int = 100,
                       image_embeddings: bool = True) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.seq_len = seq_len
        self.image_embeddings = image_embeddings
        
    def encode(self, texts: list) -> Tuple[np.array, np.array, np.array]:

        input_id = []
        token_type_id = []
        attention_mask = []
        for text in tqdm(texts):
            dictIn = self.tokenizer.encode_plus(text, max_length=self.seq_len, pad_to_max_length=True, truncation=True)
            input_id.append(dictIn['input_ids'])
            token_type_id.append(dictIn['token_type_ids'])
            attention_mask.append(dictIn['attention_mask'])
        return np.array(input_id), np.array(token_type_id), np.array(attention_mask)
    
    def build(self, vision_model: tf.keras.applications):
        
        METRICS = ['accuracy', 
                   tf.keras.metrics.AUC(), 
                   tf.keras.metrics.TruePositives(), 
                   tf.keras.metrics.TrueNegatives(), 
                   tf.keras.metrics.FalsePositives(), 
                   tf.keras.metrics.FalseNegatives()]

        model = VisionBertModel(seq_len=self.seq_len, image_embeddings=self.image_embeddings, vision_model=vision_model)
        model.compile(loss='binary_crossentropy', optimizer=Adam(2e-5), metrics=METRICS)

        return model

    def train(self, data: dict, 
                    vision_model: tf.keras.applications,
                    validation_split: float = 0.2) -> tf.keras.callbacks.History:

        model = self.build(vision_model)
        input_id, token_type_id, attention_mask = self.encode(data['texts'])
        image_data = data['image']
        labels = np.asarray(data['label']).astype('float32').reshape((-1,1))
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)
        self.history = model.fit([input_id, token_type_id, attention_mask, image_data],
                                 labels,
                                 validation_split=validation_split,
                                 batch_size=32,
                                 callbacks = [callback],
                                 epochs=100)

        return self.history