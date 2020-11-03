import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten

class VggBertModel(tf.keras.Model):

    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       image_weights_name: str = 'imagenet',
                       seq_len: int = 512) -> None:

        super(VggBertModel, self).__init__()
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False

        self.vgg = VGG19(weights=image_weights_name, include_top=False)
        self.vgg.trainable = False

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

    def call(self, inputs):
        text_inputs = inputs[:3]
        img_inputs = inputs[-1]
        _, text = self.text_model_layer(text_inputs)

        text = self.dense_text1(text)
        text = self.dense_text2(text)
        text = self.dropout(text)
        
        img = img_inputs
        concat = self.concat([text, img])
        concat = self.global_dense1(concat)
        concat = self.dropout(concat)
        concat = self.global_dense2(concat)
        return self.global_dense3(concat)


class VggBertClassifier:

    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       seq_len: int = 100) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.seq_len = seq_len
        
    
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
        model = VggBertModel()
        model.compile(loss='binary_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy']) # optimizers.RMSprop(lr=1e-4)

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
        # model.summary()
        model.save_weights('./model/VggBertModel')