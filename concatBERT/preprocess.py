import re
import emoji
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

class Preprocess():
    """preprocess the inputs for concatBERT model
    """    
    def __init__(self, df: pd.DataFrame, 
                       data_dir: string, 
                       image_embeddings_size: int = 512,
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False, pooling="avg")) -> None:
        self.df = df
        self.data_dir = data_dir
        self.data = {"image": [], "filepath": [], "text": [], "label": []}

        self.vision_model = vision_model
        self.vision_model.trainable = False
        self.image_embeddings_size = image_embeddings_size

    def preprocess(self) -> None:

        images = []
        texts = []
        for i ,(file_path, text) in tqdm(enumerate(zip(self.df.img, self.df.text))):
            images.append(self.vision_model.predict(preprocess_input(np.expand_dims(self.preprocess_image(self.data_dir + file_path), axis=0)))[0])
            texts.append(self.preprocess_text(text))
        images = np.concatenate(images)
        self.data["image"] = tf.cast(images.reshape(self.df.shape[0], self.image_embeddings_size), tf.float32)

        self.data["texts"] = np.array(texts)
        self.data["filepath"] = self.df.img.values
        self.data["label"] = self.df.label.values

    @staticmethod
    def preprocess_image(filepath: string) -> tf.Tensor:
        """perform decoding and resizing to an image

        Args:
            filepath (string): filepath of an image

        Returns:
            tf.Tensor: the image after decoding and resizing
        """        
        image = tf.io.read_file(filename=filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224,224], method='nearest')

        return image
    
    @staticmethod
    def preprocess_text(text: string, remove_emojis=True, remove_numbers=True, remove_punc=True, remove_url=True, remove_spaces=True) -> string:
            """Clean the text
            
            Arguments:
                text {string} -- the text we want to clean
            
            Keyword Arguments:
                remove_emojis {bool} -- remove emojis from our text (default: {True})
                remove_numbers {bool} -- remove numbers from our text (default: {True})
                remove_punc {bool} -- remove punctuation from our text (default: {True})
                remove_url {bool} -- remove url's from our text (default: {True})
                remove_spaces {bool} -- remove extra spaces from our text (default: {True})
            
            Returns:
                string -- the text after cleaning 
            """        

            url_re = re.compile("""((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?""")
            nl_re = re.compile(r'(\n+)')
            t_re = re.compile(r'(\t+)')
            numbers_re = re.compile(r'^\d+\s|\s\d+\s|\s\d+$')

            if type(text) != str:
                return str(text)
            else:
                if remove_spaces:
                    text = re.sub(nl_re, ' ', text)
                    text = re.sub(t_re, ' ', text)
                if remove_url:
                    text = re.sub(url_re, " ", text)
                if remove_punc:
                    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
                if remove_numbers:
                    text = re.sub(numbers_re, ' ', text)
                if remove_emojis:
                    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
                return text