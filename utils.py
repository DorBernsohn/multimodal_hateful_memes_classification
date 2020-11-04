import re
import emoji
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from contextlib import suppress
from collections import Counter
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


def ZeroDivisionDecorator(func):

    def func_wrapper(*args, **kwargs):

        with suppress(ZeroDivisionError):
           return func(*args, **kwargs)
    return func_wrapper

class Metrics():
    
    def __init__(self, y_true: np.array, y_pred: np.array) -> None:
        counts = Counter(zip(y_pred, y_true))
        self.tp = counts[1, 1]
        self.fn = counts[1, 0]
        self.fp = counts[0, 1]

    @property
    @ZeroDivisionDecorator
    def recall(self) -> float:
        """calculate the recall for binary classification

        Returns:
            float: recall score
        """
        return self.tp / (self.tp + self.fn)

    @property
    @ZeroDivisionDecorator
    def precision(self) -> float:
        """calculate the precision for binary classification

        Returns:
            float: precision score
        """    
        return self.tp / (self.tp + self.fp)

    @property
    @ZeroDivisionDecorator
    def f1(self) -> float:
        """calculate the f1 score for binary classification

        Returns:
            float: f1 score
        """    
        p = self.precision()
        r = self.recall()
        if p and r:
            return 2 * ((p * r) / (p + r))


class Preprocess():
    def __init__(self, df: pd.DataFrame, 
                       data_dir: string, 
                       image_embeddings: bool = True,
                       image_embeddings_size: int = 512,
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False, pooling="avg")) -> None:
        self.df = df
        self.data_dir = data_dir
        self.data = {"image": [], "filepath": [], "text": [], "label": []}

        self.image_embeddings = image_embeddings
        self.image_embeddings_size = image_embeddings_size
        self.vision_model = vision_model

    def preprocess(self):

        images = []
        texts = []
        if self.image_embeddings:
            for i ,(file_path, text) in tqdm(enumerate(zip(self.df.img, self.df.text))):
                images.append(self.vision_model.predict(preprocess_input(np.expand_dims(self.preprocess_image(self.data_dir + file_path), axis=0)))[0])
                texts.append(self.preprocess_text(text))
            images = np.concatenate(images)
            self.data["image"] = tf.cast(images.reshape(self.df.shape[0], self.image_embeddings_size), tf.float32)
        else:
            for i ,(file_path, text) in tqdm(enumerate(zip(self.df.img, self.df.text))):
                images.append(preprocess_input(np.expand_dims(self.preprocess_image(self.data_dir + file_path), axis=0)))
                texts.append(self.preprocess_text(text))
            images = np.concatenate(images)
            self.data["image"] = tf.cast(images.reshape(self.df.shape[0], 224, 224, 3), tf.float32)
        self.data["texts"] = np.array(texts)
        self.data["filepath"] = self.df.img.values
        self.data["label"] = self.df.label.values

    def preprocess_image(self, filepath: string) -> tf.Tensor:

        image = tf.io.read_file(filename=filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224,224], method='nearest')

        return image

    def preprocess_text(self, text: string, remove_emojis=True, remove_numbers=True, remove_punc=True, remove_url=True, remove_spaces=True) -> string:
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