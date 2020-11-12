import re
import emoji
import string
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

class Preprocess():
    """preprocess the inputs for concatBERT model
    """    
    def __init__(self, df: pd.DataFrame, 
                       data_dir: string, 
                       max_objects: int = 4,
                       image_embeddings: bool = True,
                       image_embeddings_size: int = 512,
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False, pooling="avg"),
                       detector = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1").signatures['default']) -> None:
        self.df = df
        self.data_dir = data_dir
        self.data = {"image": [], "filepath": [], "text": [], "label": []}

        self.vision_model = vision_model
        self.detector = detector
        self.image_embeddings = image_embeddings
        self.image_embeddings_size = image_embeddings_size
        self.max_objects = max_objects

    def preprocess(self) -> None:

        images = []
        texts = []
        for i ,(file_path, text) in tqdm(enumerate(zip(self.df.img, self.df.text))):

            img = self.preprocess_image(self.data_dir + file_path)
            im_width, im_height = Image.fromarray(np.uint8(img)).convert("RGB").size

            converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
            converted_img = tf.image.resize(converted_img, [224,224], method='nearest')
            result = self.detector(converted_img)
            result = {key:value.numpy() for key,value in result.items()}


            if len(result["detection_boxes"]) < self.max_objects: self.max_objects = len(result["detection_boxes"])
            embedding_list = []
            for i in range(self.max_objects):
                ymin, xmin, ymax, xmax = tuple(result["detection_boxes"][i])
                (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

                new_img = tf.image.crop_to_bounding_box(
                            img, int(yminn), int(xminn), int(ymaxx - yminn), int(xmaxx - xminn))
                new_img = tf.image.resize(new_img, [224,224], method='nearest')
                embedding_list.append(self.vision_model.predict(preprocess_input(np.expand_dims(new_img, axis=0)))[0])
                if len(embedding_list) < self.max_objects:
                    embedding_list.append(np.array([0]*(self.max_objects - len(embedding_list))*512))
            images.append(tf.cast(np.concatenate(embedding_list), tf.float32))
            texts.append(self.preprocess_text(text))
        images = np.concatenate(images)
        self.data["image"] = tf.cast(images.reshape(self.df.shape[0], self.max_objects*self.image_embeddings_size*2), tf.float32)

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