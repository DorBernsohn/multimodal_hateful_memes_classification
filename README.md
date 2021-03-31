# multimodal_hateful_memes_classification
multimodal deep learning model to detect hateful memes
#### The metrics are reported on the Validation set.
| Model | Loss | AUC | Accuracy | Test/Train ratio  |
| --- | --- | --- | ---  | --- |
| concatBERT - VGG16 | 0.6243 | 0.6473 | 0.6600 | 0.2 |
| concatBERT - VGG19 | 0.6497 | 0.6467 | 0.6641 | 0.2 |

Requirements:

- emoji -qq
- torch==1.4.0. -qq (in colab)
- tensorflow==2.2 -qq
- transformers==3.5.0 -qq
