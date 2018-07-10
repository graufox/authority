# authority
##### Train a text classifier to recognize authors in just a few lines of Python. 

This Python library is used to train a classifier that learns to recognize different authors' writings. The model is built with [minimaxir](https://github.com/minimaxir)'s [textgenrnn](https://github.com/minimaxir/textgenrnn) and the [Keras](https://keras.io) deep learning library. 

## Usage
 1. Clone repository and use [Pipenv](https://docs.pipenv.org/) to install dependencies. 

 2. Import authority's Highlighter class.

        from authority import Highlighter
        hi = Highlighter()

 3. Load the sample authors found in [Myles O'Neill](https://www.kaggle.com/mylesoneill)'s [Classic Literature in ASCII](https://www.kaggle.com/mylesoneill/classic-literature-in-ascii) (or any other "authors"/text types) by pointing to the folder in which it is located.

        hi.load_author_texts('AUTHORS/')

 4. Load any pretrained model and/or train and save. This package includes weights that have been pretrained on the authors in Classic Literature in ASCII. 

        hi.load_model('model_data/classifier_model.h5')
        hi.train(epochs=10, validation_split=0.15)
        hi.save_model('model_data/classifier_model_v2.h5')

 5. Predict the author of a given text (make sure it's at least 40 characters):

        text = 'Hey look at me I\'m some words here, make sure it\'s long enough'
        hi.classify(text)
        >> 'TWAIN'

    Or take a closer look at the probabilities of each:

        hi.highlight(text).mean(axis=0)
        >> array([0.01029149, 0.01112002, 0.15972467, 0.12881784, 0.02912773,
                  0.01518796, 0.01132576, 0.06573769, 0.00117596, 0.02139501,
                  0.00711596, 0.00492442, 0.04230085, 0.07562025, 0.16566265,
                  0.2081482 , 0.04232353], dtype=float32)

    You can also plot out the intensity of each author over time: 

        hi.plot_highlights(text)
        
    # authority
##### Train a text classifier to recognize authors in just a few lines of Python. 

This Python library is used to train a classifier that learns to recognize different authors' writings. The model is built with [minimaxir](https://github.com/minimaxir)'s [textgenrnn](https://github.com/minimaxir/textgenrnn) and the [Keras](https://keras.io) deep learning library. 

## Usage
 1. Clone repository and use [Pipenv](https://docs.pipenv.org/) to install dependencies. 

 2. Import authority's Highlighter class.

        from authority import Highlighter
        hi = Highlighter()

 3. Load the sample authors found in [Myles O'Neill](https://www.kaggle.com/mylesoneill)'s [Classic Literature in ASCII](https://www.kaggle.com/mylesoneill/classic-literature-in-ascii) (or any other "authors"/text types) by pointing to the folder in which it is located.

        hi.load_author_texts('AUTHORS/')

 4. Load any pretrained model and/or train and save. This package includes weights that have been pretrained on the authors in Classic Literature in ASCII. 

        hi.load_model('model_data/classifier_model.h5')
        hi.train(epochs=10, validation_split=0.15)
        hi.save_model('model_data/classifier_model_v2.h5')

 5. Predict the author of a given text (make sure it's at least 40 characters):

        text = 'Hey look at me I\'m some words here, make sure it\'s long enough'
        hi.classify(text)
        >> 'TWAIN'

    Or take a closer look at the probabilities of each:

        hi.highlight(text).mean(axis=0)
        >> array([0.01029149, 0.01112002, 0.15972467, 0.12881784, 0.02912773,
                  0.01518796, 0.01132576, 0.06573769, 0.00117596, 0.02139501,
                  0.00711596, 0.00492442, 0.04230085, 0.07562025, 0.16566265,
                  0.2081482 , 0.04232353], dtype=float32)

    You can also plot out the intensity of each author over time: 

        hi.plot_highlights(longer_text)
  
    ![img](https://github.com/graufox/authority/blob/master/plot_highlights_demo_image.png)
