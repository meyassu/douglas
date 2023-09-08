## Douglas

## Table of Contents
- [Overview](#overview)
- [Repository Contents](#repository-contents)
- [Background](#background)
- [Instructions](#instructions)

## Overview
Large language models (LLMs) have taken the world by storm in recent years but they can be too computationally intensive to run locally with a CPU alone. The purpose of this repository is to allow those without significant compute resources to comfortably play with language models on their own machine. Transformer-based models now dominate NLP and such a mdoel is included here (fine-tuned GPT-2) but, to give users a glimpse into the past, a non-transformer-based model is also included (uni-LSTM n-gram model).

These language models were trained to produce science-fiction text in the somewhat erratic and haphazard style of authors like Adams, Vonnegut, Colfer, and others. The rationale for this is that small language models with tiny context windows would probably struggle to reproduce complex plot structures with long threads of recurring ideas running through them (a-la Wells, Asimov, Bradbury) and would have an easier time writing something like The Hitchiker's Guide to the Galaxy, which reads like a fever dream. (At the same time, text from more conventional sci-fi books was included to give the LMs some idea of proper structure and story progression).

The dataset is admittedly small and should be expanded. Currently, it consists of the following books: Life, the Universe, and Everything (Adams), So Long, and Thanks for All the Fish (Adams), The Hitchiker's Guide to the Galaxy (Adams), The Restaurant at the End of the Universe (Adams), andThe Time Machine (Wells). And Another Thing (Colfer) was the held-out test sample.

## Background
NLG is currently performed with encoder-decoder sequence-to-sequence models equipped with attention capabilities. Usually, fine-tuning, a form of transfer learning is performed to bias the model towards a specific region of its phase space during inference.

## Repository Contents
- data/: the training data

- docs/: project milestone reports and other documentation

- output/ : program output includes preprocessed data and written stories

- src/: the neural model

## Notes
This project initially also included a bidirectional LSTM (bi-LSTM) which traverses tokens in both forwards / backwards direction during training, but it surprisingly failed to perform as well as the uni-LSTM. 

Also, GPT-2 excelled in terms of grammar and semantics but fell short in its ability to produce novel or relevant sentences. This is likely because the additional training set had minimal impact on the existing parameters of the network (likely due to low learning rate distributed over entire network). Next time, maybe it would be better to freeze the early layers of GPT-2 and then increase the learning rate to a normal value for the later layers to nudge the model towards writing more relevant / creative sentences. 

(Cool sci-fi autocomplete program: https://www.robinsloan.com/notes/writing-with-the-machine/)

## Instructions
The project was built within an Anaconda environment. Type this to collect the dependencies:

~~~
$ conda env create --file=environment.yml
$ conda activate voltaire
~~~

Then, build the FastText module:
~~~
$ cd data
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ sudo pip install .
$ # or :
$ sudo python setup.py install
~~~
Verify the installation went well:
~~~
$ python
Python 2.7.15 |(default, May  1 2018, 18:37:05)
Type "help", "copyright", "credits" or "license" for more information.
>>> import fasttext
>>>
~~~
A clean return indicates a clean installation. Now, download the embeddings with this command within /fastText:
~~~
./download_model.py en
~~~
These and other instructions can be found at: https://fasttext.cc/docs/en/crawl-vectors.html

To run the code, type:
~~~
python3 douglas.py
python3 gp2.py
~~~

douglas.py is the uni-LSTM and gpt2.py is the fine-tuning for GPT-2

