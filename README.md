## Douglas

## Environment
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
python3 vonnegut.py
python3 gp2.py
~~~

vonnegut.py is the uni-LSTM and gpt2.py is the fine-tuning for GPT-2

## Project Structure
~/data: the training data

~/docs: project milestone reports and other documentation

~/output: program output includes preprocessed data and written stories

~/src: the neural model

## Notes
Cool sci-fi autocomplete program: https://www.robinsloan.com/notes/writing-with-the-machine/

## Citations:
- Kaggle: Beginner's Guide to Text Generation with Pytorch
(https://www.kaggle.com/ab971631/beginners-guide-to-text-generation-pytorch)

- Parallel CPU Data Generation and GPU Usage
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

- Computing Perplexity with HuggingFace Transformers Library
https://huggingface.co/transformers/perplexity.html

- Fine-tuning GPT-2 Model
https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter9/Chapter9.5.ipynb


