# TinyStories-tag-prediction
A lightweight LLM-based model using Hugging Face Transformers and PyTorch to predict narrative attributes such as “conflict” and “good ending” in the TinyStory dataset.

# Introduction
We choose Track 1 between those proposed: our task is to predict the tags of a story that belongs to Tinystories Dataset. Each example in the dataset consists of a story and its corresponding features. First of all we need to split the training set into training and validation sets. Then we define a function which creates a label vector which encodes the tags, i.e., the function generates a vector of length 6, where each position corresponds to one of the 6 possible tags. Each entry in the vector is set to 1 if the story belongs to that tag, and 0 if it does not. \\
It is important to point out that each story can have more than one tag, then we have a multi-label classification problem.\\
We use two different models in order to classify our stories: the first one is built by us, then it is trained just on a subset of our training dataset, the second one is a pretrained model that is trained and made available by Hugging Face in their Transformers library. \\
For each of these models we need to tokenize differently our stories and choose different parameters that we need for our specific language model. In the following we explain in detail the construction and the results of these two different models.\\
Leonardo Cittadini focused more on the custom model, Laura Cangiotti and Pavao Jancijev focused more on the pretrained model.
