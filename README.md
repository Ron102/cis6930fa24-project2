
  

  

  

# cis6930fa24 -- Project 2 --

  

  

  

**Name: Ronit Bali**

**UFID - 58455645**

  

  

  

# Assignment Description

  

  

  

This is my project 2 submission, for which I have created an _Unredactor_. The unredactor takes redacted documents and return the most likely candidates to fill in the redacted location. I have used the provided data set of the redacted names from the corpus called _unredactor.tsv_. The first column specifies whether the example is in `training` or `validation`. The second column contains the name of the entity that was redacted. The final column is the redaction context. Each of these examples comes from somewhere in the review dataset. The dataset format is as follows:
```
split				name				context

validation			Dan Duryea			██████████ played an awesome...
training			Mari Honjo			The Dragon Lady ██████████ who...
```
I have used the _training_ examples to train the model and then evaluated its performance using the _validation_ examples. Finally, a dataset called _test.tsv_ having columns `id` and `context` is given. I create a file called _submission.tsv_ which contains the id of the context as well as the predicted names given by my model. The redacted names are represented by the Unicode full block character █ (U+2588) for each letter of the name. For example, if the name is "Sam", the redacted name would be "███" (█ 3 times for each letter of the name). 
I have used classical machine learning for this classification task. The full review dataset is too large for my machine to handle, and the subset dataset has many labels which are not a part of the training dataset. This compromises the performance measures of the model, as it has to predict names it has never seen before. I have performed pre-processing (tokenization, stop words removal, feature extraction) before applying the Random Forest Classifier, which is the model I have used to balance performance with speed. 

# How to install and use

  

To install all the dependencies:

  

```

pipenv install

```

  

To activate the virtual environment:

  

```

pipenv shell

```

  
  

To run the program, run the _main.py_ program using:

  

```

pipenv run python main.py 

```


  
The performance metrics of the model are printed on the console, and a _submission.tsv_ file is generated in the same directory. I have provided the file in my repository.

## Program Design

- Libraries such as pandas, numpy, nltk and modules like sklearn are used for text preprocessing, feature extraction, model training and evaluation.
- First step is preprocessing, in which I convert the context text string into lowercase, followed by tokenization using nltk, stopwords removal, puntuation filtering and feature extraction related to the redacted names. 
- Next, we load the datasets. The unredactor.tsv dataset is loaded and split into training and validation sets. I have set a limit to the maximum number of samples to make the program run faster. The context text in both training and validation datasets is passed through the preprocessor function, along with the test.tsv data.
- I then use the TF-IDF vectorizer to extract n-gram features (unigram to trigram) from the text. I have used Standard Scalar to normalize the redaction features based on the max and mean redaction name lenghts. Lastly, I have combined the two sets of features using a Column Transformer.
- The model pipeline combines features transformation with a Random Forest Classifier. I have tried other algorithms and verified that in this case, the Random Forest Classifier works the best, providing somewhat accurate results in comparitively less time.
- The model is trained on the training dataset and predicts labels for the validation dataset. The model performance metrics (accuracy, precision, recall, f1 score) are printed out. 
- Finally, I predict the redacted names in the test.tsv dataset and write them to a new created file called submission.tsv against their respective IDs. 
  

## Functions

  

#### main.py

  

*main()* - This function gets the path of the datasets in the same directory, and calls the other functions to preprocess the data, fit it on the model, provide the performance metrics and output the _submission.tsv_ file.

  

_preprocessor()_ - This function preprocesses the dataset. The test is first converted to lowercase to make it easier to get tokenized. It filters out important words which are not stop words from a set of stop words. Then the function extracts relevant features by considering the length of the redacted names, which correspond to the length of the actual names to be predicted. Max and Mean redaction lengths are evaluated. 


*load_data()* - This function first imports the unredactor.tsv dataset, assigns column names to make things easier and then separates the dataset into training and validation datasets based on the "Type" column. Then the function performs sampling of the two datasets based on maximum samples and random state set to 42. The contexts of both datasets are traversed and sent to the _preprocessor()_ function. Finally the _test.tsv_ file is read and passed through the _preprocessor()_ function.


  *submission_output()* - This function saves the name predictions by the mode along with the IDs (as provided by the _test.tsv_ dataset) in a file _submission.tsv_ and saves it in the current directory.


*train_eval_model()* - This function combines TF-IDF features with the redaction features  in a transformer which is fed into a model pipeline using Random Forest Classifier. The data is then fit to the model and the performance metrics such as accuracy, precision, recall and f1 score are printed. 




## Bugs and Assumptions

  

Some bugs and assumptions can be encountered/should be kept in mind while executing the program:

  

- The program might take several minutes to execute based on the size of the dataset.
- The model does not perform that well due to resource and hardware constraints. If the model is trained on the full review dataset on a good machine, then the performance metrics would improve drastically.
- The submission.tsv has one less record (in my case) than the test.tsv as it might be skipping or dropping a row, probably due to malformed data.
- A specific version of numpy and python have been installed due to compatibility issues on my system, which are the versions being installed from the pipfile for this project.

- The program first downloads packages "punkt" and "stopwords" using nltk.download. Failure to download can cause the program to not execute/give errors.

- The performance metrics might differ on each execution of the program due to random sampling.