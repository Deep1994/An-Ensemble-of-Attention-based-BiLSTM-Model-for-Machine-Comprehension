# An Ensemble of Attention-based BiLSTM Model for Machine Comprehension
An implementation for [SemEval 2018 Task 11: Machine Comprehension using Commonsense Knowledge](https://competitions.codalab.org/competitions/17184)

Our paper at SemEval 2018 Task 11 is available at [http://www.aclweb.org/anthology/S18-1174](http://www.aclweb.org/anthology/S18-1174)

![model](https://github.com/Deep1994/An-Ensemble-of-Attention-based-BiLSTM-Model-for-Machine-Comprehension/raw/master/img/model.png)

## Requirments

+ Python 3.5.2
+ [Keras](http://keras-cn.readthedocs.io/en/latest/) (deep learning library, verified on 2.0.9)
+ [NLTK](http://www.nltk.org/) (NLP tools, verified on 3.2.1)

## Datasets

+ Officially released data

	+ Officially released data is in xml format, we parse it into txt format. They can be downloaded from the above **data** file, including training set, development set and test set.

	+ The gold data can be downloaded from **gold_data** file, also in txt format.

+ Word embeddings:
	+ [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Pre-processing

+ **dataProcess.py**: Pre-processing training set and development set, including the replacement of abbreviated characters, the removal of meaningless characters, word segmentation, the removal of stop words in instances, creating vocabulary list, and vectorizing instance, question, and answer.

+ **testReader.py**: Pre-processing test set, almost the same as above.

## Model

+ Single model

	+ **basic_BiLSTM.py**: Single model uses BiLSTM structure, and dropout = 0.3.

+ Model ensemble

	+ **ensembleModel.py**: We adopt a simple ensemble method to improve the model performance, we run the single model 5 times, and set the dropout parameter = 0.3, 0.5, 0.4, 0.2, 0.6. Finally, we sum the results yielded by the 5 models to predict the correct answer. The experiments show that the accuracy of ensemble model is 2% higher than that of a single model.

	+ You can use a for loop to compress the code in the **ensembleModel.py**. We do not do so because we want to campare the perfomance after every ensemble.

## Results

| Dropout        | Dev Acc       | Test Acc      | 
| :-----------:  | :-----------: | :-----------: |    
| 0.3            |    0.7476     |     0.7311    |
| 0.5            |    0.7516     |     0.7183    |
| Ensemble 1     |    0.7608     |     0.7386    |
| 0.4            |    0.7615     |     0.7294    |
| Ensemble 2     |    0.7692     |     0.7408    |
| 0.2            |    0.7354     |     0.7143    |
| Ensemble 3     |    0.7664     |   **0.7479**  |
| 0.6            |    0.7410     |     0.7308    |
| Ensemble 4     |  **0.7699**   |   **0.7472**  |

The official leaderboard is here: [https://competitions.codalab.org/competitions/17184#results](https://competitions.codalab.org/competitions/17184#results). We ranked 8th among the 24 submissions to the task.
