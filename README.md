# what-to-watch
What are the factors that lead to hit movie? I developed and evaluated various supervised-learning models to predict movie scores based on four features to aid movie-goers and film studios to predict success early. I achieved a mean model accuracy of 0.5 points (out of 10).

### Repo structure

- /data: contains the raw and cleaned dataset
- process_notebook.ipynb: contains the process notebook (Jupyter Notebook) explaining the entire research process including data cleaning, processing and model design and development process. 

Note:

There are two datasets used in this experiment. In the /data/raw folder, the first file is titled movies.csv. The second dataset used to augment the first is too large to be uploaded to GitHub. It can be downloaded from IMDb's data repo:

IMDb movie-language metadata dataset (https://datasets.imdbws.com/title.akas.tsv.gz)
- associates a movie with its language. These datasets can then be merged on rows where language = english. 

### About the Data

The "Movie Industry" dataset selected for this project is hosted on Kaggle (https://www.kaggle.com/datasets/danielgrijalvas/movies?resource=download). This was directly downloaded. The data contains a variety of features, but has several missing or inaccurate values as detailed below. These will be dealt with before use with a machine learning algorithm, as the quality of the training data is crucial to a model's quality.

<u>About the data:</u>
* 7668 unique, multilingual movies 
* Dates: 1986-2020
* Features:
  * **name**: movie title
  * **rating**: age rating (R, PG, etc.)
  * **genre**: priamry genre of the movie
  * **year**: year of release
  * **released**: release date (YYYY-MM-DD)
  * **score**: IMDb user rating
  * **votes**: number of user votes
  * **director**: the director
  * **writer**: primary writer of the movie
  * **star**: primary actor of the movie
  * **country**: country of origin/filming
  * **budget**: budget of a movie
  * **gross**: revenue of the movie
  * **company**: production company
  * **runtime**: duration of the movie


Through exploratory data analysis (see notebook) I narrowed down the variables of interest and formulated the research question:

**How can the IMDb score of a given movie be predicted using its Motion Picture Association rating, genre, director, and main actor?**

### Models Selected

This problem lends itself to a regression problem as there is a continuous target variable (score), and multiple independent variables to consider.  While there are supervised and unsupervised learning techniques, the former is a better choice since labeled data is available. 

**Supervised Learning Paradigms:**

There are various supervised learning approaches. Linear or multilinear regression, decision trees and random forests, hidden-layer neural networks, support vector machines (SVM) and others. For this problem I selected the random forest, support vector machine and multi-layer perceptron models. The following justifies my model selection.

**Decision Trees and Random Forest:**

Decision trees are forked nodes with binary decisions at each node at varying depths. They work well with categorical data and are simple to explain. However, I found that this technique may be prone to overfitting. Overfitting is the case where the model very closely predicts the training data but in turn performs poorly when presented new, unseen data.  Overfitting in decision trees can be avoided by using multiple randomly sampled decision trees and averaging over their output values. This approach is the foundation of random forests and why it would be a good solution for this problem. 

**Support Vector Machines:**

Another approach is the support vector machine (SVM) for regression. Support vector machines use a non-linear "kernel", or function, to project the dataset into a higher dimension. This enables the algorithm to linearly separate the data into classes, even if this weren't possible with the original dimensionality of the data. This can also be extended to a regression problem - the scikit-learn library uses the "SVR" method to build such a model. The benefit of using this model is that it has only two hyperparameters to tune and works well with high-dimensional data.  

**Neural Networks:**

The third method is the multi-layer perceptron (MLP) neural network. In this approach there is an n-dimensional input layer where the features are fed to the model. This data is scaled by a set of weight and propagated to a layer of cells, each summing its inputs and performing a simple function called an activation function on the inputted data. The output of each cell, goes to the next layer of cells. Each cell is called a perceptron, or artificial neuron. The final layer is a single neuron that outputs the score.  

This approach is beneficial because there are a variety of hyperparameters that can be adjusted to improve performance like learning rate, number of neurons per layer, and number of hidden layers. This provides great design flexibility and potential for performance improvement. However, it may be less explainable than decision trees or SVM as there is a greater level of abstraction. Hyperparameter tuning is challenging too. 
