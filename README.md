# what-to-watch
What are the factors that lead to hit movie? I developed and evaluated various supervised-learning models. On average, the best model predicted movie scores within 0.5 points (out of 10) based on just four features, aiding movie-goers and film studios to predict future success early.

### Process Notebook
To explain the data cleaning, processing and model design and development process, I created a Jupyter Notebook. 

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

**Supervised Learning Paradigms: **

There are various supervised learning approaches. Linear or multilinear regression, decision trees and random forests, hidden-layer neural networks, support vector machines (SVM) and others. For this problem I selected the random forest, support vector machine and multi-layer perceptron models. The following justifies my model selection.

**Decision Trees and Random Forest:**

Decision trees are forked nodes with binary decisions at each node at varying depths. They work well with categorical data and are simple to explain. However, I found that this technique may be prone to overfitting. Overfitting is the case where the model very closely predicts the training data but in turn performs poorly when presented new, unseen data.  Overfitting in decision trees can be avoided by using multiple randomly sampled decision trees and averaging over their output values. This approach is the foundation of random forests and why it would be a good solution for this problem. 

**Support Vector Machines: **

Another approach is the support vector machine (SVM) for regression. Support vector machines use a non-linear "kernel", or function, to project the dataset into a higher dimension. This enables the algorithm to linearly separate the data into classes, even if this weren't possible with the original dimensionality of the data. This can also be extended to a regression problem - the scikit-learn library uses the "SVR" method to build such a model. The benefit of using this model is that it has only two hyperparameters to tune and works well with high-dimensional data.  

**Neural Networks: **

The third method is the multi-layer perceptron (MLP) neural network. In this approach there is an n-dimensional input layer where the features are fed to the model. This data is scaled by a set of weight and propagated to a layer of cells, each summing its inputs and performing a simple function called an activation function on the inputted data. The output of each cell, goes to the next layer of cells. Each cell is called a perceptron, or artificial neuron. The final layer is a single neuron that outputs the score.  

This approach is beneficial because there are a variety of hyperparameters that can be adjusted to improve performance like learning rate, number of neurons per layer, and number of hidden layers. This provides great design flexibility and potential for performance improvement. However, it may be less explainable than decision trees or SVM as there is a greater level of abstraction. Hyperparameter tuning is challenging too. 

### Results

The 13-hidden layer neural network (NN) model had a low training MSE (0.303), indicating that the model had not been overfit to the training data. The Random Forest (RF) and Support Vector Regression (SVR) models had even lower training MSE values (0.169, 0.164). As expected, despite the low training errors, the testing error of all models was larger. As unseen data is slightly different to what the model was trained on, there is higher variance. 

The RF model had the least validation MSE of 0.271, followed closely by the NN model at 0.339. The SVR model trailed with a MSE of 0.412. Considering the root-mean-squared-error (RMSE) of each of these models gives insight into the average difference from the true score in "score units". These are 0.582, 0.521, and 0.642 for the NN, RF and SVR models, respectively. 

This suggests that on average, scores are about 0.5-0.7 points off from the true score. This, in theory, should provide good model performance.  

### Discussion

Considering RMSE, the performance difference between the NN and RF models becomes less significant. The performance of these two models may be comparable in a real scenario. However, the random forest was computationally much easier to train, and, if one were to analyze its structure, is more transparent than a neural network. 

However, this dataset was small, and only considered a little less than 7000 movies. If millions of movies are considered, the random forest may struggle computationally, as the depth of each decision tree grows with the scale of the data. On the other hand, the neural network may still be able to perform well with a fixed architecture by tuning its weights (fewer parameters than multiple, deep decision trees). Nevertheless, better performance for data at scale may require the neural network's architecture to be tweaked. This could be an avenue for further research.  

Demo of Limitation, Future Work

The nature of this model is that it uses past success to predict future ones. However, some movies or films may be breakthroughs in their genre, involving debutant actors and directors who may not have had past success. These instances could not be evaluated by the model, which requires prior knowledge of the cast/crew. Furthermore, while the average score deviation was low, will the score always be around 0.5 points off from the true score or is there a large variance? 

We will gauge this with a few examples.

### Validation testing
Movie Title:  Bullet Train
╒═══════════════════════════╤═════════════════════╤══════════════════╕
│ Model                     │   Score (out of 10) │   Absolute Error │
╞═══════════════════════════╪═════════════════════╪══════════════════╡
│ Actual score              │                 7.3 │              0.0 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Neural Network            │                 7.1 │             -0.2 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Random Forest             │                 7.0 │             -0.3 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Support Vector Regression │                 6.8 │             -0.5 │
╘═══════════════════════════╧═════════════════════╧══════════════════╛
Movie Title:  Black Adam
╒═══════════════════════════╤═════════════════════╤══════════════════╕
│ Model                     │   Score (out of 10) │   Absolute Error │
╞═══════════════════════════╪═════════════════════╪══════════════════╡
│ Actual score              │                 6.6 │              0.0 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Neural Network            │                 6.2 │             -0.4 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Random Forest             │                 6.1 │             -0.5 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Support Vector Regression │                 6.2 │             -0.4 │
╘═══════════════════════════╧═════════════════════╧══════════════════╛
Movie Title:  The Whale
╒═══════════════════════════╤═════════════════════╤══════════════════╕
│ Model                     │   Score (out of 10) │   Absolute Error │
╞═══════════════════════════╪═════════════════════╪══════════════════╡
│ Actual score              │                 8.3 │              0.0 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Neural Network            │                 6.5 │             -1.8 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Random Forest             │                 6.6 │             -1.7 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Support Vector Regression │                 6.3 │             -2.0 │
╘═══════════════════════════╧═════════════════════╧══════════════════╛
Movie Title:  Amsterdam
╒═══════════════════════════╤═════════════════════╤══════════════════╕
│ Model                     │   Score (out of 10) │   Absolute Error │
╞═══════════════════════════╪═════════════════════╪══════════════════╡
│ Actual score              │                 6.1 │              0.0 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Neural Network            │                 7.4 │              1.3 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Random Forest             │                 7.2 │              1.1 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Support Vector Regression │                 7.3 │              1.2 │
╘═══════════════════════════╧═════════════════════╧══════════════════╛
Movie Title:  Son of the Mask
╒═══════════════════════════╤═════════════════════╤══════════════════╕
│ Model                     │   Score (out of 10) │   Absolute Error │
╞═══════════════════════════╪═════════════════════╪══════════════════╡
│ Actual score              │                 2.2 │              0.0 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Neural Network            │                 3.4 │              1.2 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Random Forest             │                 3.4 │              1.2 │
├───────────────────────────┼─────────────────────┼──────────────────┤
│ Support Vector Regression │                 3.8 │              1.6 │
╘═══════════════════════════╧═════════════════════╧══════════════════╛


The test above illustrates that for some movies, the model is accurate within less than its RMSE (0.5-0.7), while for others it deviates up to 2.0 points away. 

"The Whale" is a good example of the "past success" problem discussed earlier because it was a well-received movie with a director (Darren Aronofsky) who had had a history of good movies (average rating of 7.3) and the main actor, Brendan Fraser, had made a stellar comeback to his acting career on the big screen. However, when analyzing Fraser's movies in this dataset, movies where he was the main star got an average score of 5.8.  Consequently, when a "great director" is paired with a "bad actor", the model predicts that the overall score would be somehow negatively modulated, thus leading to the lower score. 

The biggest drawback of this model is that it has no peripheral vision on an actor or director's improvement of worsening over time. Actors and directors continue to work outside of the TV and movie industry as writers, producers or consultants on other projects which may provide insight into their aptitude. It expects that actors and directors will continue to perform as they did before (as illustrated earlier). The model would greatly benefit from being an online model that continuously learns from new data.   

Further ways to improve this model include considering more than one actor/director, including features like writers, quality and content of the work (IMDb review keywords, synopsis keywords), internet culture (twitter sentiment analysis), whether it is part of an existing successful franchise (IMDb metadata).

