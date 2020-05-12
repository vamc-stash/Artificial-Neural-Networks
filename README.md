# Artificial-Neural-Networks
[ Implemented several machine learning techniques such as regression, binary classification, multiclass classification using ANNs as part of the Udemy course Neural Networks (ANN) using Keras and Tensorflow in Python ] 

## INTRODUCTION
`Neural networks`, as its name suggests, is a machine learning technique which is modeled after the brain structure. It comprises a network of learning units called `neurons`. <br>
### Deep Learning
Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called Neural Networks. Neural networks are typically organized in layers. The term 'deep' usually refers to the number of hidden layers in the neural network. Traditional neural networks only contain 2-3 hidden layers, while deep networks can have as many as 150. Deep learning models are trained by using large sets of labeled data and neural network architectures that learn features directly from the data without the need for manual feature extraction. The different types of neural networks in deep learning, such as convolutional neural networks(CNN) , recurrent neural networks(RNN), artificial neural networks(ANN).
### Artificial Neural Networks(ANNs)
An artificial neuron network (ANN) is a computational model based on the structure and functions of biological neural networks. Information that flows through the network affects the structure of the ANN because a neural network changes - or learns, in a sense - based on that input and output.<br>
ANNs are considered nonlinear statistical data modeling tools where the complex relationships between inputs and outputs are modeled or patterns are found.<br>
In this method, first it involves in building the network for the model, parameters to be tuned in the beginning of the training process such as number of input nodes, hidden nodes, output nodes and initial weights, learning rates and activation function. 
### Supervised Learning
Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output.   
 `  Y = f(x) ` <br>
The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the output variables (Y) for that data. Supervised learning technique deals with the labelled data where the output data patterns are known to the system.<br>

* **Classification**
Classification is a process of categorizing a given set of data into classes, It can be performed on both structured or unstructured data. The process starts with predicting the class of given data points. The classes are often referred to as target, label or categories.<br>
The classification predictive modeling is the task of approximating the mapping function from input variables to discrete output variables. The main goal is to identify which class/category the new data will fall into. <br>
Ex. Classifying mail into spam or not spam<br>
  * **Binary Classification**
Binary or binomial classification is the task of classifying the elements of a given set into two groups (predicting which group each one belongs to) on the basis of a classification rule. <br>
Ex. Medical testing to determine if a patient has certain disease or not – the classification property is the presence of the disease.<br>
  * **Multi-Class Classification**
The classification with more than two classes, in multi-class classification each sample is assigned to one and only one label or target.<br>
Ex. classify a set of images of fruits which may be oranges, apples, or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.<br>
  * **Multi-Label Classification**
The classification where each sample is assigned to a set of labels or targets. This can be thought as predicting properties of a data-point that are not mutually exclusive. <br>
Ex. Find topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.<br>
* **Regression**
Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target Y) and independent variable(Xs) (predictor). In this case outputs are continuous rather than discrete.<br>
There are various kinds of regression techniques available to make predictions, such as Linear Regression, Logistic Regression, Polynomial Regression, Stepwise Regression, Ridge Regression, Lasso Regression, ElasticNet Regression etc.<br>
Ex. Quantify the relative impacts of age, gender, and diet (the predictor variables) on height (the outcome variable). <br>

## IMPLEMENTATION
#### [Binary Classification](https://github.com/vamc-stash/Artificial-Neural-Networks/blob/master/Binary_classification/binary_classification.ipynb)
To implement binary classification, I used `iris` dataset from `sklearn.datasets`. Using this dataset, i built a binary classifier that classifies whether a species is "Setosa" or not.<br>
I used Perceptron() model from sklearn.linear_model to fit X (sepal length,sepal width, petal length,petal width) and Y (Setosa(1) or not(0)) into a model. This built model is used to predict whether a species is setosa or not.<br>
[Assignment](https://github.com/vamc-stash/Artificial-Neural-Networks/blob/master/Assignment/binary_classification.ipynb)<br>
In this part, we used this [dataset](https://github.com/vamc-stash/Artificial-Neural-Networks/blob/master/Assignment/original.csv). Using this dataset, I implemented binary classifier to predict whether a house is Sold or not. Lot of pre-processing techniques like filling missing values, outlier treatment, removing independent columns that are highly correlated etc. are applied to this dataset. Model was built using `ANN Sequential` approach with  `sigmoid` as a final layer activation function, `binary_crossentropy` as a loss function, `stochastic gradient descent` as an optimizer and `accuracy` as metrics. Train accuracy of 65.85% and test accuracy of 58.82% was achieved. <br>

#### [Multi-Class Classification](https://github.com/vamc-stash/Artificial-Neural-Networks/tree/master/Multiclass_classificaton)
To implement multi-class classification model, I used `Fashion MNIST` dataset from `keras.datasets`. This Dataset consists of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. Model was built using `ANN
Sequential` approach with  `softmax` as a final layer activation function, `sparse_categorical_crossentropy` as a loss function, `stochastic gradient descent` as an optimizer and `accuracy` as metrics. Train accuracy of 93.42% and test accuracy of 89.14% was achieved.<br>

#### [Regression](https://github.com/vamc-stash/Artificial-Neural-Networks/tree/master/Regression)
For Regression analysis, I used `California housing` dataset from `sklearn.datasets`. This dataset consists of 20,640 samples and 9 features. The target variable is the median house value for California districts. Data was normalized using StandardScalar library from sklearn.preprocessing. <br>
*Sequential Model* was built using `ANN Sequential` approach with `mean_squared_error` as a loss function, `stochastic gradient descent` as an optimizer and `mean absolute error` as metrics. Training Mean absolute error of 0.4469 and testing mean absolute error of 0.4464 was obtained after running over 120 epochs. <br>
*Complex Functional Model* was built using **Functional API** approach(It specifically allows you to define multiple input or output models as well as models that share layers. More than that, it allows you to define ad hoc acyclic network graphs). Flow diagram of the model built using `ANN Functional` approach can be seen [here](https://github.com/vamc-stash/Artificial-Neural-Networks/blob/master/Regression/model.png). Hyperparameters used for this approach are same as sequential approach. Training Mean absolute error of 0.4220 and testing mean absolute error of 0.4216 is obtained after running over 120 epochs.<br>
While training, various Model Checkpoints are implemented using Keras.callbacks to store best model and early stop model of the above sequential approach. <br>

## INSTALLATIONS
`jupyter-notebook` `numpy` `pandas` `seaborn` `matplotlib` `Sklearn` `Keras` `Tensorflow`
## ACKNOWLEDGMENTS
https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6
https://towardsdatascience.com/introduction-to-artificial-neural-networks-ann-1aea15775ef9
https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
