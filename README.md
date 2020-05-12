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



