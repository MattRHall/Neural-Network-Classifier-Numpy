## N-Layer Neural Network for Classification from First Principles
Constructs a neural network based on a specified number of layers / units. ReLU activations are applied on 'L-1' layers, before a final sigmoid activation. The model is trained using gradient descent, and is uses numpy for its computations.  
  
  
### Purpose
The purpose for this project was to better understand neural networks by building and training a neural network from scratch. This project was initially applied to a spam e-mail classification problem, but it can be applied to other uses.

### Features
* Customisability - The neural network has customisable architecture (layers / units), gradient descent parameters, and regularization parameter.
* Model testing - Every 100 iterations of gradient descent the traininng data (and optional development data) is passed through the model, and the error rate in classification returned.
* Feedback - A report can be returned after gradient descent, which contains the settings, parameters, and model performance every 100 iterations of gradient descent.
* Parameter loading - The classmethod .load_model(params) will build a NN with pre-trained parameters. After training a report is returned, with key "params" which contains the neural networks parameters. This can be pickled and reloaded with this classmethod.  

### Usage  
  
(Class) **NN_layers(layer_dims)** will instantiate a neural network object. layer_dims is a list containing the dimensions of the neural network, e.g. [50, 25, 10 , 1]. [50, 25, 10, 1] would be a 4 layer neural network with 2 densely connected hidden layers (with 25 & 10 units), an output layer with sigmoid activation, and an input layer with 50 features.     
  

(Function) **.train(X, y)** will train a neural network. The parameters are as follows:  
* X = numpy array, rows = features, columns = observations  
* y = numpy array, rows = 1, columns = class observations
* num_iterations = integer, number of iterations of gradient descent
* learning_rate = integer, used to determine progress of gradient descent
* print_cost = Boolean, if True will print out the cost function at every 100 iterations of gradient descent
* report = Boolean, if True will return a report containing information about how the training performed
* dev_set = [a,b], where 'a' is a development training set, and 'b' is a development test set (used to test model)
* lambd = real number 0 to 1, used for L2- regularization, if None, no L2-regularization.

(Function) **.predict(X)** will return class predictions for a numpy array of observations X.
  
(Function) **.accuracy(preds, y)** will take the predicted observations and their true class and return the accuracy.
  
### Example
model = NN_Layers([54,25,10,1])  
model.train(X, y)  
preds = model.predict(a)  
accuracy = model.accuracy(preds, b)




