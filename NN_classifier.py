class NN_layers():
    """ Neural network for classification with 'L-1' ReLU activations, and a final sigmoid activation """

    def __init__(self, layer_dims):
        """ Initialization of NN_layers object """

        # layer_dims: list, contains the number of units in each layer, e.g. [x,10,5,1] or [x,5,1]. x = number of features.
        self.layer_dims = layer_dims 
        self.layer_num = len(layer_dims) - 1 # First layer isn't hidden layer but input layer

        self.params = {} # Dictionary containing a W, b for each hidden layer
        self.caches = [] # List of caches. A cache ((A_prev, W, b), Z) for each layer made in forward pass
        self.grads = {} # Dictionary containing the partial derivatives for W, b at each layer
        self.num_obv = 0 # The number of examples

        self._initialize_parameters() # Create weights and biases matrices/vectors in params

        self.report = {"settings":{},"iteration":[], "costs":[], "train_error":[], "dev_error":[], "params":{}}

    def parameter_checker(self): 
        """ Prints the shape of the weights and biases, used for checking linear algebra """
        for l in range(1, self.layer_num + 1):
            print(f'Layer {l}: Weight shape: {self.params["W" + str(l)].shape}, \
                                 Bias shape: {self.params["b" + str(l)].shape}')


    def _initialize_parameters(self):
        """ Create weights and biases. Biases start as zeros. Weights as small random """
        # Biases initially set to zero. Weights initially set to small positive values.
        for l in range(1, self.layer_num + 1):
            self.params['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) / np.sqrt(self.layer_dims[l-1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            # Check weight and biases are correct lengths
            assert(self.params['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert(self.params['b' + str(l)].shape == (self.layer_dims[l], 1))


    def _sigmoid(self, Z):
        """ Sigmoid activation function """
        return 1 / ( 1 + np.exp(-Z) )


    def _relu(self, Z):
        """ ReLU activation function """
        return np.maximum(0, Z)


    def _relu_backward(self, dA, Z):
        """ Implement backward prop for a single RelU unit, returns gradient dL/dZ """
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ


    def _sigmoid_backward(self, dA, Z):
        """ Implement backward prop for single sigmoid unit, returns gradient dL/dZ """
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert (dZ.shape == Z.shape)
        return dZ


    def _activation_for(self, A_prev, W, b, act_func):
        """ Returns information needed for back prop and next layer A, ((A_prev, W, b), Z) """
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)

        if act_func == "sigmoid":
            A = self._sigmoid(Z)
        elif act_func == "relu":
            A = self._relu(Z)

        assert(Z.shape == (W.shape[0], A.shape[1])) 
        return A, (linear_cache, Z)


    def _forward_propagation(self, X):
        """ Applies linear (Wx+b) -> reLu  'L-1' times before applying linear -> sigmoid once """
        
        # Empty cache ready for forward pass. Set initial activation to input matrix.
        self.caches = [] 
        A = X 
        
        # Loop through each 'L-1 layers'. Build caches. Each cache = ((A_prev, W, b), Z)
        for l in range(1, self.layer_num):
            A_prev = A
            A, cache = self._activation_for(A_prev, self.params['W'+str(l)], self.params['b'+str(l)], act_func= "relu")
            self.caches.append(cache)

        A_last, cache = self._activation_for(A, self.params['W'+str(self.layer_num)], self.params['b'+str(self.layer_num)], act_func= "sigmoid")
        self.caches.append(cache)

        # Check the last activations (i.e. predictions) are shape (1, number observations)
        assert(A_last.shape == (1,self.num_obv))

        return A_last


    def _activation_back(self, dA, cache, activation, lambd):
        """ For a given cache, return the partial derivatives of the cost function wrt w, b, A_prev"""

        # Unpack the activations, weights and biases from the cache.
        linear_cache, Z = cache
        A_prev, W, b = linear_cache
        
        # Calculate the loss function wrt Z, dependent on activation type
        if activation == 'relu':
            dZ = self._relu_backward(dA, Z)
        elif activation == 'sigmoid':
            dZ = self._sigmoid_backward(dA, Z)

        # Calculate gradients for loss function wrt dW, db, da_prev
        if lambd != None:
            dW = 1 / self.num_obv * np.dot(dZ, A_prev.T) + (lambd / self.num_obv * W)
        else:
            dW = 1 / self.num_obv * np.dot(dZ, A_prev.T)
        
        db = 1 / self.num_obv * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    def _back_propagation(self, A_last, Y, lambd):
        """ Apply backpropagation, loop from layer L to layer 1 """

        Y = Y.reshape(A_last.shape)

        # Calculate derivative of cross-entropy cost function wrt to dAL (i.e. last activations)
        dAL = - (np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))
        self.grads["dA"+str(self.layer_num)] = dAL

        # Get gradient for layer 'L' using dAL and last cache, and save gradients in grads for the last layer (sigmoid)
        curr_cache = self.caches[self.layer_num - 1] 
        dA_prev_temp, dW_temp, db_temp = self._activation_back(dAL, curr_cache, "sigmoid", lambd)

        self.grads["dA"+str(self.layer_num-1)] = dA_prev_temp
        self.grads["dW"+str(self.layer_num)] = dW_temp
        self.grads["db"+str(self.layer_num)] = db_temp

        # Loop backwards through layers, and save gradients (relu). Save in grads.
        for l in reversed(range(self.layer_num-1)):
            curr_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = self._activation_back(self.grads['dA' + str(l + 1)], curr_cache, "relu", lambd)
            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp


    def _update_params(self, learning_rate):
        """ Update the weights and biases """

        for l in range(self.layer_num):
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - (learning_rate * self.grads["dW" + str(l+1)])
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - (learning_rate * self.grads["db" + str(l+1)])  


    def train(self, X, y, num_iterations = 3000, learning_rate = 0.0075, print_cost = False, report = False, dev_set = None, lambd = None):
        """ Train the neural network, populating self.report and self.params """ 
        
        # X = numpy array of training data. Rows = features. Columns = observations. Important to get right way around.
        # y = numpy array of class data. 1 row. Column = classes (0 or 1).
        # num_iteraitons = number of iterations in gradient descent.
        # learning_rate = governs speed of progress in gradient descent.
        # print_cost = if True True will print the cost at ever 100 iterations.
        # report = will build a report of model performance.
        # dev_set = [x,y], where x = numpy array of input dev data, y = numpy array of class dev data.
        # lambd = if None then no regularization, otherwise value 0 to 1 which will 

        self.num_obv = X.shape[1]
        self.caches = []

        for i in range(0, num_iterations):
            # Execute forward pass (return the last activations, and populates self.caches)
            A_last = self._forward_propagation(X)
    
            # Compute the coss entropy loss (cost), if lambd = None then apply regularization.
            cost = (1/self.num_obv) * (-np.dot(y,np.log(A_last).T) - np.dot(1-y, np.log(1-A_last).T))
            cost = np.squeeze(cost)
            if lambd != None:
                L2_reg_cost = 0
                for l in range(1, self.layer_num + 1):
                    L2_reg_cost += np.sum(np.square(self.params["W" + str(l)]))
                cost += L2_reg_cost * lambd / (2 * self.num_obv)

            # Execute a backward pass (using the caches) which calculates gradients for each layer / parameter (W,b)
            self._back_propagation(A_last, y, lambd)

            # Update the params from the gradients.
            self._update_params(learning_rate)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

            # Build self.report which contains all the interesting information, parameters (weights / bias), iteration data, settings
            if (report and dev_set and i % 100 == 0):
                self.report["iteration"].append(i)
                self.report["costs"].append(cost)
                self.report["train_error"].append(1 - self.accuracy(self.predict(X), y))
                self.report["dev_error"].append(1 - self.accuracy(self.predict(dev_set[0]),dev_set[1]))

        self.report["settings"] = {"num_iterations":num_iterations,
                                    "learning_rate":learning_rate,
                                    "lambd":lambd,
                                    "layer_dims":self.layer_dims}
        
        self.report["params"] = self.params


    def predict(self, X):
        """ Make predictions from the trained neural network by pass data through neural network """

        A = X
        for l in range(1, self.layer_num):
            A_prev = A
            A, cache = self._activation_for(A_prev, self.params['W'+str(l)], self.params['b'+str(l)], act_func= "relu")
        A_last, cache = self._activation_for(A, self.params['W'+str(self.layer_num)], self.params['b'+str(self.layer_num)], act_func= "sigmoid")
        preds = (A_last > 0.5).astype(int)
 
        return preds


    def accuracy(self, preds, y):
        """ Returns accuracy analysis on prediction results vs. actual observations """

        num_pred = preds.shape[1]

        TP = sum([1 for i in range(num_pred) if (preds[0,i] == 1 and y[0,i] == 1)  ])
        TN = sum([1 for i in range(num_pred) if (preds[0,i] == 0 and y[0,i] == 0)  ])
        FP = sum([1 for i in range(num_pred) if (preds[0,i] == 1 and y[0,i] == 0)  ])
        FN = sum([1 for i in range(num_pred) if (preds[0,i] == 0 and y[0,i] == 1)  ])

        assert(TP + TN + FP + FN == num_pred)

        return (TP + TN) / num_pred


    def show_report(self):
        """ Shows the training performance, iterations vs. cross-entropy loss, and training / dev error """

        plt.figure(figsize = (12,8))

        plt.subplot(1,2,1)
        plt.plot(self.report["iteration"],self.report["costs"])
        plt.xlabel("iteration")
        plt.ylabel("cross-entropy loss")
        plt.title("Cross-entropy loss vs. number of iterations")

        plt.subplot(1,2,2)
        plt.plot(self.report["iteration"],self.report["train_error"], label = "train error")
        plt.plot(self.report["iteration"],self.report["dev_error"], label = "dev error")
        plt.xlabel("iteration")
        plt.ylabel("error")
        plt.title("Error vs. number of iterations")
        plt.legend()

        plt.tight_layout()
        plt.show()

        