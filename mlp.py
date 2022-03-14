import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)  # convert labels to categorical samples
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return (x_train, y_train, x_test, y_test)

def leakyrelu(a, alpha=0.01):
    # Returns a if a>0 and alpha*a if a<=0
	return a*(a>alpha) + alpha*a*(a<=alpha)

def leakyrelu_prime(a, alpha=0.01):
    # Returns 1 if a>0 and alpha if a<0
    return np.ones(a.shape)*(a>0) + alpha*(a<0)

def softmax(y):
    e = np.exp(y)  # exponentiates all output vectors y for each value in a batch
    sums = np.sum(e, axis=1)  # sums of each output vector y for each value in a batch
    return e*1/sums[:, np.newaxis]  # Divides exponentiated output vector y with corresponding sum

def cross_entropy(y, y_hat, e=1e-7):
    # Categorical cross-entropy used as loss function
    values = -y*np.log(y_hat+e)  # Small error added to avoid infinity / NaN issues
    return np.sum(values, axis=1)


class MLP():
    def __init__(self, mlp_configuration):
        self.config = mlp_configuration  # Configuration of MLP (# of neurons in each layer)
        self.hidden_layers = len(mlp_configuration) - 2  # Number of hidden layers
        # Weights between successive layers, initialised from a Gaussian
        self.weights = [np.random.normal(size=(mlp_configuration[k+1], mlp_configuration[k])) * np.sqrt(2./(mlp_configuration[k]+mlp_configuration[k+1])) for k in range(len(mlp_configuration)-1)]
        # Biases added to each layer to perform pre-activation
        self.biases = [np.zeros((mlp_configuration[k+1])) for k in range(len(mlp_configuration)-1)]
        self.a = [None for _ in range(len(mlp_configuration)-1)]  # Pre-activations
        self.h = [None for _ in range(len(mlp_configuration))]  # (Post) Activations
        self.d = [None for _ in range(len(mlp_configuration)-1)]  # Deltas - errors at each node
        self.grad_weights = [None for _ in range(len(mlp_configuration)-1)]  # Weights gradients
        self.grad_biases = [None for _ in range(len(mlp_configuration)-1)]  # Biases gradients
        self.batch_losses = []  # Will hold the batch-average (iteration) losses
        self.batch_accuracies = []  # Will hold the batch-average (iteration) accuracies
        self.epoch_losses = []  # Will hold the epoch-average losses
        self.epoch_accuracies = []  # Will hold the epoch-average accuracies
        self.test_epoch_losses = []  # Will hold the epoch-average test losses
        self.test_epoch_accuracies = []  # Will hold the epoch-average test accuracies

    def __str__(self):
        weights_shapes = [self.weights[i].shape for i in range(len(self.weights))]  # Shape of weights array
        biases_shapes = [self.biases[i].shape for i in range(len(self.biases))]  # Shape of biases array
        return "\nMLP Characteristics:\n--------------------\nMLP Configuration: {}\nWeights shapes: {} \
            \nBiases shapes: {}\n\n".format(self.config, weights_shapes, biases_shapes)
    
    def forward_pass(self, x):
        self.h[0] = x  # Initialise first activations to input nodes (input data)
        for k in range(self.hidden_layers):  # Loop through all hidden layers
            self.a[k] = np.matmul(self.weights[k], self.h[k].T).T + self.biases[k]  # Compute pre-activations
            self.h[k+1] = leakyrelu(self.a[k])  # Apply activation function (leaky RELU) to get (post) activations
        self.a[-1] = np.matmul(self.weights[-1], self.h[-2].T).T + self.biases[-1]  # Compute pre-activations for output layer
        self.h[-1] = softmax(self.a[-1])  # Apply output-activation function (softmax) to get output nodes

    def backpropagate_errors(self, y):
        self.d[-1] = self.h[-1] - y  # Get output errors using softmax and categorical cross-entropy
        for k in reversed(range(self.hidden_layers)):  # Loop through all hidden layers in reverse
            self.d[k] = leakyrelu_prime(self.a[k]) * (self.weights[k+1].T @ self.d[k+1].T).T  # Backpropagate errors
            
    def update_params(self, learning_rate):
        for k in range(self.hidden_layers+1):  # Loop through all layers (except output)
            # Taking the average weight gradients over all inputs in a batch (outer product for 3D matrix)
            self.grad_weights[k] = np.matmul(self.d[k][:, :, np.newaxis], self.h[k][:, np.newaxis, :]).mean(axis=0)
            self.grad_biases[k] = self.d[k].mean(axis=0)  # Compute biases gradients using deltas
            self.weights[k] -= learning_rate*self.grad_weights[k]  # Update weights with learning rate
            self.biases[k] -= learning_rate*self.grad_biases[k]  # Update biases with learning rate

    def make_prediction(self, x, y_t):
        # Perform a forward pass to obtain a prediction on a batch of input data
        values = x  # Initialise first activations to input nodes (input data)
        for k in range(self.hidden_layers):  # Loop through all hidden layers
            values = np.matmul(self.weights[k], values.T).T + self.biases[k]  # Compute pre-activations
            values = leakyrelu(values)  # Apply activation function (leaky RELU) to get (post) activations
        final_values = np.matmul(self.weights[-1], values.T).T + self.biases[-1]  # Compute pre-activations for output layer
        final_values = softmax(final_values)  # Apply output-activation function (softmax) to get output nodes
        
        crs_entr = cross_entropy(y_t, final_values)  # Obtain loss value (categorical cross-entropy)
        prediction = np.zeros((final_values.shape))  # Array to convert probabilities to class prediction
        successful = np.zeros((final_values.shape[0]))  # Array to hold successful predictions for each datum
        for p in range(len(prediction)):  # Looping through all data-points in a batch
            prediction[p][np.argmax(final_values[p])] = 1  # Assign the predicted class to the highest probability
            # Assign as succesful prediction if it the same as that of the ground truth
            successful[p] = [1 if np.all(prediction[p] == y_t[p]) else 0][0]
        return np.mean(crs_entr), np.mean(successful)  # Return batch-average loss and accuracy

    def train(self, x_train, y_train, x_test, y_test, learning_rate=1e-3, batch_size=256, epochs=40):
        for e in range(epochs):
            p = np.random.permutation(len(x_train))  # Randomise indices of all data
            random_indices = [p[idx:idx+batch_size] for idx in range(0, len(p), batch_size)]  # Create random batch indices
            for rnd_idx in random_indices:  # Loop through batch indices
                x_batch, y_batch = x_train[rnd_idx], y_train[rnd_idx]  # Create batches of X and Y data
                self.forward_pass(x_batch)  # Perform forward pass on batch of data
                self.backpropagate_errors(y_batch)  # Perform backprogatation of errors on batch of data
                self.update_params(learning_rate)  #  Update MLP parameters (weights, biases) after backpropagation
                batch_loss, batch_accuracy = self.make_prediction(x_batch, y_batch)  # Obtain batch-average loss and accuracy
                self.batch_accuracies.append(batch_accuracy)
                self.batch_losses.append(batch_loss)
            
            epoch_loss, epoch_accuracy = self.make_prediction(x_train, y_train)  # Obtain epoch loss and accuracy on train data
            self.epoch_losses.append(epoch_loss)
            self.epoch_accuracies.append(epoch_accuracy)
            test_loss, test_accuracy = self.make_prediction(x_test, y_test)  # Obtain epoch loss and accuracy on test data
            self.test_epoch_losses.append(test_loss)
            self.test_epoch_accuracies.append(test_accuracy)
            print('Completed:', str(e+1) + '/' + str(epochs))


    def plot(self, learning_rate=1e-3, batch_size=256, epochs=40):
        plt.figure(1)
        plt.plot(np.linspace(1, len(self.epoch_accuracies), len(self.epoch_accuracies)), self.epoch_accuracies, color='blue', label='Train Data')
        plt.plot(np.linspace(1, len(self.test_epoch_accuracies), len(self.test_epoch_accuracies)), self.test_epoch_accuracies, color='orange', label='Test Data')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xlim(1, len(self.epoch_accuracies))
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig('epoch_accuracy_'+str(epochs)+'_'+str(batch_size)+'_'+str(learning_rate)+'.png')
        
        plt.figure(2)
        plt.plot(np.linspace(1, len(self.epoch_losses), len(self.epoch_losses)), self.epoch_losses, color='blue', label='Train Data')
        plt.plot(np.linspace(1, len(self.test_epoch_losses), len(self.test_epoch_losses)), self.test_epoch_losses, color='orange', label='Test Data')
        plt.xlabel('Epoch')
        plt.ylabel('Categorical Cross-Entropy')
        plt.xlim(1, len(self.epoch_losses))
        plt.legend()
        plt.savefig('epoch_loss_'+str(epochs)+'_'+str(batch_size)+'_'+str(learning_rate)+'.png')

        plt.figure(3)
        plt.plot(np.linspace(1, len(self.batch_accuracies), len(self.batch_accuracies)), self.batch_accuracies, color='blue', label='Train Data - Batch Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.xlim(0, len(self.batch_accuracies))
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig('batch_accuracy_'+str(epochs)+'_'+str(batch_size)+'_'+str(learning_rate)+'.png')
        
        plt.figure(4)
        plt.plot(np.linspace(1, len(self.batch_losses), len(self.batch_losses)), self.batch_losses, color='orange', label='Train Data - Batch Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Categorical Cross-Entropy')
        plt.xlim(0, len(self.batch_losses))
        plt.legend()
        plt.savefig('batch_loss_'+str(epochs)+'_'+str(batch_size)+'_'+str(learning_rate)+'.png')

        plt.show()

x_train, y_train, x_test, y_test = load_data()
n_input = x_train.shape[-1]**2  # making each image pixel into a feature
x_train = x_train.reshape(len(x_train), n_input)  # flattening each datum array into a 1D array
x_test = x_test.reshape(len(x_test), n_input)  # flattening each datum array into a 1D array
n_classes = y_train.shape[1]

# determining the hyper-parameters
learning_rate = 1e-3
batch_size = 256
epochs = 40

# determining the structure of the MLP network
mlp_config = (n_input, 400, 400, 400, 400, 400, n_classes)
mlp = MLP(mlp_config)
print(mlp)
x_test = x_test.reshape(len(x_test), n_input)  # flattening each datum array into a 1D array
mlp.train(x_train, y_train, x_test, y_test, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
mlp.plot(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
