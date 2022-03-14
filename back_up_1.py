import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # convert labels to categorical samples
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return (x_train, y_train, x_test, y_test)


def leakyrelu(a, alpha=0.01):
	return a*(a>alpha) + alpha*a*(a<alpha)

def leakyrelu_prime(a, alpha=0.01):
    return np.ones(a.shape)*(a>0) + alpha*(a<0)

def softmax(y):
    e = np.exp(y)
    if y.ndim == 1:
        sums = np.sum(e)
        return e*1/sums
    else:
        sums = np.sum(e, axis=1)
        return e*1/sums[:, np.newaxis]

# def cross_entropy(y, y_hat):
#     return np.sum([-y_i * np.log(y_hat_i) if y_hat_i != 0 else 0 for y_i, y_hat_i in zip(y, y_hat)])

def cross_entropy(y, y_hat, e=1e-7):
    values = -y*np.log(y_hat+e)
    if y.ndim == 1:
        return values.sum()
    else:
        return np.sum(values, axis=1)


class MLP():
    def __init__(self, mlp_configuration):
        self.config = mlp_configuration  # Configuration of MLP (# of neurons in each layer)
        self.hidden_layers = len(mlp_configuration) - 2  # Number of hidden layers
        # Weights between successive layers, initialised from a Gaussian
        self.weights = [np.random.normal(size=(mlp_configuration[k+1], mlp_configuration[k])) * np.sqrt(2./mlp_configuration[k]) for k in range(len(mlp_configuration)-1)]
        # Biases added to each layer to perform pre-activation
        self.biases = [np.zeros((mlp_configuration[k+1])) for k in range(len(mlp_configuration)-1)]
        # self.a = [np.zeros((mlp_configuration[k+1])) for k in range(len(mlp_configuration)-1)]  # Pre-activations
        # self.h = [np.zeros((mlp_configuration[k])) for k in range(len(mlp_configuration))]  # (Post) Activations
        # self.d = [np.zeros((mlp_configuration[k+1])) for k in range(len(mlp_configuration)-1)]  # Deltas - errors at each node
        # self.grad_weights = [np.zeros((mlp_configuration[k+1], mlp_configuration[k])) for k in range(len(mlp_configuration)-1)]  # Weights gradients
        # self.grad_biases = [np.zeros((mlp_configuration[k+1])) for k in range(len(mlp_configuration)-1)]  # Biases gradients
        self.a = [None for _ in range(len(mlp_configuration)-1)]  # Pre-activations
        self.h = [None for _ in range(len(mlp_configuration))]  # (Post) Activations
        self.d = [None for _ in range(len(mlp_configuration)-1)]  # Deltas - errors at each node
        self.grad_weights = [None for _ in range(len(mlp_configuration)-1)]  # Weights gradients
        self.grad_biases = [None for _ in range(len(mlp_configuration)-1)]  # Biases gradients

    def __str__(self):
        weights_shapes = [self.weights[i].shape for i in range(len(self.weights))]  # Shape of weights array
        biases_shapes = [self.biases[i].shape for i in range(len(self.biases))]  # Shape of biases array
        # pre_activations_shapes = [self.a[i].shape for i in range(len(self.a))]  # Shape of pre-activations array
        # activations_shapes = [self.h[i].shape for i in range(len(self.h))]  # Shape of (post) activations array
        # weights_gradients_shapes = [self.grad_weights[i].shape for i in range(len(self.grad_weights))]  # Shape of weights gradients array
        # biases_gradients_shapes = [self.grad_biases[i].shape for i in range(len(self.grad_biases))]  # Shape of biases gradients array
        # deltas_shapes = [self.d[i].shape for i in range(len(self.d))]  # Shape of deltas (errors) array
        # return "\nMLP Characteristics:\n--------------------\nMLP Configuration: {}\nWeights shapes: {} \
        #     \nBiases shapes: {}\nPre-activations shapes: {}\nActivations shapes: {}\nDeltas: {}\nWeights gradients shapes: {} \
        #     \nBiases gradients shapes: {}\n\n".format(self.config, weights_shapes, biases_shapes, \
        #     pre_activations_shapes, activations_shapes, deltas_shapes, weights_gradients_shapes, biases_gradients_shapes)
        return "\nMLP Characteristics:\n--------------------\nMLP Configuration: {}\nWeights shapes: {} \
            \nBiases shapes: {}\n\n".format(self.config, weights_shapes, biases_shapes)
    
    def forward_pass(self, x):
        self.h[0] = x  # Initialise first activations to input nodes (input data)
        for k in range(self.hidden_layers):  # Loop through all hidden layers
            self.a[k] = np.matmul(self.weights[k], self.h[k]) + self.biases[k]  # Compute pre-activations
            self.h[k+1] = leakyrelu(self.a[k])  # Apply activation function (leaky RELU) to get (post) activations
        self.a[-1] = np.matmul(self.weights[-1], self.h[-2]) + self.biases[-1]  # Compute pre-activations for output layer
        self.h[-1] = softmax(self.a[-1])  # Apply output-activation function (softmax) to get output nodes

    def backpropagate_errors(self, y):
        self.d[-1] = self.h[-1] - y  # Get output errors using softmax and categorical cross-entropy
        for k in reversed(range(self.hidden_layers)):  # Loop through all hidden layers in reverse
            self.d[k] = leakyrelu_prime(self.a[k]) * (self.weights[k+1].T @ self.d[k+1])  # Backpropagate errors

    def update_params(self, learning_rate):
        for k in range(self.hidden_layers+1):  # Loop through all layers (except output)
            self.grad_weights[k] = np.outer(self.d[k], self.h[k])  # Compute weight gradients using deltas
            self.grad_biases[k] = self.d[k]  # Compute biases gradients using deltas
            self.weights[k] -= learning_rate*self.grad_weights[k]  # Update weights with learning rate
            self.biases[k] -= learning_rate*self.grad_biases[k]  # Update biases with learning rate

    def make_prediction(self, x_t, y_t):
        values = x_t
        for k in range(self.hidden_layers):
            values = leakyrelu(np.matmul(self.weights[k], values) + self.biases[k])
        final_values = softmax(np.matmul(self.weights[-1], values) + self.biases[-1])
        
        crs_entr = cross_entropy(y_t, final_values)
        prediction = np.zeros((final_values.shape))
        prediction[np.argmax(final_values)] = 1
        successful = [1 if np.all(prediction == y_t) else 0][0]
        return crs_entr, successful

    def train(self, x_train, y_train, learning_rate, batch_size=256, epochs=40):
        avg_acc = []
        avg_loss = []
        i = 0
        for e in range(epochs):
            accuracy = []
            losses = []
            for x, y in zip(x_train[i*500:i*500+500], y_train[i*500:i*500+500]):
                self.forward_pass(x)
                self.backpropagate_errors(y)
                self.update_params(learning_rate)
                accuracy.append(self.make_prediction(x, y)[1])
                losses.append(self.make_prediction(x, y)[0])
            i += 1
            avg_acc.append(np.mean(accuracy))
            avg_loss.append(np.mean(losses))
            print('Completed:', str(e+1) + '/' + str(epochs))
        
        plt.plot(np.linspace(1, len(avg_acc), len(avg_acc)), avg_acc, label='Accuracy')
        plt.plot(np.linspace(1, len(avg_loss), len(avg_loss)), avg_loss, label='Categorical Cross-Entropy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

x_train, y_train, x_test, y_test = load_data()
n_input = x_train.shape[-1]**2  # making each image pixel into a feature
x_train = x_train.reshape(len(x_train), n_input)  # flattening each datum array into a 1D array
x_test = x_test.reshape(len(x_test), n_input)  # flattening each datum array into a 1D array
n_classes = y_train.shape[1]

# determining the hyper-parameters
learning_rate = 1e-3
batch_size = 256
epochs = 20

# determining the structure of the MLP network
mlp_config = (n_input, 400, 400, 400, 400, 400, n_classes)
mlp = MLP(mlp_config)
print(mlp)
mlp.train(x_train, y_train, learning_rate, batch_size, epochs)
