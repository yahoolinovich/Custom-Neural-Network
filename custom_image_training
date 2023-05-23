import numpy as np
import imageio.v2 as imageio
import os
import random


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)
        self.syn_weights = 2 * np.random.random((len(training_inputs), len(training_inputs[0]))) - 1  # 200 x 784
        self.syn_weights1 = 2 * np.random.random((10, len(training_inputs))) - 1  # 10 x 200

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def learn(self, input, out, iterations):
        for i in range(iterations):
            print(f'Processing {i}th iteration')
            for index, data in enumerate(input):
                self.train(np.vstack(data), np.vstack(out[index]))

    def train(self, input_layer, trn_outputs):
        layer1 = self.hidden_layer_1(input_layer)
        layer2 = self.hidden_layer_2(layer1)

        error = trn_outputs - layer2

        adjustments = error * self.sigmoid_deriv(layer2)
        layer1_error = np.dot(self.syn_weights1.T, adjustments)
        layer1_adj = layer1_error * self.sigmoid_deriv(layer1)

        self.syn_weights += np.dot(layer1_adj, input_layer.T)
        self.syn_weights1 += np.dot(adjustments, layer1.T)

    def hidden_layer_1(self, input1):
        layer1 = self.sigmoid(np.dot(self.syn_weights, input1))

        return layer1

    def hidden_layer_2(self, input2):
        layer2 = self.sigmoid(np.dot(self.syn_weights1, input2))

        return layer2


def test_database(num):
    input = x_test[num]
    hidden1 = neural_network.hidden_layer_1(input)
    hidden2 = neural_network.hidden_layer_2(hidden1)
    print(hidden2)
    result = np.argmax(hidden2)
    actual_value = y_test[num]
    print(f'Guess: {result}, Real: {actual_value}')
    if result == actual_value:
        return True
    else:
        return False


def test_img(img):
    img = imageio.imread(img, as_gray=True)
    img_data = 255.0 - img.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    hidden1 = neural_network.hidden_layer_1(img_data)
    hidden2 = neural_network.hidden_layer_2(hidden1)
    result = np.argmax(hidden2)
    return result

def extract_images(path,training_ratio=0.8):
    folder = os.listdir(path)
    random.shuffle(folder)
    x_train = []
    x_test = []
    y_test = []
    zeros = np.zeros(((len(folder)), 10)) + 0.01
    for i in range(len(folder)):
        file = str(path + '/' + folder[i])
        ans = int(folder[i][5])
        img = imageio.imread(file, as_gray=True)
        img_data = 255.0 - img.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(img_data.shape)
        if i >= (training_ratio * len(folder)):
            x_test.append(img_data)
            y_test.append(ans)
        else:
            x_train.append(img_data)
    for index, z in enumerate(zeros):
        ans = int(folder[index][5])
        z[ans] = 0.99
    return np.array(x_train), np.array(x_test), zeros[:int(len(folder) * training_ratio)], y_test

# Extract custom 28x28 images from folder to be training:
  # Images must be named accordingly: "train0_.png"
  # Where "0" is the number that's drawn in the image, so the Neural Network can extract the answer for the training data.
  
x_train, x_test, y_train, y_test = extract_images(r'C:\Users\User\PycharmProjects\mnist_ml\Training Data Custom', 0.25)


training_inputs = x_train
normalized_outputs = y_train
total_test_passes = 0
correct_count = 0

neural_network = NeuralNetwork()

print(f'Output: {neural_network.learn(training_inputs, normalized_outputs, 100)}')

print(f'Synaptic weights after training: \n{neural_network.syn_weights}\nSynaptic weights 1 after training: \n{neural_network.syn_weights1}\n\n\n')
for i in range(len(x_test)):
    if test_database(i) == True:
        correct_count += 1
    total_test_passes += 1
print(f'{correct_count}/{total_test_passes}\n\nTest Accuracy: {correct_count / total_test_passes}')

# The following code can be uncommented if you want to save the synaptic weights to be used in the future after the Network is trained:
# with open('Custom_syn_weights.txt', 'w') as file:
#     np.savetxt(file, neural_network.syn_weights)
# with open('Custom_syn_weights1.txt', 'w') as file:
#     np.savetxt(file, neural_network.syn_weights1)
