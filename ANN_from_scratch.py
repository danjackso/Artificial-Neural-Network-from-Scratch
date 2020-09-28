import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]).reshape((4, 3))
y = np.array([0, 1, 1, 0]).reshape((4, 1))


def sigmoid(x): return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        print('Building model')
        self.layers = 0
        self.layer_list = []
        self.weights_list = []
        self.units = []

    def add_input(self,shape):
        self.input=shape

    def add_output(self,shape):
        self.output_layer=shape
        self.output_weights=np.random.rand(self.units[-1], shape)

    def add_hidden_layer(self, units):
        layer, weights, loop = 'layer1', 'weights1', 1
        while True:
            if layer not in self.__dict__:
                self.layers += 1
                self.units.append(units)
                self.layer_list.append(layer)
                self.weights_list.append(weights)
                break
            loop += 1
            layer = 'layer' + str(loop)
            weights = 'weights' + str(loop)


        input_shape = self.input if loop == 1 else np.array(getattr(self, 'layer' + str(loop - 1))).shape[1]
        setattr(self, 'weights' + str(loop), np.random.normal(0,1,size=(input_shape, units))*math.sqrt(1/input_shape))

        input = self.input if loop == 1 else getattr(self, 'layer' + str(loop - 1))
        setattr(self, layer, sigmoid(np.dot(input, getattr(self, weights))))

    def generate_architecture(self):
        global plot_units, max_units, coordinates
        plot_units = self.units
        plot_units.insert(0,self.input)
        plot_units.append(self.output_layer)
        max_units = max(plot_units)

        coordinates = []
        for idx, item in enumerate(plot_units, start=1):
            if item == max_units:
                array = np.concatenate(
                    [np.ones((item, 1)) * idx, np.linspace(0, max_units - 1, max_units).reshape((item, 1))], axis=1)
            else:
                array = np.concatenate([np.ones((item, 1)) * idx,
                                        np.linspace((max_units - item) / 2, max_units - 1 - (max_units - item) / 2,
                                                    item).reshape((item, 1))], axis=1)
            coordinates.append(array)

    def plot_initial_architecture(self):
        global plot_units, max_units, coordinates
        # Plot Nodes
        plt.figure(figsize=(8, 4.5))
        for item in coordinates:
            plt.scatter(item[:, 0], item[:, 1], s=100, color='black', zorder=2)

        # Plot Weights
        for item in range(1, len(coordinates)):
            perms = list(itertools.product(coordinates[item - 1], coordinates[item]))
            for item in perms:
                plt.plot([item[0][0], item[1][0]], [item[0][1], item[1][1]], linewidth=0.75, zorder=1)

        plt.axis('off')
        plt.xlim([0, len(plot_units) + 1])
        plt.show()

    def plot_live_architecture(self,weight_update):
        global plot_units, max_units, coordinates
        plt.cla()

        # Plot Nodes
        for item in coordinates:
            plt.scatter(item[:, 0], item[:, 1], s=100, color='black', zorder=2)

        for item in range(1, len(coordinates)):
            perms = list(itertools.product(coordinates[item - 1], coordinates[item]))
            delta = weight_update[::-1][item-1].ravel()
            for idx,item in enumerate(perms):
                plt.plot([item[0][0], item[1][0]], [item[0][1], item[1][1]], linewidth=0.75, zorder=1,color='red' if delta[idx]<0 else 'limegreen')

        plt.axis('off')
        plt.xlim([0, len(plot_units) + 1])

    def feedforward(self):
        previous_layer = ''
        for idx, Tuple in enumerate(zip(self.weights_list, self.layer_list)):
            weight, layer = Tuple
            if previous_layer == '':
                setattr(self, layer, sigmoid(np.dot(self.x, getattr(self, weight))))
            else:
                setattr(self, layer, sigmoid(np.dot(getattr(self, previous_layer), getattr(self, weight))))
            previous_layer = layer
        self.output = sigmoid(np.dot(getattr(self, previous_layer), self.output_weights))

    def backprop(self,plot=Flase):
        d_weight_list=[]
        loss=2 * (self.y - self.output)

        d_weight_list.append(np.dot(getattr(self,self.layer_list[-1]).T,loss*sigmoid_derivative(self.output)))
        previous_loss=loss*sigmoid_derivative(self.output)
        previous_weight='output_weights'
        previous_layer=self.layer_list[-1]

        for idx in range(len(self.weights_list)-1,-1,-1):
            if idx==0:
                loss=np.dot(previous_loss,getattr(self,previous_weight).T)
                d_weight_list.append(np.dot(self.x.T,loss*sigmoid_derivative(getattr(self,previous_layer))))
            else:
                loss=np.dot(previous_loss,getattr(self,previous_weight).T)
                d_weight_list.append(np.dot(getattr(self,self.layer_list[idx-1]).T,loss*sigmoid_derivative(getattr(self,previous_layer))))
                previous_loss=loss*sigmoid_derivative(getattr(self,previous_layer))
                previous_weight=self.weights_list[idx]
                previous_layer=self.layer_list[idx-1]

        self.output_weights=self.output_weights+d_weight_list[0]
        for item, weight in zip(d_weight_list[1:],self.weights_list[::-1]):
            new_weight=getattr(self,weight)+(item)
            setattr(self,weight,new_weight)

        if plot==True:
            self.plot_live_architecture(d_weight_list)
            plt.pause(1e-03)

    def loss(self):
        return np.mean((self.y - self.output) ** 2)

    def fit(self, x, y, epochs=10000, verbose=1,plot=False):
        self.x=x
        self.y=y
        self.output=np.zeros(self.y.shape)

        self.loss_array=np.zeros((epochs,1))
        for loop in range(epochs):
            self.feedforward()
            self.backprop(plot=plot)
            self.loss_array[loop]=self.loss()
            if verbose == 1 and loop%100==0:
                print('Epoch: {}    loss: {:.4f}'.format(loop + 1, self.loss()))
            else:
                pass

    def plot_loss(self):
        plt.plot(model.loss_array,linewidth=0.75)
        plt.title('Loss (RMSE) vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(ymin=0)
        plt.grid(True)
        plt.show()

model = NeuralNetwork()
model.add_input(3)
model.add_hidden_layer(8)
model.add_hidden_layer(16)
model.add_hidden_layer(8)
model.add_hidden_layer(8)
model.add_hidden_layer(8)
model.add_output(1)
model.generate_architecture()
