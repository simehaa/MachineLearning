# Animations

All of the following neural networks have been trained on a franke function data
set without noise and with 30,000 data points. The parameters were
* Two hidden layers consisted of 100 and 60 neurons respectively, and with different activation functions.
* Learning rate was set to 0.01.
* L2 regularization parameter was set to 0.01.
* Number of epochs was set to 100.
* The activation function of the output layer was linear.

The intention was to use different activation functions for the two hidden layers, and to see how
* *ReLU*,
* *tanh* ,
* *Sigmoid*,

evolves during the epochs.

### Sigmoid
![Sigmoid](franke_sigmoid.gif)
### tanh (hyperbolic tangent)
![tanh](franke_tanh.gif)
### ReLU (rectified linear unit)
![ReLU](franke_relu.gif)
### ReLU6
![ReLU6](franke_relu6.gif)
### Leaky ReLU
![LReLU](franke_l_relu.gif)
