# link https://keras.io/getting_started/intro_to_keras_for_researchers/
import tensorflow as tf
from tensorflow import keras
"""
in this guide you will learn about:
tensors, variables, and gradients in tensorflow
creating layers by subclassing the layer class
writing low level training loops
tracking losses created by layers via the add_loss() method
tracking metrics in a low level training loop
speeding up execution with a compiled tf.function
executing layers in training or inference mode
the keras functional api
you will also see the keras api in action in two end to end research examples:
a variational autoencoder and a hypernetwork

tensors
tensorflow is an infrastructure layer for differentiable
programming. at its heart, its a framework for manipulating
N-dimensional arrays(tensors) much like Numpy

however there are three key differences between NumPy and TensorFlow

tensorflow can leverage hardware accelerators such as GPU and TPU
tensorflow can automatically compute the gradient of arbitrary differential tensor expressions
tensorflow computation can be distributed to large numbers of devices on single machine
and large number of machines (potentially with multiple devices each).

lets take a look at the object that is at the core of TensorFlow: the Tensor

"""

# here is a constant tensor
x = tf.constant([[5, 2], [1, 3]])
print(x)

"""
tf.Tensor(
[[5 2]
 [1 3]], shape=(2, 2), dtype=int32)

"""
# you can get its value as numpy array by calling .NumPy
x.numpy()
"""
array([[5, 2],
       [1, 3]], dtype=int32)

"""
# much like numpy array, it features the attributes dtype and shape
print("dtype:", x.dtype)
print("shape:", x.shape)
"""
dtype: <dtype: 'int32'>
shape: (2, 2)
"""
# a common way to create tensors via tf.ones and tf.zeros (just like np.ones and np.zeros)
print(tf.ones(shape=(2, 1)))
print(tf.zeros(shape=(2, 1)))

"""
tf.Tensor(
[[1.]
 [1.]], shape=(2, 1), dtype=float32)
tf.Tensor(
[[0.]
 [0.]], shape=(2, 1), dtype=float32)

"""
# you can also create random constant tensors
x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")

# variables
# variables are special tensors used to store mutable state (such as the weights of a neural network)
# you create a variable using some initial value
initial_value = tf.random.normal(shape=(2, 2))
a = tf.Variable(initial_value)
print(a)

"""
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[-1.7639292,  0.4263797],
       [-0.3954156, -0.6072024]], dtype=float32)>

you can update the value of a variable by using the methods .assign(value),
.assign_add(increment) or .assign_sub(decrement)

"""
new_value = tf.random.normal(shape=(2,2))
a.assign(new_value)
for i in range(2):
    for j in range(2):
        assert a[i,j] == new_value[i, j]
added_value = tf.random.normal(shape=(2,2))
a.assign_add(added_value)
for i in range(2):
    for j in range(2):
        asset a[i, j] == new_value[i, j] + added_value[i, j]

# doing math in tensorflow
# if you have used numpy doing math in tensorflow will look very familiar
# the main difference is that your tensorflow code can run on GPU and TPU
a = tf.random.normal(shape=(2,2))
b = tf.random.normal(shape=(2,2))

c = a + b
d = tf.square(c)
e = tf.exp(d)
# gradients: here is another big difference with NumPy
# you can automatically retrieve the graident of any differentiable expressions
# just open a gradient tape, start watching a tensor via tape.watch() and compose a differentiable
# expression using this tensor as input
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
    tape.watch(a) # start recording the history of operations applied to 'a'
    c = tf.sqrt(tf.square(a) + tf.square(b)) # do some math using 'a'
    # what is the gradient of c with respect to a ?
    dc_da = tape.gradient(c, a)
    print(dc_da)

"""
tf.Tensor(
[[ 0.99851996 -0.56305575]
 [-0.99985445 -0.773933  ]], shape=(2, 2), dtype=float32)
"""
# by default, variables are watched automatically so you dont need to manually watch them
a = tf.Variable(a)

with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print(dc_da)
"""
tf.Tensor(
[[ 0.99851996 -0.56305575]
 [-0.99985445 -0.773933  ]], shape=(2, 2), dtype=float32)
"""

# note that you can compute higher order derivatives by nesting tapes
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
    d2c_da2 = outer_tape.gradient(dc_da, a)
    print(d2c_da2)

"""
tf.Tensor(
[[1.2510717e-03 4.4079739e-01]
 [2.1326542e-04 3.7843192e-01]], shape=(2, 2), dtype=float32)

 keras layers
 while tensorflow is an infrastructure layer for differentiable programming
 dealing with tensors, variables and graidents.
 keras is an interface for deep learning dealing with layers, models, optimizers
 loss functions, metrics and more

 keras serves as the high level api for tensorflow. keras is what makes tensorflow simple and productive

 the layer class is the fundamental abstraction in keras. A layer encapsulates a state (weights) and
 some computation (defined in the call method)

 a simple layer looks like this:
"""
class Linear(keras.layers.Layer):
    # y = w.x + b
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(input_dim, units), dtype="float32",
            trainable=True)
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
# you would use a layer instance much like a python function
# instantiate our layer
linear_layer = Linear(units=4, input_dim=2)

# the layer can be treated as a function
# here we call it on some data
y = linear_layer(tf.ones((2, 2)))
assert y.shape == (2, 4)

# the weight variables (created in __init__) are automatically tracked under
# the weights property
assert linear_layer_weights == [linear_layer.w, linear_layer.b]

# you have so many built in layers available from Dense to Conv2D to LSTM to fancier
# ones like Conv3DTranspose or ConvLSTM2D. be smart about reusing built in functionality

# layer weight creation
# the self.add_weight() method gives you a shortcut for creating weights
class Linear(keras.layers.Layer):
    # y = w.x + b

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# instantiate our lazy layer
linear_layer = Linear(4)
# this will also call build(input_shape) and create the weights
y = linear_layer(tf.ones((2, 2)))

# layer gradients
# you can automatically retrieve the gradients of the weights of a layers
# by calling it inside gradient tape. using these gradients, you can update linear_layer_weights
# of the layer, either manually or using an optimizer object. you can modify the apply_gradient before using them if you need to

# prepare a dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)

)

dataset = dataset.shuffle(buffer_size=1024).batch(64)

# instantiate our linear layer (defined above) with 10 units
linear_layer = Linear(10)

# instantiate a logistic loss function that expects integer targets
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# instantiate an optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# iterate over the batches of the dataset
for step, (x, y) in enumerate(dataset):

    # open a gradient tape
    with tf.GradientTape() as tape:
        # forward pass
        logits = linear_layer(x)

        # loss value for this batch
        loss = loss_fn(y, logits)


    # get gradients of the loss wrt the weights
    gradients = tape.gradient(loss, linear_layer.trainable_weights)

    # update the weights of our linear layer
    optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))

    # logging
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
"""
Step: 0 Loss: 2.4605865478515625
Step: 100 Loss: 2.3112568855285645
Step: 200 Loss: 2.1920084953308105
Step: 300 Loss: 2.1255125999450684
Step: 400 Loss: 2.020744562149048
Step: 500 Loss: 2.060229539871216
Step: 600 Loss: 1.9214580059051514
Step: 700 Loss: 1.7613574266433716
Step: 800 Loss: 1.6828575134277344
Step: 900 Loss: 1.6320191621780396


trainable and non trainable weights
weights created by layers can be either trainable or non trainable. they are exposed in
trainable_weights and non_trainable_weights respectively. here is a layer with non trainable weight

"""

class ComputeSum(keras.layers.Layer):
    # returns the sum of the inputs

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # createa non trainable weight
        self.total = tf.Variable(initial_value = tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

my_sum = ComputeSum(2)
x = tf.ones((2, 2))

y = my_sum(x)
print(y.numpy()) # [2. 2.]

y = my_sum(x)
print(y.numpy()) # [4. 4.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []

#[2. 2.]
#[4. 4.]

# layers that own layers
# layers can be recursively nested to create bigger computation blocks. each layer will track
# the weights of its sublayers (both trainable and non trainable)

# lets reuse the linear class
# with a build method that we defined above

class MLP(keras.layers.Layer):
    # simple stack of linear layers

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)

mlp = MLP()

# the first call to the mlp object will create the weights
y = mlp(tf.ones(shape=(3, 64)))

# weights are recursively tracked
assert len(mlp.weights) == 6

# note that our manually created MLP above is equvalent to the following built in optionally
mlp = keras.Sequential(
    [
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(10),
    ]
)

# tracking losses created by layers
# layers can create losses during the forward pass via the add_loss() method.
# this is especially useful for regularization_losses. the losses created by sublayers
# are recursively tracked by the parent layers

# here is a layer that creates an activity regularization loss
class ActivityRegularization(keras.layers.Layer):
    # layer that creates an activity sparsity regularization loss

    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # we use add_loss to create a regularization loss
        # that depends on the inputs
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs


# any model incorporating this layer will track this regularization loss

# lets use the loss layer in the MLP block

class SparseMLP(keras.layers.Layer):
    # stack of linear layers with a sparsity regularization loss
    def __init__(self):
        super(SparseMLP, self).__init__()
        self.linear_1 = Linear(32)
        self.regularization  = ActivityRegularization(1e-2)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.regularization(x)
        return self.linear_3(x)

mlp = SparseMLP()
y = mlp(tf.ones(10, 10))
print(mlp.losses)

# [<tf.Tensor: shape=(), dtype=float32, numpy=0.21796302>]
# these losses are cleared by the top level layer at the start of each forward pass
# they dont accumulate. layer.losses always contain only the losses created during the last forward pass
# you would typically use these losses by summing them before computing your gradients when writing a training loop

# losses correspond to the last forward pass
mlp = SparseMLP()
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1 # no accumulation

# lets demontrate how to use these losses in a training loop
# prepare a dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)

)

dataset = dataset.shuffle(buffer_size=1024).batch(64)

# a new mlp
mlp = SparseMLP()
