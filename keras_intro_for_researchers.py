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

# Loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:
         # forward pass
         logits = mlp(x)
         # external loss value for this batch
         loss = loss_fn(y, logits)
         # add the losses created during the forward pass
         loss += sum(mlp.losses)
         # get gradients of the loss wrt the weights
         gradients = tape.gradient(loss, mlp.trainable_weights)


    # update the weights of our linear layer
    optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

    # logging
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))

"""
Step: 0 Loss: 6.307978630065918
Step: 100 Loss: 2.5283541679382324
Step: 200 Loss: 2.4068050384521484
Step: 300 Loss: 2.3749840259552
Step: 400 Loss: 2.34563946723938
Step: 500 Loss: 2.3380157947540283
Step: 600 Loss: 2.3201656341552734
Step: 700 Loss: 2.3250539302825928
Step: 800 Loss: 2.344613790512085
Step: 900 Loss: 2.3183579444885254

keeping track of training metrics

keras offers a broad range of built in metrics like tf.keras.metrics.AUC or tf.keras.metrics.PrecisionAtRecall

it is also easy to create your own metrics in a few lines of code

to use a metric in a custom training loop, you would

instantiate the metric object eg metric = tf.keras.metrics.AUC()
call its metric.update_state(targets, predictions) method for each batch of data
query its result via metric.result()
reset the metric's state at the end of an epoch or at the start of an evaluation via metric.reset_state()

simple example
"""

# instantiate a metric object
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# prepare our layer, loss, and optimizer
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10)
    ]
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

for epoch in range(2):
    # iterate over the batches of a dataset
    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(x)
            # compute the loss value for this batch
            loss_value = loss_fn(y, logits)

        # update the state of the accuracy metric
        accuracy.update_state(y, logits)

        # update the weights of the model to mimimize the loss value
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # logging the current accuracy value so far
        if step % 200 == 0:
            print("Epoch:", epoch, "Step:", step)
            print("Total running accuracy so far: %.3f" % accuracy.result())


     # reset the metrics state at the end of an epoch
     accuracy.reset_state()

"""
Epoch: 0 Step: 0
Total running accuracy so far: 0.141
Epoch: 0 Step: 200
Total running accuracy so far: 0.751
Epoch: 0 Step: 400
Total running accuracy so far: 0.827
Epoch: 0 Step: 600
Total running accuracy so far: 0.859
Epoch: 0 Step: 800
Total running accuracy so far: 0.876
Epoch: 1 Step: 0
Total running accuracy so far: 0.938
Epoch: 1 Step: 200
Total running accuracy so far: 0.944
Epoch: 1 Step: 400
Total running accuracy so far: 0.944
Epoch: 1 Step: 600
Total running accuracy so far: 0.945
Epoch: 1 Step: 800
Total running accuracy so far: 0.945

in addition to this, similary to the self.add_loss() method, you have access to an self.add_metric()
method on layers. it tracks the average of whatever quantity you pass to it. you can reset the value
of these metrics by calling layer.reset_metrics() on any layer or model.

you can also define your own metrics by subclassing keras.metrics.Metric you need to over ride the three
functions called above

override update_state() to update the statistics values
override result() to return the metric value
overrise reset_state() to reset the metric to its initial state

here is an example where we implement the F1-score metric (with support for sample weighting)
"""
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", dtype="float32", threshold=0.5, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = 0.5
        self.true_positives = self.add_weight(name="tp", dtype=dtype, initializer="zeros")
        self.false_positives = self.add_weight(name="fp", dtype=dtype, initializer="zeros")
        self.false_negatives = self.add_Weight(name="fn", dtype=dtype, initializer="zeros")


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.greater_equal(y_pred, self.threshold)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.cast(y_true & y_pred, self.dtype)
        false_positives = tf.cast(~y_true & y_pred, self.dtype)
        false negatives = tf.cast(y_true & ~y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            true_positives *= sample_weight
            false_positives *= sample_weight
            false_negatives *= sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return precision * recall * 2.0 / (precision + recall)

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


m = F1Score()
m.update_state([0, 1, 0, 0], [0.3, 0.5, 0.8, 0.9])
print("Intermediate result:", float(m.result()))

m.update_state([1, 1, 1, 1], [0.1,0.7, 0.6, 0.0])
print("Final result:", float(m.result()))

"""
Intermediate result: 0.5
Final result: 0.6000000238418579

Compiled functions

running eagerly is great for debuggging, but you will get better performance by
compiling your computations into static graphs. you can compile any function
by wrapping it in a tf.function decorator

"""

# prepare the layer, loss and optimizer
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10)
    ]
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# create a training step function
@tf.function # make it fast
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# prepare a dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)

dataset = dataset.shuffle(buffer_size=1024).batch(64)

for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, y)
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))

"""
Step: 0 Loss: 2.291861057281494
Step: 100 Loss: 0.5378965735435486
Step: 200 Loss: 0.48008084297180176
Step: 300 Loss: 0.3359006941318512
Step: 400 Loss: 0.28147661685943604
Step: 500 Loss: 0.31419697403907776
Step: 600 Loss: 0.2735794484615326
Step: 700 Loss: 0.3001103401184082
Step: 800 Loss: 0.18827161192893982
Step: 900 Loss: 0.15798673033714294

training mode and inference mode

some layers in particular the batch normalization layer and the drop out layer
have different behaviours during training and inference. for such layers, it is
standard practice to expose a training boolean argument in the call method

by exposing this argument in call, you enable the built in training and evaluation
loops (eg fit) to correctly use the layer in training and inference modes
"""

class Dropout(keras.layers.Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

class MLPwithDropout(keras.layers.Layer):
    def __init__(self):
        super(MLPwithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training=None):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return self.linear_3(x)

mlp = MLPwithDropout()
y_train = mlp(tf.ones((2, 2)), training=True)
y_test = mlp(tf.ones((2, 2)), training=False)

"""
the functional api for model building
to build deep learning models you dont have to use object oriented programming
all the time.

all layers we have seen can also be composed functionally

we call it the functional api

you can use an input object to describe the shape and dtype of inputs
this is the deep learning equivalent of declaring a type
the shape argument is per sample, it doesnt include batch size
the functional api is focused on defining per sample transformations
the model we create will automatically batch the per sample transformations
so that it can be called on batches of data
"""

inputs = tf.keras.Input(shape=(16,), dtype="float32")

# we call layers on these type objects
# and they return updated types (new shapes / dtypes)
x = Linear(32)(inputs) # we are re-using the linear layer we defined earlier
# we are re-using the drop out layer we defined earlier
x = Dropout(0.5)(x)
outputs = Linear(10)(x)

# a functional model can be defined by specifying inputs and outputs
# a model is itself a layer like any other
model = tf.keras.Model(inputs, outputs)

# a functional model already has weights, before being called on any data
# that is because we defined its input shape in advance (in Input)
assert len(model.weights) == 4

# lets call our model on some data
y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)

# you can pass a training argument in call
# it will get passed down to the drop out layer
y = model(tf.ones((2, 16)), training=True)

"""
the functional api tends to be more concise than subclassing and provides
a few other advantages ( generally the same advantages that functional typed
languages provide over untyped oo developed) it can be only to defined DAGs of
layers recursive networks should be defined as layer subclasses instead

in your research workflows you may find yourself mix and matching oo models with functional models

note that the model class also features built in training & evaluation loops: fit(), predict()
and evaluate() configured via the compile() method. these built in functions give you
access to the following built in training infrastructure features

call backs you can leverage built in callbacks for early stopping, model check checkpointing
and monitoring training with tensor board. you can also implement custom call callbacks

distributed training, you can easily scale up your training to multiple gpus, tpu or multiple machines
with tf.distribute api with no changes to your code
step fusing, with the steps_per_execution argument in model.compile() you can process
multiple batches in a single tf.function call, which greatly improves device utilization on tpu

"""
inputs = tf.keras.Input(shape=(784,), dtype="float32")
x = keras.layers.Dense(32, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)

# specifying the loss, optimizer, and metrics with compile()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
)

# train the model with the dataset for 2 epochs
model.fit(dataset, epochs=2)
model.predict(dataset)
model.evaluate(dataset)

"""
Epoch 1/2
938/938 [==============================] - 1s 1ms/step - loss: 0.3958 - sparse_categorical_accuracy: 0.8872
Epoch 2/2
938/938 [==============================] - 1s 1ms/step - loss: 0.1916 - sparse_categorical_accuracy: 0.9447
938/938 [==============================] - 1s 798us/step - loss: 0.1729 - sparse_categorical_accuracy: 0.9485

[0.1728748232126236, 0.9484500288963318]


you can always subclass the model class (it works like how the subclassing layer works)

if you want to leverage built in training loops for you OO models, just override the model.train_step()
to customize what happens in fit() while retaining support for th built in infrastructure
"""
