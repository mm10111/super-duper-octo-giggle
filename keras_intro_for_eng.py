#link to tutorial: https://keras.io/getting_started/intro_to_keras_for_engineers/

# importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

####
"""
in this guide you will learn to prepare your dataset before training a model
by either turning it into Numpy arrays or a tensorflow dataset object (tf.data.Dataset objects)

- do data preprocessing, for instance feature normalization or vocabulary indexing.
- build your model that turns the data into useful predictions using the keras functional API
- train your model with the built in keras fit () method while being mindful of chckingpointing, metrics monitoring and fault tolerance
- evaluate your model on test data and how to use it for inference on new dataset
- customize what fit() does, for instanceto build a GAN
- speed up training by leveraging multiple GPUs
- refine your model through hyperparameter tuning.

Data loading and preprocessing

neural networks dont process raw data like text files, images, csv files,
they process vectorized and standardized representations

text files need to be read into string tensors, then split into words, finally the words need to
need to be indexed and turned into integer tensors

images need to be read and decoded into integer tensors then converted to float
point and normalized to small values usually between 0 and 1

csv data needs to be parsed with numerical features converted to floating chckingpointing
tensors  and categorical features indexed and converted to integer tensors
then each feature typically needs to be normalized to zero mean and unit variance

Data loading

numpy arrays just like with scikit learn and many other python based libraries, this is a good option if your data fits in memory

tensorflow dataset objects: this is a high performance option that is more suitable for datasets
that dont fit in memory and are streaming from disk or from a distributed file system

python generators: that yield batches of data (such as custom subclasses of the keras.utils.Sequence class)

before you start training your model, you will need to make the data available
in one of these formats and if you have a large dataset and you are using a GPU
consider using the dataset objects they will take care of performance critical
details such as:
 asynchronously preprocessing your data on cpu while gpu is busy
 and buffering it to queue
 prefetching data on gpu memory so immediately available when the gpu is
 finished processing the previous batch so you can reach full gpu utilization

 keras features a range of utilities to help you turn raw data on disk into a dataset
 tf.keras.preprocessing.image_dataset_from_directory turns image files
 sorted into class specific folders into a labeled dataset of image tensors
 tf.keras.preprocessing.text_dataset_from_directory does the same for text files

 in addition the tensorflow tf.data includes other similar utilities such as tf.data.experimental.make_csv_dataset
 to load structured data from csv files
 tf.data.experimental.make_csv_dataset to load structured data from csv files.

 suppose you have image fils sortd by class in different folders like this

 run example

 main_directory/
 ...class_a/
 ........a_image_1.jpg
  ........a_image_2.jpg
 ...class_b/
  ........b_image_1.jpg
   ........b_image_2.jpg
   then you can do

"""

# create a dataset
dataset = keras.preprocessing.image_dataset_from_directory(
    'path/to/main_directory', batch_size=64, image_size=(200,200))

# for demonstration purposes iterate over the batches yielded by the datasets
for data, labels in dataset:
    print(data.shape) # (64, 200, 200, 3)
    print(data.dtype) # float32
    print(labels.shape) #(64,)
    print(labels.dtype) # int32
"""
the label of the sample is the rank of its folders in the alphanumeric order,
naturally this can also be configured explicitly by passing
eg class_names=['class_a', 'class_b'] in which cases label 0 will be class_a
and 1 will be class_b

eg obtaining a labeled dataset from textfiles on disk
for text. if you have .txt documents sorted by class in different folders, you
can do:

"""
dataset = keras.preprocessing.text_dataset_from_directory('path/to/main_directory', batch_size=64)

# for demo purposes, iterate over the batches yielded by the dataset
for data, labels in dataset:
    print(data.shape) #(64,)
    print(data.dtype) # string
    print(labels.shape) #(64,)
    print(labels.dtype) #int32

"""
data preprocessing with keras
once your data is in the form of string/int/float numpy arrays or a dataset
object (or python generator) that provides string/int/float tensors, it is time to
preprocess the data

it can mean
tokenization of string data, followed by token indexing
feature normalization
rescaling the data to small values(in general, input values to a neural network
should be close to zero- typically we expect either data with zero mean and unit
variance or data in the [0.1] range)

the ideal machine learning model is end to end

in general you should seek to do data preprocessing as part of your model,
not via external data preprocessing pipeline because external data preprocessing
makes your model less portable when it is time to use them in production

consider a model that processes text, it uses a specific tokenization algorithm
and a specific vocabulary inex. when you want to ship your model to mobile app
or a javascript app, you will need to recreate the exact same preprocessing
setup in the target language. this can get very tricky: any small discrepancy
between the original pipeline and the one you recreate has the potential to
completely invalidate your model, or atleast severely degrade its performance.

it would be much easier to simply export an end to end model that already
includes preprocessing. the ideal model should expect a input something as close
as possible to raw data: an image model should expect RGB pixel values in the [0
, 255] range and a text model should accept strings of utf-8 characters. that
way the consumer of the exported model doesnt have to know about the preprocessing
pipeline.

using keras preprocessing layers
in keras, you do in model data preprocessing via preprocessing layers.
this includes: vectorizing raw strings of text via the text TextVectorization layer
feature normalization via the Normalization layer.
image rescaling, cropping or image data augmentation.

the key advantage of using keras preprocessing layers that they can be included
directly into your model, either during training or after training which makes
your models portable.

some preprocessing layers have a state:
TextVectorization holds an index mapping words or tokens to integer indices
Normalization holds the mean and variance of your features.

The state of a preprocessing layer is obtained by calling layer.adapt(data) on a
sample of the training data (or all of it).
"""
# Example: turning strings into sequences of integer word indices
from tensorflow.keras.layers import TextVectorization

# Example training data, of dtype string
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# create a TextVectorization layer instance
# it can be configured to either return integer token indices or a dense token
# representation (eg multi-hot or TF-IDF). the text standardization and text
# splitting algorithms are fully configurable
vectorizer = TextVectorization(output_mode="int")

#calling 'adapt' on an array or dataset makes the layer generate a vocabulary
# index for the data, which can then be reused when seeing new data
vectorizer.adapt(training_data)
# after calling adapt, the layer is able to encode any n-gram it has seen before
# in the 'adapt()' data. unknown n-grams are encoded via an "out-of-vocabulary"
#token.
integer_data = vectorizer(training_data)
print(integer_data)

"""
run example

output
tf.Tensor(
[[4 5 2 9 3]
 [7 6 2 8 3]], shape=(2, 5), dtype=int64)
"""
# example turning strings into sequences of one hot encoded bigrams
# following the code from above just changing the vectorizer variable
vectorizer = TextVectorization(output_mode="binary", ngrams=2)
# repeating previous steps
vectorizer.adapt(training_data)
integer_data = vectorizer(training_data)
print(integer_data)

"""
run example

output
tf.Tensor(
[[0. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1.]
 [0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0.]], shape=(2, 17), dtype=float32)

"""
# example normalizing features
from tensorflow.keras.layers import Normalization
# example image data with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)
normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

"""
run example
output
var: 1.0000
mean: -0.0000


example rescaling and centre cropping images
both the rescaling layer and the centrecrop layer are stateless so it isn't
necessary to call adapt() in this case

"""

from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

# example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CentreCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)
output_data = scaler(cropper(training_data))
print("Shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))

"""
run example
output
shape: (64, 150, 150, 3)
min: 0.0
max: 1.0

building models with the Keras functional api
a layer is a simple input output transformation such as a the scaling and centre
cropping transformation above. for instance here is a linear projection layer
that maps its inputs to 16 dimensional feature space
dense = keras.layers.Dense(units=16)

a model is a directed acyclic graph of layers. you can think of a model as a
bigger layer that encompasses multiple sublayer and that can be trained via
exposure to data
the most common and most powerful way to build keras models in the functional API
to build models with the functional api, you start by specifying the shape and
optionally the dtype of your inputs. if any dimension of your input can vary,
you specifyc it as None. For instance, an input 200 x 200 RGB image would have
shape (200, 200, 3) but an input for RBG images of any size would have shape
(None, None, 3)

lets say we expect our inputs to be RGB images of arbitrary size
inputs = keras.Input(shape=(None, None, 3))
"""
# After defining your inputs, you can chain layer transformations on top of your
# inputs until your final output

from tensorflow.keras import layers

# Centre crop images to 150 x 150
x = CentreCrop(height=150, width=150)(inputs)
# rescale images to [0,1]
x = Rescaling(scale=1.0 / 255)(x)

# apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)


# apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifer on top
num_classes = 10
outputs= layers.Dense(num_classes, activation="softmax")(x)

# once you have defined the directed acyclic graph of layers that turns your
# inputs into your outputs, instantiate a model object
model = keras.Model(inputs=inputs, outputs=outputs)

# this model behaves like a bigger layer. you can call it on the batches of data
data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data.shape)
# output: (64, 10)
# you can print a summary of how your data gets transformed at each stage of your
# model. this is useful for debugging.
# note that the output shape displayed for each layer includes the batch size.
# here the batch size is None which indicates our model can process batches of
# any size
model.summary()
"""
run example
output

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, None, None, 3)]   0
_________________________________________________________________
center_crop_1 (CenterCrop)   (None, 150, 150, 3)       0
_________________________________________________________________
rescaling_1 (Rescaling)      (None, 150, 150, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 148, 148, 32)      896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 32)        9248
_________________________________________________________________
global_average_pooling2d (Gl (None, 32)                0
_________________________________________________________________
dense (Dense)                (None, 10)                330
=================================================================
Total params: 19,722
Trainable params: 19,722
Non-trainable params: 0
_________________________________________________________________

the functional api makes it easy to build models that have multiple inputs (eg
an image and its meta data or multiple output (predicting the class of the image
and the likelihood that a user will click on it). for a deeper dive into what you
can do see our guide to functional api

Training models with fit()
at this point you know:
how to prepare your data (eg as NumPy array or a tf.data.Dataset object)
how to build a model that will process your data.

the next step is to train your model on your data. The Model class features a
built in training loop, the fit() method. It accepts Datasets objects, Python
generators that yield batches of data or NumPy arrays.

Before you can call fit(), you need to specify an optimizer and a loss function
(we assume you are already familiar with these concepts). this is the compile()
step:
"""
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss=keras.losses.CategoricalCrossentropy())
# loss and optimizer can be specified via their string identifiers (in this case
# default constructor argument values are used):
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# once your model is compiled, you can start fitting the model to the data
# here is what fitting a model looks like with the NumPy data
model.fit(numpy_array_of_samples, numpy_array_of_labels,
          batch_size=32, epochs=10)
"""
besides the data you have to specify two key parameters: the batch_size and the
number of epochs(iterations on the data). here our data will get sliced on batches
of 32 samples, and the model will iterate 10 times over the data during training.

here is what fitting a model looks like with a dataset:
"""
model.fit(dataset_of_samples_and_labels, epochs=10)

# since the data yielded by a dataset is expected to be already batched, you dont need to specify
# the batch size here, lets look at it in practice with a toy example model that learns to classify MNIST digits:

# get the data as numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# build a simple model
inputs = keras.Input(shape=(28, 28))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()
# compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# train the model for 1 epoch from numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# train the model for 1 epoch using a dataset
dataset = tf.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(dataset, epochs=1)

"""
run example
output

Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         [(None, 28, 28)]          0
_________________________________________________________________
rescaling_2 (Rescaling)      (None, 28, 28)            0
_________________________________________________________________
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               100480
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
_________________________________________________________________
Fit on NumPy data
938/938 [==============================] - 1s 940us/step - loss: 0.4771
Fit on Dataset
938/938 [==============================] - 1s 942us/step - loss: 0.1138

------

the fit() call returns a history object which records what happened over the course of training.
the history.history dict contains per epoch time series of metric values (here we have only one metric
, the loss and one epoch so we only get a single scalar)

"""
print(history.history)

" run example  output: {'loss': [0.11384169012308121]}"

"""
for a detailed overview of how to use fit(), see the guide to training & evaluation with the built-in
keras methods

keeping track of performance metrics

as you are training your model, you want to keep track of metrics such as classification accuracy,
precision, recall, AUC etc, besides you want to monitor these metrics not only on the training data but also on the validation set

monitoring metrics
you can pass a list of metric objects to compile() like this:
"""
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(dataset, epochs=1)

# 938/938 [==============================] - 1s 929us/step - loss: 0.0835 - acc: 0.9748

"""
passing validation data to fit()
you can pass validation data to fit() to monitor your validtion loss and validation metrics. validation
metrics get reported at the end of each epoch
"""
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=1, validation_data=val_dataset)
# 938/938 [==============================] - 1s 1ms/step - loss: 0.0563 - acc: 0.9829 - val_loss: 0.1041 - val_acc: 0.9692
"""
using callbacks for checkpointing and more
if training goes on for more than a few minutes, it is important to save your model at regular intervals during training
you can then use your saved models to restart training in case your training process crashes (this is important for multiworker
distributed training since with many workers atleast one of them is bound to fail at some point).

an important feature of keras is callbacks, configured in fit(). callbacks are objects that get called by the model
at different point during training, in particular:
- at the beginning and end of each batch
- at the beginning and end of each epoch
call backs are a way to make model trainable entirely scriptable
you can use call backs to periodically save your model.
here is a simple example a ModelCheckPoint callback configured to save the model at the end of every epoch.
the filename will include the current epoch

"""
callbacks = [
    keras.callbacks.ModelCheckPoint(
        filepath='path/to/my/model_{epoch}'
        save_freq='epoch')
]
model.fit(dataset, epochs=2, callbacks=callbacks)

"""
you can also use call backs to do things like periodically changing the learning of your optimizer,
streaming metrics to a slack bot, sending yourself an email notification when training is complete etc.

for detailed overview of what call backs are available and how to write your own, see the callbacks api
documentation and the guide to writing custom callbacks

monitoring training progress with tensor board
staring at the keras progress bar isnt the most ergonomic way to monitor how your loss and metrics are evolving
over time. there is a beter solution: tensor board a web application that can display real time graphs of your metrics and more

to use tensorboard with fit(), simply pass a keras.callbacks.tensorBoard callback specifying the main_directory
where to store TensorBoard logs
"""
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')

]
model.fit(dataset, epochs=2, callbacks=callbacks)

# you can then launch a tensorboard instance that you can open in your browser to monitor logs getting written to this location
# tensorboard --logdir=./logs
# you can launch an inline tensorboard tab when trailing models in jupyter notebooks
# after fit() evaluating test performance and generating predictions on new data
# once you have trained model, you can evaluate its loss and metrics on new data via evaluate()
loss, acc = model.evaluate(val_dataset) # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

"""
run example

157/157 [==============================] - 0s 688us/step - loss: 0.1041 - acc: 0.9692
loss: 0.10
acc: 0.97
"""
# you can also generate numpy arrays of predictions (the activations of the output layers in the model via predict())
predictions = model.predict(val_dataset)
print(predictions.shape)
# (10000, 10)
"""
using fit() with a custom training step
by default fit() is configured for supervised learning, if you need a different kind of training loop
for instance a GAN training loop, you can provide your own implementation of the Model.train_step() method.

this is the method that is repeatedly called during fit()
metrics, callbacks, etc will work as usual

here is a simgple example that reimplements what fit() normally does:

"""
class CustomModel(keras.Model):
    def train_step(self, data):
        # unpack the data, its structure depends on your model and what you pass fit on
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # forward pass
            # compute the loss value
            # the loss function is configured in compile()
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

# construct and compile an instance of custom model
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=[...])

# just use fit as usual
model.fit(dataset, epochs=3, callbacks=...)

# for detailed overview of how you customize the built in training and evaluation loops see the guide: customizing what happens in fit()

"""
debugging your model with eager execution
if you write custom training steps or custom layers you will need to debug them the debugging experience is an integral part of a fraemwork with keras the debugging workflow is designed with the user in mind. by default your keras models are compiled to highly optimized computation graphs that deliver fast execution times.
that means that the pythong code you write (eg in a custom train_step) is not the code you are actually executing.
this introduces a layer of indirection that can make debugging hard.
debugging is best done step by step. you want to be able to sprinkle your code with print()
statement to see what your data looks like after every operation, you want to be able to use pdb. you can achieve this by running your model eagerly. with eager execution the python code you write is the code that gets executed
simply pass run_eagerly=True to compile()

model.compile(optimizer='adam', loss='mse', run_eagerly=True)

the downside is that it makes your model significantly slower. make sure to switch it off to get the benefits of compiled computation graphs once you are done debugging

in general you will use run_eagerly=True everytime you need to debug whats happening inside your fit() call.

speeding up training with multiple GPUs
keras has built in industry strength support for multi gpu training and distributed multiworker training via the tf.distribute API

if you have multiple GPUs on your machine, you can train your model on all of them by:
    creating a tf.distribute.MirroredStrategy object
    building & compiling your model inside the strategy's scope
    calling fit() and evaluate() on a dataset as usual
"""
# create a mirrored strategy
strategy = tf.distribute.MirroredStrategy()

# open a strategy scope
with strategy.scope():
    # everything that creates variables should be under the strategy scope
    # in general this is only model construction and compile()
    model = Model(...)
    model.compile(...)

# train the model on all available devices
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# test the model on all available devices
model.evaluate(test_dataset)
# for a detailed introduction to multi grpu and distributed training, see the guide

"""
doing preprocessing synchoronously on device vs asynchronosuly on host cpu
you have learnt about preprocessing and you have seen example where we put image preprocessing layers(CenterCrop and Rescaling) directly inside our model.

having preprocessing happen as part of the model during training is great if you want to do on device preprocessing, for instance GPU accelerated feature normalization or image augmentation.
but there are kinds of preprocessing that are not suited to this setup: in particular, text preprocessing with the TextVectorization layer.
Due to its sequential nature and due to the fact that it can only run on CPU, its often a good idea to do asynchronous preprocessing.

with asynchronous preprocessing, your preprocessing operations will run on CPU, and the preprocessed samples will be buffered into a queue while your GPU is
busy with previous batch of data. The next batch of preprocessed samples will then be fethced from the queue to the GPU memory right before the GPU becomes
available again (prefetching). This ensures that preprocessing will not be blocking and that your GPU can run at full utilization.

to do asynchronous preprocessing, simply use dataset.map to inject a preprocessing operation into your data pipeline.
"""
# example training data, of dtype 'string'
samples = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])
labels = [[0], [1]]

# prepare a textvectorization layer
vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(samples)

# asynchronous preprocessing: the text vectorization is part of the tf.data.pipeline
# first, create a dataset
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
# apply text vectorization to the samples
dataset = dataset.map(lambda x, y: (vectorizer(x), y))
# prefetech with a buffer size of 2 batches
dataset = dataset.prefetch(2)

# our model should expect sequences of integers as inputs
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(input_dim=10, output_dim=32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)

# 1/1 [==============================] - 0s 13ms/step - loss: 0.5028
# <tensorflow.python.keras.callbacks.History at 0x147777490>

# compare this to doing text vectorization as part of the model

# our dataset will yield samples that are strings
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)

# our model should expect strings as inputs
inputs = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inputs)
x = layers.Embedding(input_dim=10, output_dim=32)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)

# 1/1 [==============================] - 0s 16ms/step - loss: 0.5258
# <tensorflow.python.keras.callbacks.History at 0x1477b1910>

"""
when training text models on CPU, you will generally not see any performance difference between the two setups.
when training on GPU, however doing asynchronous buffered preprocessing on he host CPU while the GPU is running the model itself
can result in a significant speedup.

after training, if you want to export an end to end model that includes the preprocessing layer(s), this is easy to do since TextVectorization is a layer

"""
inputs = keras.Input(shape=(1,), dtype='string')
x = vectorizer(inputs)
outputs = trained_model(x)
end_to_end_model = keras.Model(inputs, outputs)
"""
finding the best model configuration with hyperparameter tuning.
once you have a working model, you are going to optimize its configuration - architecture choices, layer sizes
you will want to leverage systematic approach: hyperparameter search

you can use KerasTuner to find the best hyperparameter for your keras models. it is as easy as calling fit()

first place your model definition in a function that takes a single hp argument. inside this function replace any value you want to tune with a call
to hyperparameter sampling methods eg hp.Int() or hp.Choice()

"""

def build_model(hp):
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation = 'relu')) (inputs)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                        values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# the function should return a compiled model
# next instantiate a tuner object specifcying your optimization objective and other search parameters

import keras_tuner

tuner = keras_tuner.tuners.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=100,
    max_trials=200,
    executions_per_trial=2,
    directory='my_dir')

# finally start the search with the search() method which takes the same arguments as model.fit()
tuner.search(dataset, validation_data=val_dataset)
# when search is over you can retrieve the best models
models = tuner.get_best_models(num_models=2)
# or print a summary of the results
tuner.results_summary()
