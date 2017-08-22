import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot = True)

nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500

classes = 10
batch_size = 100


x = tf.placeholder("float32",[None,784])
y = tf.placeholder("float32")

def neural_model(data):
    #create dynamic layers.Neww properties Tensorflow 1.0
    nodes_hidden1 = {"weights":tf.Variable(tf.random_normal([784,nodes_layer1])),
                     "biases":tf.Variable(tf.random_normal([nodes_layer1]))}

    nodes_hidden2 = {"weights":tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2])),
                     "biases":tf.Variable(tf.random_normal([nodes_layer2]))}

    nodes_hidden3 = {"weights":tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3])),
                     "biases":tf.Variable(tf.random_normal([nodes_layer3]))}

    output_layer = {"weights":tf.Variable(tf.random_normal([nodes_layer3,classes])),
                     "biases":tf.Variable(tf.random_normal([classes]))}

    #(input*weights) + biases


    layer_1 = tf.add(tf.matmul(data,nodes_hidden1["weights"]),nodes_hidden1["biases"])
    layer_1 = tf.nn.relu(layer_1)  #threshold functions for hidden layer_1

    layer_2 = tf.add(tf.matmul(layer_1,nodes_hidden2["weights"]),nodes_hidden2["biases"])
    layer_2 = tf.nn.relu(layer_2) #threshold functions for hidden layer_2

    layer_3 = tf.add(tf.matmul(layer_2,nodes_hidden3["weights"]),nodes_hidden3["biases"])
    layer_3 = tf.nn.relu(layer_3) #threshold functions for hidden layer_3

    output_layer = tf.add(tf.matmul(layer_3,output_layer["weights"]),output_layer["biases"])

    return output_layer
def train_network(x,y):
    prediction = neural_model(x)
    # The softmax "squishes" the inputs so that sum(input) = 1; it's a way of normalizing.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels =y))
    #optimize cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
                epoch_loss +=c
            print("Epoch",epoch,"complet out of",epochs,"loss",epoch_loss)


        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,"float"))
        print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
train_network(x,y)
