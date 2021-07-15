import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
# print([net.layers[i].get_weights() for i in range(len(net.layers))])
X = tf.random.uniform((2, 20))
print(net(X))
print([W.shape for W in net.get_weights()])
