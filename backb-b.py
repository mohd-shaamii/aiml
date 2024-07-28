import numpy as np

# Data and parameters
x = np.array([[2, 9], [1, 5], [3, 6]], dtype=float) / 9
y = np.array([[92], [86], [89]], dtype=float) / 100
sigmoid = lambda x: 1 / (1 + np.exp(-x))
deriv_sigmoid = lambda x: x * (1 - x)
epoch = 5000
lr = 0.1

# Initialize weights and biases
input_size = x.shape[1]
hidden_size = 3
output_size = 1

wh = np.random.uniform(size=(input_size, hidden_size))
bh = np.random.uniform(size=(1, hidden_size))
wout = np.random.uniform(size=(hidden_size, output_size))
bout = np.random.uniform(size=(1, output_size))

# Training
for _ in range(epoch):
    # Forward pass
    hlayer_act = sigmoid(np.dot(x, wh) + bh)
    output = sigmoid(np.dot(hlayer_act, wout) + bout)

    # Compute the loss and gradients
    d_output = (y - output) * deriv_sigmoid(output)
    d_hiddenlayer = np.dot(d_output, wout.T) * deriv_sigmoid(hlayer_act)

    # Update weights and biases
    wout += np.dot(hlayer_act.T, d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += np.dot(x.T, d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

# Output results
predicted_output = output
print("Input (scaled):\n", x)
print("Actual Output (scaled):\n", y)
print("Predicted Output (scaled):\n", predicted_output)
