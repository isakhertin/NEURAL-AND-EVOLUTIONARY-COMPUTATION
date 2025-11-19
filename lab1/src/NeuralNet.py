import numpy as np

class NeuralNet:
    def __init__(self, layers, epochs=1000, lr=0.01, momentum=0.9, activation="sigmoid", validation_percent=0):
        self.L = len(layers)
        self.n = layers.copy()
        self.xi = [np.zeros(l) for l in layers]
        self.w = [np.zeros((1, 1))]
        for lay in range(1, self.L):
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * 0.1)
        self.theta = [np.zeros(l) for l in layers]

        # For momentum
        self.d_w_prev = [np.zeros_like(w) for w in self.w]
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]

        # Additional variables as per requirements
        self.h = [None] * self.L
        self.delta = [np.zeros(l) for l in self.n]
        self.d_w = [np.zeros_like(w) for w in self.w]
        self.d_theta = [np.zeros_like(t) for t in self.theta]

        # Activation function (fact for hidden, output fixed to linear for regression)
        self.fact = activation
        self.activation_hidden = activation
        self.activation_output = "linear"

        # Training parameters
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.validation_percent = validation_percent / 100.0  # Convert percent to fraction

        # Loss tracking
        self.train_losses = []
        self.val_losses = []

    # --- Activation functions ---
    def activation(self, x, func_name):
        if func_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif func_name == "tanh":
            return np.tanh(x)
        elif func_name == "relu":
            return np.maximum(0, x)
        elif func_name == "linear":
            return x

    def activation_derivative(self, x, func_name):
        if func_name == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif func_name == "tanh":
            return 1 - np.tanh(x)**2
        elif func_name == "relu":
            return np.where(x > 0, 1, 0)
        elif func_name == "linear":
            return np.ones_like(x)

    # --- Forward pass ---
    def forward(self, X, activation_hidden, activation_output):
        self.h = [None] * self.L
        self.xi[0] = X
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) + self.theta[l]
            func_name = activation_output if l == self.L - 1 else activation_hidden
            self.xi[l] = self.activation(self.h[l], func_name)
        return self.xi[-1]

    # --- Training (Backpropagation) ---
    def fit(self, X, y, shuffle=False):
        # Reset losses
        self.train_losses = []
        self.val_losses = []

        # Split data into train and validation
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        split_idx = int(n_samples * (1 - self.validation_percent))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        if self.validation_percent == 0:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for epoch in range(self.epochs):
            total_error = 0
            indices = np.arange(len(X_train))
            if shuffle:
                np.random.shuffle(indices)

            for i in indices:
                xi_input = X_train[i]
                yi_target = y_train[i]

                # Forward pass
                output = self.forward(xi_input, self.activation_hidden, self.activation_output)

                # Backward pass
                error = yi_target - output
                total_error += np.mean(error**2)

                # Delta for output layer
                self.delta[-1] = error * self.activation_derivative(
                    self.h[-1], self.activation_output
                )

                # Delta for hidden layers
                for l in range(self.L - 2, 0, -1):
                    self.delta[l] = np.dot(self.w[l + 1].T, self.delta[l + 1]) * \
                                    self.activation_derivative(self.h[l], self.activation_hidden)

                # Update weights and biases
                for l in range(1, self.L):
                    self.d_w[l] = self.lr * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
                    self.d_theta[l] = self.lr * self.delta[l] + self.momentum * self.d_theta_prev[l]

                    self.w[l] += self.d_w[l]
                    self.theta[l] += self.d_theta[l]

                    self.d_w_prev[l] = self.d_w[l]
                    self.d_theta_prev[l] = self.d_theta[l]

            train_loss = total_error / len(X_train)
            self.train_losses.append(train_loss)

            if X_val is not None and len(X_val) > 0:
                y_pred_val = self.predict(X_val)
                val_error = y_val - y_pred_val
                val_loss = np.mean(val_error**2)
                self.val_losses.append(val_loss)
            else:
                self.val_losses.append(0.0)

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print_str = f"Epoch {epoch+1}/{self.epochs}, Train Error: {train_loss:.6f}"
                if X_val is not None and len(X_val) > 0:
                    print_str += f", Val Error: {self.val_losses[-1]:.6f}"
                print(print_str)

    # --- Prediction ---
    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.forward(X[i], self.activation_hidden, self.activation_output))
        return np.array(y_pred)

    # --- Loss evolution ---
    def loss_epochs(self):
        train_losses = np.array(self.train_losses)
        val_losses = np.array(self.val_losses)
        return train_losses, val_losses