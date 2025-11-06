import numpy as np

class NeuralNet:
    def __init__(self, layers):
        self.L = len(layers)
        self.n = layers.copy()
        self.xi = [np.zeros(l) for l in layers]
        self.w = [np.zeros((1, 1))]
        for lay in range(1, self.L):
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * 0.1)
        self.theta = [np.zeros(l) for l in layers]

        # För momentum (senare uppdatering)
        self.d_w_prev = [np.zeros_like(w) for w in self.w]
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]

    # --- Aktiveringsfunktioner ---
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

    # --- Framåtpass ---
    def forward(self, X, func_name):
        self.h = [None] * self.L
        self.xi[0] = X
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self.activation(self.h[l], func_name)
        return self.xi[-1]

    # --- Träning (Backpropagation) ---
    def fit(self, X, y, epochs=1000, lr=0.01, momentum=0.9, activation="sigmoid"):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                xi_input = X[i]
                yi_target = y[i]

                # Framåtpass
                output = self.forward(xi_input, activation)

                # Bakåtpass
                delta = [np.zeros(l) for l in self.n]
                error = yi_target - output
                total_error += np.mean(error**2)

                # Delta för utlagret
                delta[-1] = error * self.activation_derivative(self.h[-1], activation)

                # Delta för dolda lager
                for l in range(self.L - 2, 0, -1):
                    delta[l] = np.dot(self.w[l + 1].T, delta[l + 1]) * \
                               self.activation_derivative(self.h[l], activation)

                # Uppdatera vikter och bias
                for l in range(1, self.L):
                    d_w = lr * np.outer(delta[l], self.xi[l - 1]) + momentum * self.d_w_prev[l]
                    d_theta = lr * delta[l] + momentum * self.d_theta_prev[l]

                    self.w[l] += d_w
                    self.theta[l] -= d_theta

                    self.d_w_prev[l] = d_w
                    self.d_theta_prev[l] = d_theta

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Error: {total_error / len(X):.6f}")

    # --- Prediktion ---
    def predict(self, X, activation="sigmoid"):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.forward(X[i], activation))
        return np.array(y_pred)
