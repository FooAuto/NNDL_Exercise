import numpy as np
import matplotlib.pyplot as plt
import time


class MyFNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, clip_value=1.0):
        self.lr = learning_rate
        self.clip_value = clip_value  # 梯度裁剪阈值, 不然容易梯度爆炸
        # He 初始化，适用于 ReLU
        self.W1 = np.random.randn(
            input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(
            hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        """ ReLU 激活函数 """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ ReLU 的导数 """
        return (x > 0).astype(float)

    def loss(self, y_pred, y_true):
        """ 计算均方误差 (MSE) """
        return np.mean((y_pred - y_true) ** 2)

    def train(self, x_train, y_train, epochs=5000, batch_size=32, verbose=True):
        """ 训练 """
        n_samples = x_train.shape[0]
        for epoch in range(epochs):
            # 打乱数据
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # 前向传播
                z1 = np.dot(x_batch, self.W1) + self.b1
                a1 = self.relu(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                y_pred = z2

                # 计算损失
                loss = self.loss(y_pred, y_batch)

                # 反向传播
                dL_dy = 2 * (y_pred - y_batch) / y_batch.shape[0]
                dL_dW2 = np.dot(a1.T, dL_dy)
                dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)

                dL_da1 = np.dot(dL_dy, self.W2.T)
                dL_dz1 = dL_da1 * self.relu_derivative(z1)
                dL_dW1 = np.dot(x_batch.T, dL_dz1)
                dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

                # **梯度裁剪**
                dL_dW2 = np.clip(dL_dW2, -self.clip_value, self.clip_value)
                dL_db2 = np.clip(dL_db2, -self.clip_value, self.clip_value)
                dL_dW1 = np.clip(dL_dW1, -self.clip_value, self.clip_value)
                dL_db1 = np.clip(dL_db1, -self.clip_value, self.clip_value)

                # **参数更新**
                self.W2 -= self.lr * dL_dW2
                self.b2 -= self.lr * dL_db2
                self.W1 -= self.lr * dL_dW1
                self.b1 -= self.lr * dL_db1

            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def predict(self, x):
        """ 预测 """
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2


def target_function(x):
    """ 目标有界闭集函数 """
    return x ** 3 - 2 * x ** 2 + 3 * x  # 多项式函数


# **采样数据**
np.random.seed(int(time.time()))
x = np.linspace(-5, 5, 16384).reshape(-1, 1)
y_true = target_function(x)

# **划分数据集**
x_train, y_train = x[0:-1024], y_true[0:-1024]  # 训练集
x_test, y_test = x, y_true  # 全部数据用于测试

nn = MyFNN(input_dim=1, hidden_dim=16, output_dim=1,
           learning_rate=0.01, clip_value=1.0)
nn.train(x_train, y_train, epochs=5000, batch_size=32)

y_test_pred = nn.predict(x_test)

mse_test = np.mean((y_test_pred - y_test) ** 2)
print(f"Test MSE: {mse_test:.5f}")

plt.scatter(x_train, y_train, label="Train Data", color="blue", alpha=0.5)
plt.plot(x_test, y_test, label="True Function", color="green", linewidth=2)
plt.plot(x_test, y_test_pred, label="NN Prediction", color="red", linewidth=2)
plt.legend()
plt.show()
