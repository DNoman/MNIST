import numpy as np


class NeuralNetwork:
    def __init__(self, x):
        self.x = x

    def get_input_dimension(self):
        din = self.x.shape[1]
        return din


# Tạo một mảng NumPy đại diện cho dữ liệu đầu vào (ví dụ: 100 mẫu, mỗi mẫu có 3 đặc trưng)
data = np.random.rand(100, 3)

# Khởi tạo một đối tượng NeuralNetwork với dữ liệu đầu vào
model = NeuralNetwork(data)

# Lấy kích thước của đầu vào
input_dim = model.get_input_dimension()
a = np.eye(input_dim)

# In kích thước của đầu vào (số lượng đặc trưng)
print("Kích thước của đầu vào là:", input_dim)
print(a)