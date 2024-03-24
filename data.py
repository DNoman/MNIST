import numpy as np
from sklearn.datasets import load_diabetes
from mlp import *
trainx = load_diabetes()['data']
target = load_diabetes()['target']
print(trainx.min(),trainx.max())
print(target.min(),target.max())
target = target - target.min()
target = target - target.max() /2
target = target / target.max()


trainx = trainx - trainx.min()
trainx = trainx - trainx.max() /2
trainx = trainx / trainx.max()
print(trainx.min(),trainx.max())
print(target.min(),target.max())

mlp1 = MLP(10, 20)
relu1 = RELU()
mlp2 = MLP(20, 20)
relu2 = RELU()
mlp3 = MLP(20, 1)

nn = CompoundNN([mlp1, relu1, mlp2,relu2,mlp3])


Epochs = 4420

optimizier = Optimizier(1e-4, nn)

#training

training_loss = []
for epoch in range(Epochs):
    loss_fct = MSELoss()

    idx = np.random.randint(0,442)
    x = trainx[idx].reshape(1,10)
    batch_target = target[idx].reshape(1,1)
    #####forward pass

    prediction = nn(x)
    loss_value = loss_fct(prediction,batch_target ) #compute the loss
    training_loss.append(loss_value)
    gradout = loss_fct.backward()
    nn.backward(gradout)

    #Update the weights
    optimizier.step()

# print(training_loss[-10:])
# plt.plot(training_loss)
# plt.show()

new_training_loss = []
M = 100

for i in range(len(training_loss) -10):
    new_training_loss.append(np.mean(training_loss[i:i+M]))

plt.plot(new_training_loss)
plt.show()

