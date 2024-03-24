from mlp import *
from keras.datasets.mnist import load_data
from tqdm import tqdm

(trainx,trainy) , (testx,testy) =load_data()

trainx = (trainx - 127.5) /127.5
testx = (testx - 127.5)/127.5

trainx = trainx.reshape(trainx.shape[0],28 * 28)
plt.imshow(trainx[1].reshape(28,28),cmap = 'gray')


mlp1 = MLP(28*28,128)
relu1 = RELU()
mlp2 = MLP(128,64)
relu2 = RELU()
mlp3 = MLP(64,10)
s = LogSoftMax()
nn = CompoundNN([mlp1,relu1,mlp2,relu2,mlp3,s])
optimizer = Optimizier(1e-3,nn)
Epochs = 14000
batch_size = 128
training_loss = []
for epoch in tqdm(range(Epochs)):
    loss_fct = NLLLoss()
    idx = [np.random.randint(0,trainx.shape[0]) for _ in range(batch_size)]
    x = trainx[idx]
    target = trainy[idx]
    #####forward pass
    prediction = nn(x)
    loss_value = loss_fct(prediction,target ) #compute the loss
    training_loss.append(loss_value)
    gradout = loss_fct.backward()
    nn.backward(gradout)

    #Update the weights
    optimizer.step()

accuracy = 0
for i in range(testx.shape[0]):
    prediction = nn(testx[i].reshape(1,784)).argmax()

    if prediction == testy[i]:
        accuracy += 1
    else:
        pass

print('Accuracy', accuracy /testx.shape[0] * 100, '%')