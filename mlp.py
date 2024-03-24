import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class MLP():
    def __init__(self,din,dout):
        self.w = (2 * np.random.rand(dout,din) - 1 ) * (np.sqrt(6) / np.sqrt(din + dout))
        self.b = (2 * np.random.rand(dout) - 1 ) * (np.sqrt(6) / np.sqrt(din + dout))
        self.dout = dout
        self.din = din

    def forward(self,x):
        self.x = x
        return x @ self.w.T + self.b

    def backward(self,gradout):

        self.deltaw = gradout.T @ self.x
        self.deltab = gradout.sum(0)


        return gradout @ self.w

    def __call__(self, x):
        return self.forward(x)

    def load(self,path: str):
        self.w = np.load(path + '_W.npy')
        self.b = np.load(path + '_b.npy')

    def save(self,path: str):
        np.save(path + '_W',self.w)
        np.save(path + '_b',self.b)

class RELU():
    def forward(self,x): # x.shape = (batch_size,din)
        self.x = x #Storing x for letter (backward)
        return np.maximum(0,x)

    def backward(self,gradout):

        new_grad = gradout.copy()
        new_grad[self.x < 0 ] = 0.
        return new_grad

    def __call__(self, x):
        return self.forward(x)

    def load(self, path:str):
        pass #nothing to do

    def save(self, path:str):
        pass

class CompoundNN():
    def __init__(self,blocks: list):
        self.blocks = blocks

    def forward(self,x):
        for block in self.blocks:
            x = block(x)

        return x

    def backward(self,gradout):
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
        return gradout

    def __call__(self, x):
        return self.forward(x)

    def load(self, path:str):
        for i,block in enumerate(self.blocks):
            block.load(path + f'_{i}')

    def save(self, path:str):
        for i,block in enumerate(self.blocks):
            block.save(path + f'_{i}')

# class Softmax():
#     def forward(self,x):
#         return np.exp(x) / np.exp(x).sum()
#     def backward(self):
#         raise NotImplementedError
#
#     def __call__(self, x):
#         return self.forward(x)
#
#     def load(self, path:str):
#         pass #nothing to do
#
#     def save(self, path:str):
#         pass

class MSELoss():
    def forward(self,pred,true):
        self.pred = pred
        self.true = true
        return ((pred-true)**2).mean(1).mean()
    def __call__(self, pred,true):
        return self.forward(pred,true)
    def backward(self):
        batch_size = self.pred.shape[1]
        din = self.pred.shape[1]
        jacobian = 2 * (self.pred - self.true) * 1 / din /batch_size
        return  jacobian

class Optimizier():
    def __init__(self,lr,compound_nn : CompoundNN):
        self.lr = lr
        self.compound_nn = compound_nn
    def step(self):
        for block in self.compound_nn.blocks:
            if block.__class__ == MLP:
                block.w = block.w - self.lr * block.deltaw
                block.b = block.b - self.lr * block.deltab

class LogSoftMax():
    def forward(self,x):
        self.x = x
        return x - logsumexp(x,axis = 1)[...,None]
        # return x - np.log(np.exp(x).sum())

    def __call__(self, x):
        return self.forward(x)

    def backward(self,gradout):

        gradients = np.eye(self.x.shape[1])[None,...]
        gradients = gradients - (np.exp(self.x) / np.sum(np.exp(self.x),axis=1)[...,None])[...,None]
        return (np.matmul(gradients,gradout[...,None]))[:,:,0]


class NLLLoss():
    def forward(self,pred,true):
        self.pred = pred
        self.true = true
        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b,true[b]]
        return loss
    def __call__(self, pred,true):
        return self.forward(pred,true)

    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((self.pred.shape[0],din))
        for b in range(self.pred.shape[0]):
            jacobian[b,self.true[b]] =-1
        return jacobian

######Test init
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# x = np.random.rand(1,6)
# nn = CompoundNN([mlp1,relu1,mlp2,relu2])


####test block
# print(nn(x))
# print(relu2(mlp2(relu1(mlp1(x)))))
###test load and save
# mlp1 = MLP(6,5)
# mlp2 = MLP(6,5)
# print(mlp1(x))
# print(mlp2(x))
# mlp1.save('mlp')
# mlp2.load('mlp')
# print()
# print(mlp1(x))
# print(mlp2(x))
###test load_and save block
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# x = np.random.rand(1,6)
# nn1 = CompoundNN([mlp1,relu1,mlp2,relu2])
#
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# nn2 = CompoundNN([mlp1,relu1,mlp2,relu2])
#
# print(nn1(x))
# print(nn2(x))
# nn1.save('nn')
# nn2.load('nn')
# print()
# print(nn1(x))
# print(nn2(x))
#######Test backward
# mlp1 = MLP(16,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# x = np.random.rand(1,16)
# nn1 = CompoundNN([mlp1,relu1,mlp2,relu2])
#
# mlp1 = MLP(16,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# nn2 = CompoundNN([mlp1,relu1,mlp2,relu2])
# nn1(x)
# gradout = np.random.rand(1,4)
# y = nn1.backward(gradout)
# print(y)
#########loss function
# loss_fct = MSELoss()
# a = loss_fct(np.array([[1,2]]),np.array([[1.4,2.5]]))
# print(a)
# b = loss_fct.backward()
# print(b)
############Training Neural
# for lr in [1e-5,1e-4,1e-3,1e-2,1e-1,1]:
#     mlp1 = MLP(6,5)
#     relu1 = RELU()
#     mlp2 = MLP(5,4)
#     relu2 = RELU()
#     x = np.random.rand(1,6)
#     nn = CompoundNN([mlp1,relu1,mlp2])
#
#     target = np.array([[1.,2.,3.,4.]])
#     Epochs = 100
#
#     optimizier = Optimizier(lr,nn)
#
#     initial_prediction = nn(x)
#
#     training_loss = []
#     for epoch in range(Epochs):
#         loss_fct = MSELoss()
#         #####forward pass
#         prediction = nn(x)
#         loss_value = loss_fct(prediction,target ) #compute the loss
#         training_loss.append(loss_value)
#         gradout = loss_fct.backward()
#         nn.backward(gradout)
#
#         #Update the weights
#         optimizier.step()
#
#     plt.plot(training_loss,label = lr)
#     plt.legend()
#
#
# print(initial_prediction)
# print(prediction)
# print(target)
# print(training_loss[-1])
# plt.show()
###################Log softmax
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# log_softmax = LogSoftMax()
# x = np.random.rand(1,6)
# nn = CompoundNN([mlp1,relu1,mlp2,log_softmax])
# nn(x)
# log_probabilities = nn(x)
# probilities = np.exp(log_probabilities)
# print(np.sum(probilities))
# np.array([[1,2,3]])
#
# predicted_probilities = np.array([[0.2,0.7,0.1]])
# real_label = 0
# y = predicted_probilities[0,real_label]
# print(y)
# maximize: predicted_probilities[real_label]
# minimize: -predicted_probilities[real_label]