import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

n,T = 50000,50

betas = np.linspace(0.001,0.01,T)
alphas = 1-betas
alphas_tilda = np.cumprod(alphas)
sqrt_alphas = np.sqrt(alphas_tilda)
sqrt_betas = np.sqrt(betas)
sqrt_one_minus_alphas = np.sqrt(1-alphas_tilda)

x = np.zeros((n,T+1,2))
x0 = np.random.rand(n)
x[:,0,0]=x0
x[:,0,1]=0
y = np.zeros((n,T))

noise = np.random.randn(n,T)

for i in range(T):
    x[:,i+1,0] = sqrt_alphas[i]*x[:,i,0]+sqrt_betas[i]*noise[:,i]
    x[:,i+1,1] = (i+1)/T
    y[:,i] = (x[:,i+1,0]-sqrt_alphas[i]*x0)/sqrt_one_minus_alphas[i]
    
x = np.delete(x, (0), axis=1)
x = x.reshape((n*T),2)
y = y.reshape((n*T))

x,y = shuffle(x,y,random_state=0)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(10, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='tanh'))

model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
model.fit(x,y,batch_size=128,epochs=100,callbacks=[callback])

model.save_weights('./checkpoints/my_checkpoint')

test_size = 1000
u = np.random.randn(test_size)
for i in range(T-1,-1,-1):
    ux = np.array([[w,(i+1)/T] for w in u])
    e = model.predict(ux)
    u = (u-(1-alphas[i])*e[:,0]/sqrt_one_minus_alphas[i])/np.sqrt(alphas[i])+np.random.randn(test_size)

plt.hist(u,bins=25)
plt.savefig('hist.jpg')