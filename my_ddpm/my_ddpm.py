#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[43]:


n,T=10000,1000
batch_size = 128


# In[44]:


betas = np.linspace(0.001,0.01,T)
alphas = 1-betas
alphas_tilda = np.cumprod(alphas)
sqrt_alphas = np.sqrt(alphas_tilda)
sqrt_betas = np.sqrt(betas)
sqrt_one_minus_alphas = np.sqrt(1-alphas_tilda)


# In[45]:


x0 = np.random.rand(n)
x = np.zeros((n*(T+1)))
t = np.zeros((n*(T+1)))
y = np.zeros((n*(T+1)))
x[0:n] = x0
t[0:n] = 0
noise = np.random.randn(n)
y[0:n] = noise
for i in range(1,T+1):
    x[n*i:n*(i+1)] = sqrt_alphas[i-1]*x0+sqrt_one_minus_alphas[i-1]*noise
    t[n*i:n*(i+1)] = i
    y[n*i:n*(i+1)] = noise


# In[46]:


i = T
plt.hist(x[n*i:n*(i+1)],bins=25)
plt.show()


# In[33]:


x,t,y = shuffle(x,t,y,random_state=0)


# In[47]:


input_x = tf.keras.layers.Input(shape=(1,))
input_t = tf.keras.layers.Input(shape=(1,))
embedding = tf.keras.layers.Embedding(T+1,128)(input_t)
model = tf.keras.layers.Dense(64,activation='tanh')(embedding)
model = tf.keras.layers.Dense(32,activation='tanh')(model)
output = tf.keras.layers.Dense(1,activation='tanh')(model)
model = tf.keras.Model([input_x,input_t],output)


# In[ ]:


model.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.KLDivergence()])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(x=[x,t],y=y,batch_size=128,epochs=50,callbacks=[callback])


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['kullback_leibler_divergence'])
plt.savefig('history.jpg')


# In[ ]:


test_size = 1000
u = np.random.randn(test_size)
for i in range(T-1,-1,-1):
    e = model.predict([u,np.ones(test_size)*(i+1)])
    u = (u-betas[i]*e[:,0]/sqrt_one_minus_alphas[i])/np.sqrt(alphas[i])+np.random.randn(test_size)*sqrt_betas[i]
    u[:,i+1] = 0


# In[ ]:


plt.hist(u[:,0], bins=25)
plt.savefig('answer.jpg')


# In[ ]:




