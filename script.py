import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

fp = 'indy_20160407_02.mat'

data = hdf5storage.loadmat(fp)

wf = data['wf']
wf_raw = wf[7][0]
wf_unit_true = wf[7][2]
plt.plot(wf_raw.T)
plt.show()

plt.plot(wf_unit_true.T)
plt.show()

pca = PCA(n_components = 2)
feature = pca.fit_transform(wf_raw)
x = feature[:,0]
y = feature[:,1]
plt.figure(figsize=(10,5))
plt.plot(x,y,'o')
plt.show()

X = np.array(list(zip(x,y)))
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

x = X[:, 0]
y = X[:, 1]
plt.figure(figsize=(10,5))
plt.scatter(x,y , c=y_kmeans, s=30, cmap='rainbow')

plt.figure(figsize=(10,20))
mask1 = y_kmeans == 1
mask2 = y_kmeans == 0
ax1 = plt.subplot(311)
ax1.plot(wf_raw.T[:,mask1.T])
plt.title('Neuron 1')
plt.xlabel('Channel Number')
plt.ylabel('Amplitude')
plt.subplots_adjust(hspace=0.5)

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(wf_raw.T[:,mask2.T])
plt.title('Neuron 2')
plt.xlabel('Channel Number')
plt.ylabel('Amplitude (μV)')