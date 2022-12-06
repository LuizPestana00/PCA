import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

df = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')
df_drop = df.drop(labels=['Unnamed: 0','Date','type','region'], axis=1)
mean = np.mean(df_drop,axis=0)
mRd = df_drop - mean
mRd.head()

cov_matrix = np.cov(mRd.T)
cov_matrix

autval,autvet = np.linalg.eig(cov_matrix)
autval = np.array(np.sort(autval[::-1]))

sns.lineplot(x=df_drop.columns[0],y=df_drop.columns[1],data=df_drop)
plt.show()