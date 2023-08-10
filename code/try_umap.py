import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import umap

def get_int_encoded(y_train):
    # transform the HEX-values
    hex_val_color = [str(s) for s in y_train]

    red = [int(h[1:3], 16) / 255.0 for h in hex_val_color]
    green = [int(h[3:5], 16) / 255.0 for h in hex_val_color]
    blue = [int(h[5:7], 16) / 255.0 for h in hex_val_color]

    color_vals_rgb = [red, green, blue]

    y_train = np.array(color_vals_rgb).T
    y_train = np.argmax(y_train, axis=1)
    y_train = y_train.astype("int")
    return y_train

df = pd.read_pickle(os.path.join("data", "feat_df.pkl"))




l_ = []
label_ = []
for ANIMAL_ID in df["ANIMAL_ID"].unique():
    for RUN in df.query("ANIMAL_ID == @ANIMAL_ID")["RUN"].unique():
        df_query = df.query("ANIMAL_ID == @ANIMAL_ID and RUN == @RUN")
        l_.append(stats.zscore(np.array(df_query.iloc[:, 1:-4]), axis=0))

        y_run = get_int_encoded(df_query.iloc[:, -1])
        label_.append(y_run)

X_train = np.concatenate(l_)#pd.concat(l_)
#X_train = stats.zscore(X_train, axis=0)
y_train = np.concatenate(label_)


umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='euclidean')
umap_result = umap_model.fit_transform(X_train)

plt.scatter(umap_result[:, 0], umap_result[:, 1], c=y_train[:], s=5)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# healthy embedding:
X_ = X_train[df["CONDITION"] == "H"][::10]
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='euclidean')

umap_result_H = umap_model.fit_transform(X_)
plt.scatter(umap_result_H[:, 0], umap_result_H[:, 1], c=y_train[df["CONDITION"] == "H"][::10], s=5, alpha=.2)
plt.xlabel('UMAP 1')
plt.title("Healthy")
plt.ylabel('UMAP 2')
plt.show()

X_ = X_train[df["CONDITION"] == "PD"]
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='euclidean')
umap_result_PD = umap_model.fit_transform(X_)

plt.scatter(umap_result_PD[:, 0], umap_result_PD[:, 1], c=y_train[df["CONDITION"] == "PD"], s=5, alpha=.2)
plt.xlabel('UMAP 1')
plt.title("PD")
plt.ylabel('UMAP 2')
plt.show()
