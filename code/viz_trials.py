import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from scipy import stats

df = pd.read_pickle(os.path.join("data", "feat_df_v01.pkl"))

animals = df["ANIMAL_ID"].unique()
runs = df.query("ANIMAL_ID == @animals[0]")["RUN"].unique()


plt.figure(figsize=(10, 4), dpi=300)
for run in runs:
    df_run_ = df.query("ANIMAL_ID == @animals[0] and RUN == @run")

    plt.plot(np.array(df_run_["left_knee_angle_mean"]), label=run)
plt.legend()

# first question: what is the duration of each run?
df_count_size = df.groupby(["ANIMAL_ID", "RUN"]).count()["CONDITION"].reset_index()
df_count_size["RUN_TIME"] = df_count_size["CONDITION"]/200
df_count_size["name"] = df_count_size["ANIMAL_ID"] + "_" + df_count_size["RUN"]


fig = plt.figure(figsize=(5, 15))
ax = fig.add_subplot(1,1,1)
df_count_size.plot.barh(y="RUN_TIME", x="name", ax=ax)
plt.xlabel("Run time [s]")
plt.tight_layout()


plt.hist(df_count_size["RUN_TIME"], bins=50)


##############################################
l_ = []
label_ = []
for ANIMAL_ID in df["ANIMAL_ID"].unique():
    for RUN in df.query("ANIMAL_ID == @ANIMAL_ID")["RUN"].unique():
        df_query = df.query("ANIMAL_ID == @ANIMAL_ID and RUN == @RUN")
        l_.append(stats.zscore(df_query.iloc[:, 1:-4], axis=0))
        label_.append(df_query.iloc[:, -1])

X_train = pd.concat(l_)
y_train = np.concatenate(label_)


# transform the HEX-values
hex_val_color = [str(s) for s in y_train]

red = [int(h[1:3], 16) / 255.0 for h in hex_val_color]
green = [int(h[3:5], 16) / 255.0 for h in hex_val_color]
blue = [int(h[5:7], 16) / 255.0 for h in hex_val_color]

color_vals_rgb = [red, green, blue]

y_train = np.array(color_vals_rgb).T
y_train = np.argmax(y_train, axis=1)
y_train = y_train.astype("int")

# plot the counts of states for each animal

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

label_animals = []
for ANIMAL_ID in df["ANIMAL_ID"].unique():
    label_ = []

    for RUN in df.query("ANIMAL_ID == @ANIMAL_ID")["RUN"].unique():
        df_query = df.query("ANIMAL_ID == @ANIMAL_ID and RUN == @RUN")
        label_.append(df_query.iloc[:, -1])
    label_animals.append(get_int_encoded(np.concatenate(label_)))


# encode: immobility, interruption, gait (0, 1, 2)
def return_label_encoding(values):
    d = {0: "immobility", 1: "interruption", 2:"gait"}
    return [d[v] for v in values]

plt.figure(figsize=(12, 9))
for idx, ANIMAL_ID in enumerate(df["ANIMAL_ID"].unique()):
    plt.subplot(2,3,idx+1)
    unique, counts = np.unique(label_animals[idx], return_counts=True)
    plt.bar(np.arange(unique.shape[0]), counts)
    plt.xlabel("counts")
    plt.xticks(np.arange(unique.shape[0]), return_label_encoding(unique))
    plt.title(ANIMAL_ID)
plt.tight_layout()