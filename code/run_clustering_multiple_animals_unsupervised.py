import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from scipy import stats

from torch import nn
import cebra
import cebra.models
import cebra.data
from cebra import CEBRA
import cebra.models.layers as cebra_layers
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin

@cebra.models.register("my-model") # --> add that line to register the model!
class Offset10Model(_OffsetModel, ConvolutionalModelMixin):
    """CEBRA model with a 10 sample receptive field."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(5, 5)

if __name__ == "__main__":
    
    # Access the model
    print(cebra.models.get_options('my-model'))

    df = pd.read_pickle(os.path.join("data", "feat_df.pkl"))

    def define_CEBRA_model():
        cebra_model = CEBRA(
            model_architecture = "offset40-model-4x-subsample",#'offset40-model-4x-subsample', # previously used: offset1-model-v2'    # offset10-model  # my-model
            batch_size = 100,
            temperature_mode="auto",
            learning_rate = 0.005,
            max_iterations = 1000,
            #time_offsets = 10,
            output_dimension = 3,  # check 10 for better performance
            device = "cuda",
            conditional="time_delta",  # assigning CEBRA to sample temporally and behaviorally for reference
            hybrid=False,  # needs to be false, otherwise not implemented error in cebra.py Line 221
            verbose = True
        )

        return cebra_model

    cebra_model = define_CEBRA_model()

    # idea remove the average per trial:
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

    # plot the states over time:
    #plt.scatter(np.arange(len(red)), np.ones(len(red)), c=np.array(color_vals_rgb).T, marker="o",s=5)
    #plt.show()

    # challenge: how to transform the individual labels into immobility (r), interruption (g), gait (b)

    cebra_model.fit(X_train)  # y_train

    cebra.plot_loss(cebra_model)
    cebra.plot_temperature(cebra_model)

    #plt.figure()
    #plt.imshow(X_train.T, aspect="auto")
    #plt.clim(-3, 3)
    #plt.show()


    X_train_emb = cebra_model.transform(X_train)

    #cebra.plot_embedding(X_train_emb, cmap="viridis", markersize=10, alpha=0.5, embedding_labels=y_train.T) # embedding_labels=y_train
    #plt.axis("off")
    # plt.savefig("3DEmbedding.pdf")

    embedding_scale = 11
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train_emb[::embedding_scale, 0], X_train_emb[::embedding_scale, 1], X_train_emb[::embedding_scale, 2], c=y_train[::(embedding_scale*4)])  # c=y_train[::(embedding_scale*4)
    plt.show()

    # implement cross validation and check performance

    kf = model_selection.KFold(n_splits=3, shuffle=False)
    X_ = X_train
    y_ = y_train

    pr_ = []
    true_ = []


    for i, (train_index, test_index) in enumerate(kf.split(X_)):
        X_train = X_.iloc[train_index, :]
        y_train = y_[train_index]

        X_test = X_.iloc[test_index, :]
        y_test = y_[test_index]

        cebra_model = define_CEBRA_model()

        cebra_model.fit(X_train)  # y_train

        X_train_emb = cebra_model.transform(X_train)
        X_test_emb = cebra_model.transform(X_test)
        model_regressor = linear_model.LogisticRegression().fit(X_train_emb, y_train[::4])

        y_pr = model_regressor.predict(X_test_emb)

        pr_.append(y_pr)
        true_.append(y_test[::4])

    # get a certain set of performances

    # get performance for each value:

    from sklearn.metrics import ConfusionMatrixDisplay

    fig = plt.figure(figsize=(5,5), dpi=300)
    ax = fig.add_subplot(111)

    ConfusionMatrixDisplay.from_predictions(np.concatenate(true_), np.concatenate(pr_), normalize="true", display_labels=["immobility", "gait", "interruption"], ax=ax)  #, 
    plt.show()

    metrics.balanced_accuracy_score(np.concatenate(true_).astype("int"), np.rint(np.concatenate(pr_)).astype("int"))
