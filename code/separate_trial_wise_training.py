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

    def __init__(self, num_neurons, num_units, num_output, normalize=True):  # normalize=False for MSE?
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 51),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 11), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 11), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 11), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 11), nn.GELU()),
            nn.Conv1d(num_units, num_output, 10),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(50, 50)

if __name__ == "__main__":

    # Access the model
    print(cebra.models.get_options('my-model'))

    df = pd.read_pickle(os.path.join("data", "feat_df.pkl"))

    def define_CEBRA_model():
        cebra_model = CEBRA(
            model_architecture = "my-model",#"my-model",#"offset40-model-4x-subsample",#'offset40-model-4x-subsample', # previously used: offset1-model-v2'    # offset10-model  # my-model
            batch_size = 100,
            temperature_mode="auto",
            learning_rate = 0.005,  # learning rate = 0.005
            max_iterations = 1000,
            #distance="con",
            #time_offsets = 10,
            output_dimension = 3,  # check 10 for better performance
            device = "cuda",
            conditional="time_delta",  # assigning CEBRA to sample temporally and behaviorally for reference
            hybrid=True,  # needs to be false, otherwise not implemented error in cebra.py Line 221
            verbose = True
        )

        return cebra_model

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

    cebra_model = define_CEBRA_model()

        # idea remove the average per trial:
    l_ = []
    label_ = []
    for ANIMAL_ID in df["ANIMAL_ID"].unique():
        for RUN in df.query("ANIMAL_ID == @ANIMAL_ID")["RUN"].unique():
            df_query = df.query("ANIMAL_ID == @ANIMAL_ID and RUN == @RUN")
            #l_.append(stats.zscore(df_query.iloc[:, 1:-4], axis=0))

            X_run = df_query.iloc[:, 1:-4]
            y_run = get_int_encoded(df_query.iloc[:, -1])

            l_.append(X_run)
            label_.append(label_)

            cebra_model.partial_fit(X_run, y_run)

    X_train = pd.concat(l_)
    #X_train = stats.zscore(X_train, axis=0)
    y_train = np.concatenate(label_)


    X_train_emb = cebra_model.transform(X_train)
    embedding_scale = 10
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train_emb[::embedding_scale, 0], X_train_emb[::embedding_scale, 1], X_train_emb[::embedding_scale, 2], c=y_train[::(embedding_scale)])  # c=y_train[::(embedding_scale*4)
    plt.show()