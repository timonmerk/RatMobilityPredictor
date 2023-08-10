# conda env: pn_env

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

import xgboost
from sklearn import linear_model

from sklearn.metrics import ConfusionMatrixDisplay

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

def define_CEBRA_model():

    cebra_model = CEBRA(
        model_architecture = "offset1-model",#"my-model",#"offset40-model-4x-subsample",#'offset40-model-4x-subsample', # previously used: offset1-model-v2'    # offset10-model  # my-model
        batch_size = 100,
        temperature_mode="auto",
        learning_rate = 0.005,  # learning rate = 0.005
        max_iterations = 1000,
        distance="cosine",
        #time_offsets = 10,
        output_dimension = 3,  # check 10 for better performance
        device = "cuda",
        conditional="time_delta",  # assigning CEBRA to sample temporally and behaviorally for reference
        hybrid=False,  # needs to be false, otherwise not implemented error in cebra.py Line 221
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

if __name__ == "__main__":
    
    # Access the model
    print(cebra.models.get_options('my-model'))

    df = pd.read_pickle(os.path.join("data", "feat_df.pkl"))

    # things to try out:

    # 1. run CEBRA clustering across the different animals
    # 2. Try normalization across all data (for each condition separately)
    # 3. Try partial fitting with same learning rate


    df_healthy = df.query("CONDITION == 'H'")
    y_healthy = get_int_encoded(df_healthy.iloc[:, -1])
    X_healhy = df_healthy.iloc[:, 1:-4]
    X_healhy_normalized = stats.zscore(X_healhy, axis=0)

    cebra_model = define_CEBRA_model()

    cebra_model.fit(X_healhy_normalized, y_healthy)
    cebra.plot_loss(cebra_model)

    X_train_emb = cebra_model.transform(X_healhy_normalized)
    embedding_scale = 10
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train_emb[::embedding_scale, 0], X_train_emb[::embedding_scale, 1], X_train_emb[::embedding_scale, 2], c=y_healthy[::(embedding_scale)])  # c=y_train[::(embedding_scale*4)
    plt.show()


    
    # the embedding looks reasonable, is it also shared across animals?
    df_healthy = df.query("CONDITION == 'H'")

    # immobility (r) = 0, interruption (g) = 1, gait (b) = 2
    # 

    pr_gait = []
    pr_immobility = []
    label_gait = []
    label_immobility = []

    for test_animal in df_healthy["ANIMAL_ID"].unique():

        print(test_animal)
        df_test = df.query("ANIMAL_ID == @test_animal")
        y_test = get_int_encoded(df_test.iloc[:, -1])
        X_test = df_test.iloc[:, 1:-4]
        X_test_normalized = stats.zscore(X_test, axis=0)

        df_train = df.query("ANIMAL_ID != @test_animal")
        y_train = get_int_encoded(df_train.iloc[:, -1])
        X_train = df_train.iloc[:, 1:-4]
        
        # potentially replace zscore with sklearn.preprocessing.QuantileTransformer
        X_train_normalized = stats.zscore(X_train, axis=0)

        cebra_gait = define_CEBRA_model()

        cebra_gait.fit(X_train_normalized, y_train==2)
        X_train_normalized_transf = cebra_gait.transform(X_train_normalized)
        X_test_normalized_transf = cebra_gait.transform(X_test_normalized)

        model_gait = linear_model.LogisticRegression(class_weight="balanced")
        model_gait = model_gait.fit(X_train_normalized_transf, (y_train==2))
        pr_gait.append(model_gait.predict(X_test_normalized_transf))
        label_gait.append(y_test==2)

        if np.unique(y_test).shape[0] > 2:
            model_immobility = linear_model.LogisticRegression(class_weight="balanced")
            y_train_immobility = y_train[y_train != 2]
            y_test_immobility = y_test[y_test != 2]

            X_train_immobility = X_train_normalized.loc[y_train != 2]
            X_test_immbobility = X_test_normalized.loc[y_test != 2]

            cebra_immobility = define_CEBRA_model()
            cebra_immobility.fit(X_train_immobility, y_train_immobility)
            X_train_normalized_transf = cebra_immobility.transform(X_train_immobility)
            X_test_normalized_transf = cebra_immobility.transform(X_test_immbobility)


            model_immobility = model_immobility.fit(X_train_normalized_transf, y_train_immobility)
            pr_immobility.append(model_immobility.predict(X_test_normalized_transf))
            label_immobility.append(y_test_immobility)
        else:
            label_immobility.append([])
            pr_immobility.append([])

        #cebra_model = define_CEBRA_model()
        #cebra_model = xgboost.XGBClassifier()
        #cebra_model = linear_model.LogisticRegression()

        #cebra_model.fit(X_train_normalized, y_train)

        #X_train_emb = cebra_model.transform(X_train)
        #X_test_emb = cebra_model.transform(X_test)

        

        #model = xgboost.XGBClassifier()

        
        #model_regressor.predict(X_test_emb)

        #pr.append(model_regressor.predict(X_test_normalized))

        #label.append(y_test)


    # implement the same for the PD animal

    idx_label = 4

    cm_true = label_immobility[idx_label]
    cm_pr = pr_immobility[idx_label]
    #cm_true = label_gait[idx_label]
    #cm_pr = label_gait[idx_label]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ConfusionMatrixDisplay.from_predictions(cm_true, cm_pr, ax=ax)  #,normalize="true", display_labels=["immobility", "gait", "interruption"] 
    plt.show()



    fig = plt.figure(figsize=(5,5), dpi=300)
    ax = fig.add_subplot(111)

    ConfusionMatrixDisplay.from_predictions(np.concatenate(np.array(label_immobility)[[0, 2,3]]),
                                            np.concatenate(np.array(pr_immobility)[[0, 2,3]]), normalize="true", ax=ax)  #, display_labels=["immobility", "gait", "interruption"] 
    plt.show()


    # immobility seems to be a super difficult class to predict

    # I almost never predict it correctly; even with 


    # A basic CV for the single PD animal 

    kf = model_selection.KFold(n_splits=3, shuffle=False)

    pr_ = []
    true_ = []

    df_disease = df.query("CONDITION == 'PD'")
    y_disease = get_int_encoded(df_healthy.iloc[:, -1])

    for i, (train_index, test_index) in enumerate(kf.split(df_disease)):
        X_train = stats.zscore(df_disease.iloc[train_index, 1:-4], axis=0)
        y_train = y_disease[train_index]

        X_test = stats.zscore(df_disease.iloc[test_index, 1:-4], axis=0)
        y_test = y_disease[test_index]

        cebra_gait = define_CEBRA_model()

        cebra_gait.fit(X_train, y_train==2)

        X_train_emb = cebra_model.transform(X_train)
        X_test_emb = cebra_model.transform(X_test)
        model_regressor = linear_model.LogisticRegression(class_weight="balanced").fit(X_train_emb, y_train)

        y_pr = model_regressor.predict(X_test_emb)

        pr_.append(y_pr)
        true_.append(y_test)
