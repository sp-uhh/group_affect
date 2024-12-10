import numpy as np

from readers.memo_ga import MEMODataset
from utilities.train_infer import get_model, calc_loss, stack_av_labels


def simple_train(config, trainset: MEMODataset, validset: MEMODataset, testset: MEMODataset):
    
    # Dataprep
    X_train = trainset.features
    y_train = stack_av_labels(config, trainset.ArousalGT, trainset.ValenceGT, trainset.ArousalAvg, trainset.ValenceAvg)
    
    X_test = testset.features
    y_test = stack_av_labels(config, testset.ArousalGT, testset.ValenceGT, testset.ArousalAvg, testset.ValenceAvg)
    
    X_valid = validset.features
    y_valid = stack_av_labels(config, validset.ArousalGT, validset.ValenceGT, validset.ArousalAvg, validset.ValenceAvg)
    
    # Load Model (new or from checkpoint, using config)
    model = get_model(config)
    # Train Model
    print("Training X: ", X_train.shape, "y: ", y_train.shape)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Metics & Loss calc
    arous_loss, valen_loss, total_loss = calc_loss(config.loss_term, y_test, y_pred)
    
    print("Train for Features: ", trainset.feats_cols)
    print("arousal loss: ", arous_loss)
    print("valence loss: ", valen_loss)
    print("Total loss: ", total_loss)