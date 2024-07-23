# -*- coding: utf-8 -*-
"""
Note from CyprienGille : This code is an amalgamation of the work of several people, and needs a complete overhaul
for better factorization and readability. I will delete this comment when this is done.

Copyright   I3S CNRS UCA 

This code is an implementation of the statistical evaluation of the autoencoder described in the article :
Learning a confidence score and the latent space of a new Supervised Autoencoder for diagnosis and prognosis in clinical metabolomic studies.

When using this code , please cite

 author={Barlaud, Michel and Guyard, Frédéric},  
booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},   
title={Learning sparse deep neural networks using efficient structured projections on convex constraints for green AI},  
year={2020}, 

author={Perez Guillaume, Condat Laurent, Barlaud, Michel },  
booktitle={arXiv},   
title={Near-Linear Time Projection onto the ℓ1,infini Ball;
Application to Sparse Autoencoders},  
year={2023}, 

              
author={Barlaud, Michel ,Perez Guillaume and Marmorat Jean-Paul},  
booktitle={Preprint Hal-I3S -2024},   
title={Linear Time Bi-level ℓ1,infini projection ; Application to
sparse auto-encoders neural networks},  
year={2024}, 



Parameters : 
    
    - Seed
    - Database (variable file_name)
    - Projection
    - Constraint ETA
    
Results_stat
    -Accuracy, F1 score (+other metrics)
    -Predicted labels on test set with confidence scores
    -Top features     
"""
#%%
import os

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch import nn
from sklearn import metrics
from numpy import linalg as LA
from sklearn.metrics import precision_recall_fscore_support

import functions.functions_torch_V8 as ft
import functions.functions_network_pytorch as fnp


#%%

if __name__ == "__main__":

    ######## Parameters ########
    start_time= time.time()
    # Set seed
    SEEDS = [5]

    # Set device (GPU or CPU)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nfolds = 4  # Number of folds for the cross-validation process
    N_EPOCHS = 30  # Number of epochs for the first descent
    N_EPOCHS_MASKGRAD = 30  # Number of epochs for training masked gradient
    LR = 0.0005  # Learning rate
    BATCH_SIZE = 4  # Optimize the trade off between accuracy and computational time
    LOSS_LAMBDA = 0.00005  # Total loss = λ * loss_reconstruction +  loss_classification

    # unit scaling of the input data
    doScale = True
    # log transform of the input data
    doLog = True

    # loss function for reconstruction
    criterion_reconstruction = nn.SmoothL1Loss(reduction="sum")  # SmoothL1Loss
    # criterion_reconstruction = nn.MSELoss(  reduction='sum'  ) # MSELoss
   

    # Classification
    # Weights for each class
   
    class_weights = [1,1]
    # Loss function
    criterion_classification = nn.CrossEntropyLoss(reduction="sum", weight=torch.Tensor(class_weights).to(DEVICE))
    
    #criterion_classification = nn.MSELoss(reduction="sum")

    ## Dataset choice
    #file_name = "LUNG.csv"
    file_name = "HIF2.csv"
   

    ## Choose Architecture
    #net_name = 'LeNet'
    net_name = "netBio"
    #net_name = 'FAIR'
    
    #Choose if normalisation layer in the network 
    norm = False 
    n_hidden = 100  # amount of neurons on netbio's hidden layer

    # Do pca or t-SNE
    Do_pca = True
    Do_tSNE = True
    run_model = "No_proj"  # default model run
    # Do projection at the middle layer or not
    DO_PROJ_MIDDLE = False
    # Do projection on the decoder part or not
    DO_PROJ_DECODER = False

    ETA=0.25
    
    GRADIENT_MASK = True  # Whether to do a second descent
    if GRADIENT_MASK:
        run_model = "ProjectionLastEpoch"

    ## Choose projection function
    if not GRADIENT_MASK:
        TYPE_PROJ = "No_proj"
        TYPE_PROJ_NAME = "No_proj"
    else:

        #TYPE_PROJ = ft.proj_l1ball  # projection l1
        #TYPE_PROJ = ft.bilevel_proj_l11ball  # bilevel projection l11 (col-wise zeros)
        #TYPE_PROJ = ft.proj_l11ball # Bilevel L11 projection
        #TYPE_PROJ = ft.proj_l21ball   # Bilevel projection l21
        #TYPE_PROJ = ft.proj_l1infball  # projection l1,inf 
        TYPE_PROJ = ft.bilevel_proj_l1Inftyball  # projection bilevel l1,inf
        
        TYPE_PROJ_NAME = TYPE_PROJ.__name__
        
    #TYPE_ACTIVATION = "tanh"
    #TYPE_ACTIVATION = "gelu"
    #TYPE_ACTIVATION = "relu"
    TYPE_ACTIVATION = "silu"

    AXIS = 1  #  1 for columns (features), 0 for rows (neurons)
    TOL = 1e-3  # error margin for the L1inf algorithm and gradient masking

    bW = 0.5  # Kernel size for distribution plots

    DoTopGenes = True  # Compute feature rankings
    
    DoTopFeatures = True # Compute features selected by the projection with a norm
    
    DoSparsity= True # Show the sparsity of the SAE

    # Save Results or not
    SAVE_FILE = True

    ######## Main routine ########
    
    # Output Path
    outputPath = (
        "results_stat"
        + ((not DO_PROJ_DECODER) * "_halfproj")
        + "/"
        + file_name.split(".")[0]
        + "/"
    )
    if not os.path.exists(outputPath):  # make the directory if it does not exist
        os.makedirs(outputPath)

    # Load data
    X, Y, feature_names, label_name, patient_name = ft.ReadData(
        file_name, doScale=doScale, doLog=doLog
    )

    feature_len = len(feature_names)
    class_len = len(label_name)
    print(f"Number of features: {feature_len}, Number of classes: {class_len}")
    
    

    # matrices to store accuracies
    accuracy_train = np.zeros((nfolds * len(SEEDS), class_len + 1))
    accuracy_test = np.zeros((nfolds * len(SEEDS), class_len + 1))
    # matrices to store metrics
    data_train = np.zeros((nfolds * len(SEEDS), 7))
    data_test = np.zeros((nfolds * len(SEEDS), 7))
    #matrices to store sparsity
    sparsity_matrix= np.zeros((nfolds * len(SEEDS), 1))
    
    correct_prediction = []
    seed_idx = 0
    
    
    
    for seed in SEEDS:
        
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        for fold_idx in range(nfolds):
            start_time = time.time()

            train_dl, test_dl, train_len, test_len, Ytest = ft.CrossVal(
                X, Y, patient_name, BATCH_SIZE, fold_idx, seed
            )
            print(
                "Len of train set: {}, Len of test set: {}".format(train_len, test_len)
            )
            print("----------- Start fold ", fold_idx, "----------------")
            # Define the SEED to fix the initial parameters
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # run AutoEncoder
            if net_name == "LeNet":
                net = ft.LeNet_300_100(n_inputs=feature_len, n_outputs=class_len, activation=TYPE_ACTIVATION, norm = norm).to(
                    DEVICE
                )  # LeNet
            if net_name == "netBio":
                net = ft.netBio(feature_len, class_len, n_hidden, activation=TYPE_ACTIVATION, norm = norm).to(DEVICE)  # netBio
            if net_name == "FAIR":
                net = ft.FairAutoEncoder(feature_len, class_len,n_hidden =n_hidden, activation=TYPE_ACTIVATION, norm = norm).to(DEVICE)  # FAIR

            weights_entry, spasity_w_entry = fnp.weights_and_sparsity(net, TOL)

            if GRADIENT_MASK:
                run_model = "ProjectionLastEpoch"

            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=150, gamma=0.1
            )
            data_encoder, data_decoded, epoch_loss, best_test, trained_net = ft.RunAutoEncoder(
                net,
                criterion_reconstruction,
                criterion_classification,
                train_dl,
                train_len,
                test_dl,
                test_len,
                optimizer,
                outputPath,
                TYPE_PROJ,
                seed,
                SEEDS,
                fold_idx,
                nfolds,
                LOSS_LAMBDA,
                lr_scheduler,
                N_EPOCHS,
                run_model,
                DO_PROJ_MIDDLE,
                DO_PROJ_DECODER,
                ETA,
                AXIS=AXIS,
                TOL=TOL,
                typeEpoch="Adam"
            )
            labelpredict = data_encoder[:, :-1].max(1)[1].cpu().numpy()


            weights_interim_enc , _ = fnp.weights_and_sparsity(trained_net.encoder, TOL)
            weights_interim_dec , _ = fnp.weights_and_sparsity(trained_net.decoder, TOL)


            # Do masked gradient
            if GRADIENT_MASK:
                #print("\n--------Running with masked gradient-----")
                #print("-----------------------")

                prev_data = [param.data for param in list(trained_net.parameters())]

                # Get initial network and set zeros
                # Recall the SEED to get the initial parameters
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                # run AutoEncoder
                if net_name == "LeNet":
                    net = ft.LeNet_300_100(n_inputs=feature_len, n_outputs=class_len, activation=TYPE_ACTIVATION, norm = norm).to(
                        DEVICE
                    )  # LeNet
                if net_name == "netBio":
                    net = ft.netBio(feature_len, class_len, n_hidden, activation=TYPE_ACTIVATION, norm = norm).to(DEVICE)  # netBio
                if net_name == "FAIR":
                    net = ft.FairAutoEncoder(feature_len, class_len,n_hidden = n_hidden, activation=TYPE_ACTIVATION, norm = norm).to(DEVICE)  # FAIR

                optimizer = torch.optim.Adam(trained_net.parameters(), lr=LR)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 150, gamma=0.1
                )  # unused in the paper

                net_parameters = list(trained_net.parameters())
                for index, param in enumerate(net_parameters):
                    is_middle = index == (len(net_parameters) / 2) - 1
                    is_decoder_layer = index >= len(net_parameters) / 2
                    if (
                        not DO_PROJ_MIDDLE
                    ) and is_middle:  # Do no gradient masking at middle layer
                        pass
                    elif is_decoder_layer and (
                        not DO_PROJ_DECODER
                    ):  # Do no gradient masking on the decoder layers
                        pass
                    elif index % 2 == 0:
                        param.data = torch.where(
                            prev_data[index].abs() < TOL,
                            torch.zeros_like(param.data),
                            param.data,
                        )

                run_model = "MaskGrad"
                (
                    data_encoder,
                    data_decoded,
                    epoch_loss,
                    best_test,
                    net,
                ) = ft.RunAutoEncoder(
                    trained_net,
                    criterion_reconstruction,
                    criterion_classification,
                    train_dl,
                    train_len,
                    test_dl,
                    test_len,
                    optimizer,
                    outputPath,
                    TYPE_PROJ,
                    seed,
                    SEEDS,
                    fold_idx,
                    nfolds,
                    LOSS_LAMBDA,
                    lr_scheduler,
                    N_EPOCHS_MASKGRAD,
                    run_model,
                    DO_PROJ_MIDDLE,
                    DO_PROJ_DECODER,
                    ETA,
                    AXIS=AXIS,typeEpoch=run_model
                )
                    
                    
                end_time = time.time()

                # Calculate and print the execution time
                execution_time = end_time - start_time
                print(f"Execution time for training : {execution_time} seconds for fold {fold_idx}")
                #print("\n--------Finised masked gradient-----")
                #print("-----------------------")
                
                
            data_encoder = data_encoder.cpu().detach().numpy()
            data_decoded = data_decoded.cpu().detach().numpy()

            (
                data_encoder_test,
                data_decoded_test,
                class_train,
                class_test,
                _,
                correct_pred,
                softmax,
                Ytrue,
                Ypred,
            ) = ft.runBestNet(
                train_dl,
                test_dl,
                best_test,
                outputPath,
                fold_idx,
                class_len,
                net,
                feature_names,
                test_len,
            )

            if seed == SEEDS[-1]:
                if fold_idx == 0:
                    Ytruef = Ytrue
                    Ypredf = Ypred
                    LP_test = data_encoder_test.detach().cpu().numpy()
                else:
                    Ytruef = np.concatenate((Ytruef, Ytrue))
                    Ypredf = np.concatenate((Ypredf, Ypred))
                    LP_test = np.concatenate(
                        (LP_test, data_encoder_test.detach().cpu().numpy())
                    )

            accuracy_train[seed_idx * 4 + fold_idx] = class_train
            accuracy_test[seed_idx * 4 + fold_idx] = class_test
            X_encoder = data_encoder[:, :-1]
            labels_encoder = data_encoder[:, -1]
            data_encoder_test = data_encoder_test.cpu().detach()

            # SIL score
            data_train[seed_idx * 4 + fold_idx, 0] = metrics.silhouette_score(
                X_encoder, labels_encoder, metric="euclidean"
            )

            X_encodertest = data_encoder_test[:, :-1]
            labels_encodertest = data_encoder_test[:, -1]
            data_test[seed_idx * 4 + fold_idx, 0] = metrics.silhouette_score(
                X_encodertest, labels_encodertest, metric="euclidean"
            )
            # ARI score

            data_train[seed_idx * 4 + fold_idx, 1] = metrics.adjusted_rand_score(
                labels_encoder, labelpredict
            )
            data_test[seed_idx * 4 + fold_idx, 1] = metrics.adjusted_rand_score(
                Ytest, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
            )

            # AMI Score
            data_train[seed_idx * 4 + fold_idx, 2] = metrics.adjusted_mutual_info_score(
                labels_encoder, labelpredict
            )
            data_test[seed_idx * 4 + fold_idx, 2] = metrics.adjusted_mutual_info_score(
                Ytest, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
            )
            # AUC Score
            if class_len == 2:
                data_train[seed_idx * 4 + fold_idx, 3] = metrics.roc_auc_score(
                    labels_encoder, labelpredict
                )
                data_test[seed_idx * 4 + fold_idx, 3] = metrics.roc_auc_score(
                    Ytest, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
                )

            # F1 precision recall
            data_train[seed_idx * 4 + fold_idx, 4:] = precision_recall_fscore_support(
                labels_encoder, labelpredict, average="macro"
            )[:-1]
            data_test[seed_idx * 4 + fold_idx, 4:] = precision_recall_fscore_support(
                Ytest, data_encoder_test[:, :-1].max(1)[1].numpy(), average="macro"
            )[:-1]

            # Correct labels storage
            correct_prediction += correct_pred

            # Get Top Genes of each class

            # method = 'Shap'   # (SHapley Additive exPlanation) needs a nb_samples
            nb_samples = 300  # Randomly choose nb_samples to calculate their Shap Value, time vs nb_samples seems exponential
            #method = 'Captum_ig'   # Integrated Gradients
            method = "Captum_dl"  # Deeplift
            #method = 'Captum_gs'  # GradientShap

            if DoTopGenes:
                tps1 = time.perf_counter()
                if fold_idx == 0:  # first fold, never did topgenes yet
                    print("Running topGenes...")
                    df_topGenes = ft.topGenes(
                        X,
                        Y,
                        feature_names,
                        class_len,
                        feature_len,
                        method,
                        nb_samples,
                        DEVICE,
                        net,
                        TOL
                    )
                    df_topGenes.index = df_topGenes.iloc[:, 0]
                    
                    print("topGenes finished")
                    tps2 = time.perf_counter()
                else:
                    print("Running topGenes...")
                    df_topGenes = ft.topGenes(
                        X,
                        Y,
                        feature_names,
                        class_len,
                        feature_len,
                        method,
                        nb_samples,
                        DEVICE,
                        net,
                        TOL
                    )
                    print("topGenes finished")
                    df = pd.read_csv(
                        "{}{}_topGenes_{}_{}.csv".format(
                            outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                        header=0,
                        index_col=0,
                    )
                    
                    df_topGenes.index = df_topGenes.iloc[:, 0]
                    df_topGenes = df.join(df_topGenes.iloc[:, 1], lsuffix="_",)

                df_topGenes.to_csv(
                    "{}{}_topGenes_{}_{}.csv".format(
                        outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                    ),
                    sep=";",
                )
                tps2 = time.perf_counter()
                print("Execution time topGenes  : ", tps2 - tps1)
                
            if DoSparsity: 
                mat_in = net.state_dict()["encoder.0.weight"]
                mat_col_sparsity = ft.sparsity_col(mat_in, device=DEVICE)
                sparsity_matrix[seed_idx * 4+ fold_idx, 0] = mat_col_sparsity
                
            if DoTopFeatures:
                
                weightsF, spasity_w = fnp.weights_and_sparsity(net.encoder, TOL)
                layer_list = [x for x in weightsF.values()]
                #Chercher les colonnes qui non pas que des 0
                non_zero_columns =~np.all(layer_list[0] == 0, axis=0)
                indices_non_zero_columns = np.where(non_zero_columns)[0]
                featuresSelected = feature_names[indices_non_zero_columns]
                normL =LA.norm(layer_list[0], 2, axis=0)
                normL = normL/ max(normL)
                normFeaturesSelected = normL[indices_non_zero_columns]
                #dfFeatureSelected = pd.DataFrame(featuresSelected, columns=['Fold'+str(fold_idx)])
                dfFeatureSelected = pd.DataFrame({
                        'Fold': featuresSelected,
                        'NormL2 ': normFeaturesSelected
                    })

                if fold_idx!=0:
                    df = pd.read_csv(
                        "{}{}_topFeatures_NormL2.csv".format(
                            outputPath, str(TYPE_PROJ_NAME)
                        ),
                        sep=";",
                        header=0,
                        index_col=0,
                    )
                    dfFeatureSelected = df.merge(dfFeatureSelected.iloc[:, 0:], how='left',
                            on = 'Fold', suffixes= (f"{fold_idx-1}", f"{fold_idx}"))
                dfFeatureSelected.fillna(0, inplace = True)
                dfFeatureSelected.to_csv('{}{}_topFeatures_NormL2.csv'.format(outputPath,
                            str(TYPE_PROJ_NAME)),sep=";")
                

        if seed == SEEDS[0]:
            df_softmax = softmax
            df_softmax.index = df_softmax["Name"]
            # softmax.to_csv('{}softmax.csv'.format(outputPath),sep=';',index=0)
        else:
            softmax.index = softmax["Name"]
            df_softmax = df_softmax.join(softmax, rsuffix="_")

        # Moyenne sur les SEED
        if DoTopGenes:
            df = pd.read_csv(
                "{}{}_topGenes_{}_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                ),
                sep=";",
                header=0,
                index_col=0,
            )
            df_val = df.values[1:, 1:].astype(float)
            df_mean = df_val.mean(axis=1).reshape(-1, 1)
            df_std = df_val.std(axis=1).reshape(-1, 1)
            df = pd.DataFrame(
                np.concatenate((df.values[1:, :], df_mean, df_std), axis=1),
                columns=[
                    "Features",
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Mean",
                    "Std",
                ],
            )
            df_topGenes = df
            df_topGenes = df_topGenes.sort_values(by="Mean", ascending=False)
            df_topGenes = df_topGenes.reindex(
                columns=[
                    "Features",
                    "Mean",
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Std",
                ]
            )
            df_topGenes.to_csv(
                "{}{}_topGenes_{}_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                ),
                sep=";",
                index=0,
            )

            if seed == SEEDS[0]:
                df_topGenes_mean = df_topGenes.iloc[:, 0:2]
                df_topGenes_mean.index = df_topGenes.iloc[:, 0]
            else:
                df = pd.read_csv(
                    "{}{}_topGenes_Mean_{}_{}.csv".format(
                        outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                    ),
                    sep=";",
                    header=0,
                    index_col=0,
                )
                
                df_topGenes.index = df_topGenes.iloc[:, 0]
                df_topGenes_mean = df.join(df_topGenes.iloc[:, 1], lsuffix=seed)
                

            df_topGenes_mean.to_csv(
                "{}{}_topGenes_Mean_{}_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                ),
                sep=";",
            )
        
        if DoTopFeatures:
            df = pd.read_csv(
                "{}{}_topFeatures_NormL2.csv".format(
                    outputPath, str(TYPE_PROJ_NAME)
                ),
                sep=";",
                header=0,
                index_col=0,
            )
            df_val = df.values[0:, 1:].astype(float)
            df_mean = df_val.mean(axis=1).reshape(-1, 1)
            df_std = df_val.std(axis=1).reshape(-1, 1)
            df = pd.DataFrame(
                np.concatenate((df.values[0:, :], df_mean, df_std), axis=1),
                columns=[
                    "Features",
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Mean",
                    "Std",
                ],
            )
            df_topFeatures = df
            df_topFeatures = df_topFeatures.sort_values(by="Mean", ascending=False)
            df_topFeatures = df_topFeatures.reindex(
                columns=[
                    "Features",
                    "Mean",
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Std",
                ]
            )
            df_topFeatures.to_csv(
                "{}{}_topFeatures_NormL2.csv".format(
                    outputPath, str(TYPE_PROJ_NAME)
                ),
                sep=";",
                index=0,
            )
    
            if seed == SEEDS[0]:
                df_topFeatures_mean = df_topFeatures.iloc[:, 0:2]
                df_topFeatures_mean.index = df_topFeatures.iloc[:, 0]
            else:
                df = pd.read_csv(
                    "{}{}_topFeatures_Mean_NormL2.csv".format(
                        outputPath, str(TYPE_PROJ_NAME)
                    ),
                    sep=";",
                    header=0,
                    index_col=0,
                )
                df_topFeatures.index = df_topFeatures.iloc[:, 0]
                
                df_topFeatures_mean = df.merge(df_topFeatures.iloc[:, 1], how='left',
                        on = 'Features', suffixes= (f"{seed-1}", f"{seed}"))
                
                #df.join(df_topFeatures.iloc[:, 1], lsuffix=seed)

            df_topFeatures_mean.to_csv(
                "{}{}_topFeatures_Mean_NormL2.csv".format(
                    outputPath, str(TYPE_PROJ_NAME)
                ),
                sep=";",
            )

        seed_idx += 1

    # accuracies
    df_accTrain, df_acctest = ft.packClassResult(
        accuracy_train, accuracy_test, nfolds * len(SEEDS), label_name
    )
    print("\nAccuracy Train")
    print(df_accTrain)
    print("\nAccuracy Test")
    print(df_acctest)

    # metrics
    df_metricsTrain, df_metricsTest = ft.packMetricsResult(
        data_train, data_test, nfolds * len(SEEDS)
    )

    # separation of the metrics in different dataframes
    clustering_metrics = ["Silhouette", "ARI", "AMI"]
    classification_metrics = ["AUC", "Precision", "Recall", "F1 score"]
    df_metricsTrain_clustering = df_metricsTrain[clustering_metrics]
    df_metricsTrain_classif = df_metricsTrain[classification_metrics]
    df_metricsTest_clustering = df_metricsTest[clustering_metrics]
    df_metricsTest_classif = df_metricsTest[classification_metrics]
    
    

    #print("\nMetrics Train")
    # print(df_metricsTrain_clustering)
    #print(df_metricsTrain_classif)
    print("\nMetrics Test")
    # print(df_metricsTest_clustering)
    print(df_metricsTest_classif)
    
    if DoSparsity:
        #make df for the sparsity:
        columns = (
                ["Sparsity"]
            )
        ind_df = ["Fold " + str(x + 1) for x in range(nfolds* len(SEEDS))]
    
        df_sparcity = pd.DataFrame(sparsity_matrix, index=ind_df, columns=columns)
        df_sparcity.loc["Mean"] = df_sparcity.apply(lambda x: x.mean())
        df_sparcity.loc["Std"] = df_sparcity.apply(lambda x: x.std())
        print('\n Sparsity on the encoder')
        print(df_sparcity)
        print(f'\n On average we have {round(100-df_sparcity.loc["Mean", "Sparsity"])}% features selected, thus {round(((100- df_sparcity.loc["Mean", "Sparsity"])/100)*feature_len)} features')

    # Reconstruction by using the centers in latent space and datas after interpolation
    center_mean, center_distance = ft.Reconstruction(0.2, data_encoder, net, class_len)

    # Do pca,tSNE for encoder data
    if Do_pca and Do_tSNE:
        tit = "Latent Space"
        ft.ShowPcaTsne(X, Y, data_encoder, center_distance, class_len, tit)

    if DoTopGenes:
        df = pd.read_csv(
            "{}{}_topGenes_Mean_{}_{}.csv".format(
                outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
            ),
            sep=";",
            header=0,
            index_col=0,
        )
        df_val = df.values[:, 1:].astype(float)
        df_mean = df_val.mean(axis=1).reshape(-1, 1)
        df_std = df_val.std(axis=1).reshape(-1, 1)
        df_meanstd = df_std / df_mean
        col_seed = ["Seed " + str(i) for i in SEEDS]
        df = pd.DataFrame(
            np.concatenate((df.values[:, :], df_mean, df_std, df_meanstd), axis=1),
            columns=["Features"] + col_seed + ["Mean", "Std", "Mstd"],
        )
        df_topGenes = df
        df_topGenes = df_topGenes.sort_values(by="Mean", ascending=False)
        df_topGenes = df_topGenes.reindex(
            columns=["Features", "Mean"] + col_seed + ["Std", "Mstd"]
        )
        df_topGenes.to_csv(
            "{}{}_topGenes_Mean_{}_{}.csv".format(
                outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
            ),
            sep=";",
            index=0,
        )
        
    if DoTopFeatures:
        df = pd.read_csv(
            "{}{}_topFeatures_Mean_NormL2.csv".format(
                outputPath, str(TYPE_PROJ_NAME)
            ),
            sep=";",
            header=0,
            index_col=0,
        )
        df_val = df.values[:, 1:].astype(float)
        df_mean = df_val.mean(axis=1).reshape(-1, 1)
        df_std = df_val.std(axis=1).reshape(-1, 1)
        df_meanstd = df_std / df_mean
        col_seed = ["Seed " + str(i) for i in SEEDS]
        df = pd.DataFrame(
            np.concatenate((df.values[:, :], df_mean, df_std, df_meanstd), axis=1),
            columns=["Features"] + col_seed + ["Mean", "Std", "Mstd"],
        )
        df_topFeatures = df
        df_topFeatures = df_topFeatures.sort_values(by="Mean", ascending=False)
        df_topFeatures = df_topFeatures.reindex(
            columns=["Features", "Mean"] + col_seed + ["Std", "Mstd"]
        )
        df_topFeatures.to_csv(
            "{}{}_topFeatures_Mean_NormL2.csv".format(
                outputPath, str(TYPE_PROJ_NAME)
            ),
            sep=";",
            index=0,
        )

    if class_len == 2:
        plt.figure()
        plt.title("Kernel Density")
        plt.plot([0.5, 0.5], [0, 3])
        lab = 0
        for col in softmax.iloc[:, 2:]:
            distrib = softmax[col].where(softmax["Labels"] == lab).dropna()
            if lab == 0:
                sns.kdeplot(
                    1 - distrib,
                    bw_method=bW,
                    fill=True,
                    color="tab:blue",
                    label="Proba class 0",
                )
                # sns.kdeplot(1 - distrib, bw=0.1, fill=True, shade="True")
            else:
                sns.kdeplot(
                    distrib,
                    bw_method=bW,
                    fill=True,
                    color="tab:orange",
                    label="Proba class 1",
                )
                # sns.kdeplot(distrib, bw=0.1, fill=True, shade="True")
    
            lab += 1
        plt.legend(loc="upper left")
        plt.xlabel("")
        plt.ylabel("")
        plt.show()

    spasity_percentage_entry = {}
    for keys in spasity_w_entry.keys():
        spasity_percentage_entry[keys] = spasity_w_entry[keys] * 100
    print("spasity % of all layers entry \n", spasity_percentage_entry)
    # print("-----------------------")
    weights, spasity_w = fnp.weights_and_sparsity(net.encoder, TOL)
    # spasity_percentage = {}
    # for keys in spasity_w.keys():
    #     spasity_percentage[keys] = spasity_w[keys] * 100
    # print("spasity % of all layers \n", spasity_percentage)

    weights_decoder, spasity_w_decoder = fnp.weights_and_sparsity(net.decoder, TOL)
    # spasity_percentage_decoder = {}
    # for keys in spasity_w_decoder.keys():
    #     spasity_percentage_decoder[keys] = spasity_w_decoder[keys] * 100
    # print("spasity % of all layers \n", spasity_percentage_decoder)
    # print("-----------------------")

    # mat_in = net.state_dict()["encoder.0.weight"]
    # mat_col_sparsity = ft.sparsity_col(mat_in, device=DEVICE)
    # print(" Column sparsity of input matrix: \n", mat_col_sparsity)
    # mat_in_sparsity = ft.sparsity_line(mat_in, device=DEVICE)
    # print("Line sparsity of input matrix: \n", mat_in_sparsity)
    layer_list = [x for x in weights.values()]
    layer_list_decoder = [x for x in weights_decoder.values()]
    if norm ==True :
       layer_list_decoder = [np.array(layer_list_decoder[2]).T, np.array(layer_list_decoder[0]).T]
    else:
       layer_list_decoder = [np.array(layer_list_decoder[1]).T, np.array(layer_list_decoder[0]).T]
    titile_list = [x for x in spasity_w.keys()]
    print(f"After Projection, Sum is: {np.sum(np.abs(weights_interim_enc['0.weight']))}")
    print(f"After Projection, Sum is: {np.sum(np.abs(weights_interim_dec['2.weight']))}")

    ft.show_img(layer_list, file_name)
    ft.show_img(layer_list_decoder, file_name)

    # Loss figure
    if os.path.exists(file_name.split(".")[0] + "_Loss_No_proj.npy") and os.path.exists(
        file_name.split(".")[0] + "_Loss_MaskGrad.npy"
    ):
        loss_no_proj = np.load(file_name.split(".")[0] + "_Loss_No_proj.npy")
        loss_with_proj = np.load(file_name.split(".")[0] + "_Loss_MaskGrad.npy")
        plt.figure()
        plt.title(file_name.split(".")[0] + " Loss")
        plt.xlabel("Epoch")
        plt.ylabel("TotalLoss")
        plt.plot(loss_no_proj, label="No projection")
        plt.plot(loss_with_proj, label="With projection ")
        plt.legend()
        plt.show()
    if SAVE_FILE:
        df_acctest.to_csv(
            "{}{}_acctest.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
        )
        df_metricsTest_classif.to_csv(
            "{}{}_auctest.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
        )
        if DoSparsity:
            df_sparcity.to_csv(
                "{}{}_sparsity.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
                )

        print("Save topGenes results to: ' {} ' ".format(outputPath))
    end_time= time.time()
    execution_time=end_time-start_time
    print(f"Execution time: {execution_time} seconds")

    print("Save topGenes results to: ' {} ' ".format(outputPath))
    
    

