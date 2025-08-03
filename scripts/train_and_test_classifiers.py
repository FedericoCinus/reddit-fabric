"""Train and test models in classification
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, 
                             brier_score_loss, make_scorer)

from imblearn.over_sampling import RandomOverSampler
import scipy
import sys
import time
sys.path += ['../src/']
from model_baseline import RawActivityModel
from model_nature import Nature_model
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, set_mkl_threads
import yaml

# Define colors for plots
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.keys())


set_mkl_threads(8)

# ---- Main Script ----

def main():
    time0 = time.time()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--n_seeds", type=int, help='Number of seeds.')
    parser.add_argument("-g", "--run_grid_search", action='store_true', help='Whether to run grid search for hyperparameter tuning')
    parser.add_argument("-a", "--additional_users", action='store_true', help='Adds additional data to distance supervision. Firstly, it labels users in other dimensions. Then it adds these data to the training.')
    parser.add_argument("-t", "--test_size", type=float, help='Percentage for test size', default=0.2)
    parser.add_argument("-d", "--debug", help='Debug mode', action='store_true')
    args = vars(parser.parse_args())

    if args['debug']: print("Setting debug mode on ..")
    
    # 1. Define paths
    with open("./parameters.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    dataset_path = Path(parameters['dataset_path'])
    input_path = Path(parameters['input_path'])
    folder_results = Path(parameters['folder_results'])
    score_file = input_path / Path('scores.big.csv')
    subreddit_names = np.load(input_path / Path("list_subreddits.npy"), allow_pickle=True)
    DIMENSIONS = ['year', 'gender', 'demo_rep']
    
    results_class = {'model': [], 'score': [], 'metric': [], 'use_declaration_in_train': [], 'dimension': [], 'seed': [],
                     'Class counts test': [], 'Class counts train': [], 'Class counts declarations': [], 'n samples train': [], 'n samples test': [],
                     'hyper-param grid search': []}


    for dimension in DIMENSIONS:
        print(f"\n\n---> DIMENSION: {dimension}")
        # 2. Load data
        print(" -- declaration data:")
        X_declarations, y_declarations = load_data(attribute_path = dataset_path / Path(f"{dimension}__body"),
                                                      attribute_to_label = None,
                                                      input_path = input_path,
                                                      use_bool_features = False,
                                                      save = True)
        print("\n -- distance-supervision data:")
        X_nature_seeds, y_nature_seeds = load_data(attribute_path = dataset_path / Path(f"{dimension}__nature_seeds"),
                                                      attribute_to_label = dataset_path / Path(f"{dimension}__nature_seeds"),
                                                      input_path = input_path,
                                                      use_bool_features = False,
                                                      save = True)
        if args['additional_users']:
            addit_X_nature_seeds = []
            addit_y_nature_seeds = []
            for additional_dim in DIMENSIONS:
                if additional_dim != dimension:
                    __X_nature_seeds, __y_nature_seeds = load_data(attribute_path = dataset_path / Path(f"{additional_dim}__nature_seeds"),
                                                                    attribute_to_label = dataset_path / Path(f"{dimension}__nature_seeds"),
                                                                    input_path = input_path,
                                                                    use_bool_features = False,
                                                                    save = True)
                    addit_X_nature_seeds.append(__X_nature_seeds)
                    addit_y_nature_seeds.append(__y_nature_seeds)

        

        for use_declaration_in_train in [False, True]:
            print(f"\n ---> use_declaration_in_train = {use_declaration_in_train}")
            
            # Dataset size
            if use_declaration_in_train:
                size = int(X_declarations.shape[0]*(1-args['test_size']))
            elif args['additional_users']:
                size = sum([X_nature_seeds.shape[0]] + [addit_data.shape[0] for addit_data in addit_X_nature_seeds])
            else:
                size = X_nature_seeds.shape[0]
            print(f"  Final training dataset size: {size}")


            for seed in range(args['n_seeds']):
                print(f" seed: {seed}", end="\r")
                # 3. Shuffle all arrays using the same indices, NB: nature seeds are not shuffled because we use all of them
                np.random.seed(seed)
                ## shuffle declarations (train + test)
                indices = np.random.permutation(len(y_declarations))
                y = y_declarations[indices]
                y_true = y_declarations[indices]
                X = X_declarations[indices]
                ## shuffle nature seeds labels (only train) -- performed later
                indices = np.random.permutation(len(y_nature_seeds))
                __y_nature_seeds_shuffled = y_nature_seeds[indices]
                __X_nature_seeds_shuffled = X_nature_seeds[indices]

                # 4. Split data into training and testing sets
                num_train = int(X.shape[0]*(1-args['test_size']))

                X_train = X[:num_train, :] if use_declaration_in_train else X_nature_seeds
                X_test = X[num_train:, :]
                y_train = y_true[:num_train] if use_declaration_in_train else y_nature_seeds
                y_test = y_true[num_train:]
                
                # this step appends all other dimensions data to the current dimension data, labeled using distance supervision
                if args['additional_users'] and not use_declaration_in_train:
                    X_train = scipy.sparse.vstack([X_train] + addit_X_nature_seeds)
                    y_train = np.hstack([y_train] + addit_y_nature_seeds)
                    
                    indices = np.random.permutation(len(y_train))
                    X_train = X_train[indices]
                    y_train = y_train[indices]
                
                # balance train
                if not use_declaration_in_train:
                    np.random.seed(0)
                    rus = RandomOverSampler(sampling_strategy=1)
                    X_train, y_train = rus.fit_resample(X_train, y_train)

                


                # 5. Initialize models
                model_NB = MultinomialNB()
                attribute = dimension
                if attribute == "demo_rep": attribute = "partisan"
                nature_model = Nature_model(score_file=score_file, subreddit_names=subreddit_names, attribute=attribute)
                majority_model = RawActivityModel(subreddit_names, dimension, 5)
                random_forest = RandomForestClassifier(n_estimators=50, max_depth=10)

                # 6. Predict the labels for the test set
                for name, model in [("model_NB", model_NB),
                                    ("nature_model", nature_model),
                                    ("majority_model", majority_model),
                                    ("random_forest", random_forest)
                                    ]:
                    
                    # Grid search on NB
                    if name == "model_NB" and args["run_grid_search"]:
                        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
                        f1_weighted_scorer = make_scorer(f1_score, average='weighted')
                        grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring=f1_weighted_scorer)
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_


                    model.fit(X_train, y_train)
                    
                    # Classification
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for imbalanced classes
                    accuracy = accuracy_score(y_test, y_pred)
                    if args['debug']:
                        print(f"{name}:  f1={f1:.3f},  acc={accuracy:.3f}")
                        import pdb; pdb.set_trace()
                    
                    # Proba
                    y_probs = model.predict_proba(X_test)
                    if y_probs.shape[1] == 2:
                        # Use the probabilities of the positive class
                        y_probs_pos = y_probs[:, 1]
                        roc_auc = roc_auc_score(y_test, y_probs_pos)
                        brier_score = brier_score_loss(y_test, y_probs_pos)
                        fpr, tpr, thresholds = roc_curve(y_test, y_probs_pos)

                    # Save feature importance
                    if name == "model_NB":
                        folder = Path("../data/results/log_probs")
                        folder.mkdir(parents=True, exist_ok=True)

                        log_prob = model.feature_log_prob_
                        title = f"dimension{dimension}-use_declaration_in_train{use_declaration_in_train}-seed{seed}"
                        np.save(folder / Path(title + "log_prob.npy"), log_prob)
                        np.save(folder / Path(title + "y_pred.npy"), y_pred)
                        np.save(folder / Path(title + "y_test.npy"), y_test)


                    for score, metric in zip([accuracy, f1, roc_auc, brier_score, fpr, tpr, thresholds],
                                             ["accuracy", "f1", "roc_auc", "brier_score", "fpr", "tpr", "thresholds"]):
                        results_class['model'].append(name)
                        if isinstance(score, (list, np.ndarray)):
                            results_class['score'].append(list(score))
                        else:
                            results_class['score'].append(score)
                        results_class['metric'].append(metric)
                        results_class['use_declaration_in_train'].append(use_declaration_in_train)
                        results_class['dimension'].append(dimension)
                        results_class['seed'].append(seed)
                        _, counts = np.unique(y_test, return_counts=True)
                        results_class['Class counts test'].append(counts)
                        _, counts = np.unique(y_train, return_counts=True)
                        results_class['Class counts train'].append(counts)
                        _, counts = np.unique(y_true, return_counts=True)
                        results_class['Class counts declarations'].append(counts)
                        results_class['n samples train'].append(len(y_train))
                        results_class['n samples test'].append(len(y_test))
                        results_class['hyper-param grid search'].append(args["run_grid_search"])
                        
    if not args['debug']:
        pd.DataFrame(results_class).to_csv(folder_results / Path("classification.csv"), index=False)
    print(f"Finished in {(time.time() - time0) / 60:.3f} min")

if __name__ == "__main__":
    main()