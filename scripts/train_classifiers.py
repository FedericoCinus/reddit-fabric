"""Train and save models
"""



import argparse
from joblib import dump
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             roc_auc_score
                             )
import sys
import time
sys.path += ['../src/']
from model_baseline import RawActivityModel
from model_nature import Nature_model
from utils import load_data, set_mkl_threads
import yaml


set_mkl_threads(8)

# ---- Main Script ----

def main():
    time0 = time.time()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, help='Seed', default=0)
    parser.add_argument("-t", "--test_size", type=float, help='Percentage for test size', default=0.05)
    parser.add_argument("-d", "--debug", help='Debug mode', action='store_true')
    args = vars(parser.parse_args())

    # Parameters
    DIMENSIONS = ['year', 'gender', 'demo_rep']
    


    if args['debug']: print("Setting debug mode on ..")
    seed = args['seed']
    
    # 1. Define paths
    with open("./parameters.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    dataset_path = Path(parameters['dataset_path'])
    input_path = Path(parameters['input_path'])
    folder_results = Path(parameters['folder_results'])
    folder_results.mkdir(parents=True, exist_ok=True)
    score_file = input_path / Path('scores.big.csv')
    folder_models = Path(parameters['folder_models'])
    folder_models.mkdir(parents=True, exist_ok=True)

    subreddit_names = np.load(input_path / Path("list_subreddits.npy"), allow_pickle=True)
    
    results_class = {'model': [],
                     'score': [], 
                     'metric': [], 
                     'dimension': [], 
                     'seed': [],
                     'Class counts test': [], 
                     'Class counts train': [], 
                     'Class counts declarations': [], 
                     'n samples train': [], 
                     'n samples test': [],
                     }


    for dimension in DIMENSIONS:
        print(f"\n\n---> DIMENSION: {dimension}")
        # 2. Load data
        print(" -- declaration data:")
        X_declarations, y_declarations = load_data(attribute_path = dataset_path / Path(f"{dimension}__body"),
                                                      attribute_to_label = None,
                                                      input_path = input_path,
                                                      use_bool_features = False,
                                                      save = True)

        

        
        # Dataset size
        size = int(X_declarations.shape[0]*(1-args['test_size']))
        print(f"  Final training dataset size: {size}")


        print(f" seed: {seed}", end="\r")
        # 3. Shuffle all arrays using the same indices, NB: nature seeds are not shuffled because we use all of them
        np.random.seed(seed)
        ## shuffle declarations (train + test)
        indices = np.random.permutation(len(y_declarations))
        y_true = y_declarations[indices]
        X = X_declarations[indices]

        # 4. Split data into training and testing sets
        num_train = int(X.shape[0]*(1-args['test_size']))

        X_train = X[:num_train, :]
        X_test = X[num_train:, :]
        y_train = y_true[:num_train]
        y_test = y_true[num_train:]
        


        # 5. Initialize models
        model_NB = MultinomialNB()
        attribute = dimension
        if attribute == "demo_rep": attribute = "partisan"
        nature_model = Nature_model(score_file=score_file, subreddit_names=subreddit_names, attribute=attribute)
        majority_model = RawActivityModel(subreddit_names, dimension, 5)

        # 6. Predict the labels for the test set
        for name, model in [("model_NB", model_NB),
                            ("nature_model", nature_model),
                            ("majority_model", majority_model),
                            ]:
            
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

            # Save Model and Feature importance
            if name == "model_NB":
                __dimension_name = 'partisan' if dimension == 'demo_rep' else dimension

                # A. model
                dump(model, folder_models / Path(f"mnb_classifier-{__dimension_name}-seed{seed}.joblib"))

                # B. feature importance
                log_prob = model.feature_log_prob_
                
                title = f"scores-{__dimension_name}-seed{seed}"
                np.save(folder_models / Path(title + "log_prob.npy"), log_prob)


            for score, metric in zip([accuracy, f1, roc_auc], ["accuracy", "f1", "roc_auc"]):
                results_class['model'].append(name)
                if isinstance(score, (list, np.ndarray)):
                    results_class['score'].append(list(score))
                else:
                    results_class['score'].append(score)
                results_class['metric'].append(metric)
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
                        
    if not args['debug']:
        pd.DataFrame(results_class).to_csv(folder_results / Path("classification_final_models.csv"), index=False)
    print(f"Finished in {(time.time() - time0) / 60:.3f} min")

if __name__ == "__main__":
    main()