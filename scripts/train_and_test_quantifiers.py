"""Train and test models in quantification
"""
SAMPLE_SIZE = 50
import argparse
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from pathlib import Path
import quapy as qp
qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import time
import yaml

import sys
sys.path += ['../src/']
from model_baseline import RawActivityModel
from model_nature import Nature_model
from model_SDI import SS_NB_logGMM
from utils import load_data, set_mkl_threads


set_mkl_threads(8)

def init_models(dimension, X_test, score_file, subreddit_names):
    """Initialize models with their respective parameters."""
    # Define and initialize your models here
    attribute = dimension
    if attribute == "demo_rep": attribute = "partisan"


    MODELS = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10),
        'Multinomial NB': MultinomialNB(),
        'Majority Model':  RawActivityModel(subreddit_names, dimension, 5),
        'Nature Model': Nature_model(score_file=score_file, subreddit_names=subreddit_names, attribute=attribute),
        'SS_NB_logGMMT': SS_NB_logGMM(X_u=X_test, unsupervised_perc=1., beta=0.01, threshold=0.6,), # SS models will be trained in the predict method --> X_u is passed there
        'NB_logGMMT': SS_NB_logGMM(X_u=X_test, unsupervised_perc=0., beta=0.01, gamma=1., threshold=0.6),
        'SS_NB': SS_NB_logGMM(X_u=X_test, unsupervised_perc=1., beta=0.01, gamma=0., threshold=0.6,)
    }
    return MODELS




# ---- Main Script ----
def main():
    time0 = time.time()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--n_seeds", type=int, help='Number of seeds.')
    parser.add_argument("-g", "--run_grid_search", action='store_true', help='Whether to run grid search for hyperparameter tuning')
    parser.add_argument("-a", "--additional_users", action='store_true', help='Adds additional data to distance supervision. Firstly, it labels users in other dimensions. Then it adds these data to the training.')
    parser.add_argument("-t", "--test", action='store_true', help='Not saving')
    args = vars(parser.parse_args())

    # 1. Define paths
    with open("./parameters.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    dataset_path = Path(parameters['dataset_path'])
    input_path = Path(parameters['input_path'])
    folder_results = Path(parameters['folder_results'])
    score_file = input_path / Path('scores.big.csv')
    subreddit_names = np.load(input_path / Path("list_subreddits.npy"), allow_pickle=True)
    DIMENSIONS = ['year', 'gender', 'demo_rep']
    
    
    results_quant = {'model': [], 'score': [], 'metric': [],
                     'use_declaration_in_train': [], 
                     'dimension': [], 
                     'seed': [],
                     'Class counts test': [], 
                     'Class counts train': [], 
                     'Class counts declarations': [], 
                     'n samples': [],
                     'n samples train': [], 
                     'n samples test': [],
                     'perc samples train': [], 
                     'perc samples test': [],
                     'hyper-param grid search': []
                }
    

    for dimension in DIMENSIONS:
        print(f"\n\n---> DIMENSION: {dimension}")
        # 2. Load data
        X_declarations, y_declarations = load_data(attribute_path = dataset_path / Path(f"{dimension}__body"),
                                                      attribute_to_label = None,
                                                      input_path = input_path,
                                                      use_bool_features = False,
                                                      save = True)
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
            print(f"\n use_declaration_in_train = {use_declaration_in_train}")



            for seed in range(args['n_seeds']):
                print(f"\n   seed: {seed}/{args['n_seeds']}")
                # 3. Shuffle all arrays using the same indices, NB: nature seeds are not shuffled because we use all of them
                np.random.seed(seed)
                ## shuffle declarations (train + test)
                indices = np.random.permutation(len(y_declarations))
                y_true = y_declarations[indices]
                X = X_declarations[indices]
                ## shuffle nature seeds labels (only train)
                indices = np.random.permutation(len(y_nature_seeds))
                y_nature_seeds_shuffled = y_nature_seeds[indices]
                X_nature_seeds_shuffled = X_nature_seeds[indices]


                # 4. Split data into training and testing sets
                TRAIN_PERC, _TEST_PERC = (0.7, 0.3)
                max_num_train = int(TRAIN_PERC * X.shape[0]) # max num samples depends on the dimension
                train_sizes = np.append(np.logspace(2, int(np.log10(max_num_train)), num=25, dtype=int), max_num_train) if use_declaration_in_train else [-1,]
                for num_train in train_sizes[::-1]:
                    print()
                    if num_train + 10 > X.shape[0]:
                        raise Exception(f"Exceeding dataset size in {dimension}! {num_train}, {X.shape[0]}")

                    X_train = X[:num_train, :] if use_declaration_in_train else X_nature_seeds_shuffled[:num_train, :]
                    X_test = X[max_num_train:, :]
                    y_train = y_true[:num_train] if use_declaration_in_train else y_nature_seeds_shuffled[:num_train]
                    y_test = y_true[max_num_train:]

                    if args['additional_users'] and not use_declaration_in_train: # this step appends all other dimensions data to the current dimension data, labeled using distance supervision
                        X_train = scipy.sparse.vstack([X_train] + addit_X_nature_seeds)
                        y_train = np.hstack([y_train] + addit_y_nature_seeds)
                        
                        indices = np.random.permutation(len(y_train))
                        X_train = X_train[indices]
                        y_train = y_train[indices]

                        #X_train, y_train = remove_multilabeled_rows(X_train, y_train) # Removing equal rows with multiple different labels (in pratice it does have effect)
                    

                    # balance train
                    if not use_declaration_in_train:
                        np.random.seed(0)
                        rus = RandomOverSampler(sampling_strategy=1)
                        X_train, y_train = rus.fit_resample(X_train, y_train)

                    training_set = qp.data.LabelledCollection(X_train, y_train)
                    test_set = qp.data.LabelledCollection(X_test, y_test)
                    my_dataset = qp.data.Dataset(training_set, test_set)


                    # # 5. Define evaluation protocol
                    # prot = qp.protocol.NPP(test_set, repeats=100, random_state=seed) # Since we use Natural protocol the SS models can see the whole test
                    # errors = qp.evaluation.evaluate(model_aggr, protocol=prot, error_metric='ae')

                    # 5. Define evaluation protocol & Unpack evaluation 
                    protocol = qp.protocol.NPP(test_set, repeats=SAMPLE_SIZE, random_state=0)
                    true_prevs, sample_instances_set = [], []
                    for sample_instances, sample_prev in protocol():
                        sample_instances_set.append(sample_instances)
                        true_prevs.append(sample_prev)
                    

                    model2errors = {}
                    i = 1
                    for _true_prev, sample_instances in zip(true_prevs, sample_instances_set):
                        print(f"{i}/{len(true_prevs)} samples", end="\r")
                        MODELS = init_models(dimension, sample_instances, score_file, subreddit_names) # Initialize everytime the SS models with the new test set (sample_instances)
                        for name, model in MODELS.items():
                            aggregations = [("", qp.method.aggregative.ACC(model)), ("_CC", qp.method.aggregative.CC(model))] if name in ('Nature Model') else [("", qp.method.aggregative.ACC(model)),]
                            for name_agg, model_aggr in aggregations:
                                name += name_agg
                                
                                model_aggr.fit(my_dataset.training)

                                # Calculate the error
                                AE = qp.error.ae(_true_prev, model_aggr.quantify(sample_instances))

                                if name in model2errors:
                                    model2errors[name].append(AE)
                                else:
                                    model2errors[name] = [AE]
                        i += 1
                    
                
                    for name, errors in model2errors.items():
                        print(f'{name}  MAE={np.mean(errors):.3f}')
                        results_quant['model'].append(name)
                        results_quant['metric'].append('ae')
                        results_quant['score'].append(errors)
                        results_quant['use_declaration_in_train'].append(use_declaration_in_train)
                        results_quant['dimension'].append(dimension)
                        results_quant['seed'].append(seed)
                        _, counts = np.unique(y_test, return_counts=True)
                        results_quant['Class counts test'].append(counts)
                        _, counts = np.unique(y_train, return_counts=True)
                        results_quant['Class counts train'].append(counts)
                        _, counts = np.unique(y_true, return_counts=True)
                        results_quant['Class counts declarations'].append(counts)
                        results_quant['n samples'].append(X.shape[0])
                        results_quant['n samples train'].append(len(y_train))
                        results_quant['n samples test'].append(len(y_test))
                        results_quant['perc samples train'].append(len(y_train)/X.shape[0])
                        results_quant['perc samples test'].append(len(y_test)/X.shape[0])
                        results_quant['hyper-param grid search'].append(args["run_grid_search"])
                    
    if not args['test']:
        pd.DataFrame(results_quant).to_csv(folder_results / Path("quantification.csv"), index=False)
    print(f"Finished in {(time.time() - time0) / 60:.3f} min")

if __name__ == "__main__":
    main()