"""Train and test models in quantification
"""
SAMPLE_SIZE = 5
import argparse
from joblib import dump
import numpy as np
import pandas as pd
from pathlib import Path
import quapy as qp
qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE
from sklearn.naive_bayes import MultinomialNB
import time
import yaml

import sys
sys.path += ['../src/']
from model_baseline import RawActivityModel
from model_nature import Nature_model
from utils import load_data, set_mkl_threads


set_mkl_threads(8)

def init_models(dimension, X_test, score_file, subreddit_names):
    """Initialize models with their respective parameters."""
    # Define and initialize your models here
    attribute = dimension
    if attribute == "demo_rep": attribute = "partisan"


    MODELS = {
        'Multinomial NB': MultinomialNB(),
        'Majority Model':  RawActivityModel(subreddit_names, dimension, 5),
        'Nature Model': Nature_model(score_file=score_file, subreddit_names=subreddit_names, attribute=attribute),
    }
    return MODELS




# ---- Main Script ----
def main():
    time0 = time.time()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, help='Seed', default=0)
    parser.add_argument("-t", "--test_size", type=float, help='Percentage for test size', default=0.05)
    args = vars(parser.parse_args())

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
    DIMENSIONS = ['year', 'gender', 'demo_rep']
    test_size = args["test_size"]
    seed = args['seed']
    
    
    results_quant = {'model': [], 
                     'score': [], 
                     'metric': [],
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
                }
    

    for dimension in DIMENSIONS:
        print(f"\n\n---> DIMENSION: {dimension}")
        # 2. Load data
        X_declarations, y_declarations = load_data(attribute_path = dataset_path / Path(f"{dimension}__body"),
                                                      attribute_to_label = None,
                                                      input_path = input_path,
                                                      use_bool_features = False,
                                                      save = True)

        
        print(f"\n   seed: {seed}")
        # 3. Shuffle all arrays using the same indices, NB: nature seeds are not shuffled because we use all of them
        np.random.seed(seed)
        ## shuffle declarations (train + test)
        indices = np.random.permutation(len(y_declarations))
        y_true = y_declarations[indices]
        X = X_declarations[indices]


        # 4. Split data into training and testing sets
        TRAIN_PERC, _TEST_PERC = (1-test_size, test_size)
        max_num_train = int(TRAIN_PERC * X.shape[0]) # max num samples depends on the dimension
        
        X_train = X[:max_num_train, :]
        X_test = X[max_num_train:, :]
        y_train = y_true[:max_num_train]
        y_test = y_true[max_num_train:]


        training_set = qp.data.LabelledCollection(X_train, y_train)
        test_set = qp.data.LabelledCollection(X_test, y_test)
        my_dataset = qp.data.Dataset(training_set, test_set)


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

                    # Save Model
                    if name == 'Multinomial NB':
                        __dimension_name = 'partisan' if dimension == 'demo_rep' else dimension
                        dump(model_aggr, folder_models / Path(f"mnb_quantifier-{__dimension_name}-seed{seed}.joblib"))

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
                    
    pd.DataFrame(results_quant).to_csv(folder_results / Path("quantification.csv"), index=False)
    print(f"Finished in {(time.time() - time0) / 60:.3f} min")

if __name__ == "__main__":
    main()