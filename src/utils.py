import ctypes
import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm



################################################################################################
####################################        Format              ################################
################################################################################################
def load_data(attribute_path, attribute_to_label, input_path, use_bool_features=False, save=False, unsupervised_path=None):
    """attribute_path: Path : where the attribute csv are stored
       attribute_to_label: Path or str: used to label the users in attribute_path according to attribute_to_label's seeds in the Nature paper
    """

    # 1. Useful files
    subreddit_names = np.load(input_path / Path("list_subreddits.npy"), allow_pickle=True)

    # 2. Defining paths
    suffix = "_b" if use_bool_features else ""
    y_npy_path = attribute_path / Path(f"y{suffix}.npy")
    X_npy_path = attribute_path / Path(f"X{suffix}.npz")
    X_u_npy_path = attribute_path / Path(f"X_u{suffix}.npz")
    if str(attribute_path).endswith("__nature_seeds"): # y will contain the seed labeling and y_true the declarations
        y_true_npy_path = attribute_path / Path(f"y_true_declarations_{suffix}.npy")
    
    # 3. If the scipy sparse matrices exist --> LOADING
    if y_npy_path.exists() and X_npy_path.exists():
        print(f"\n   Scipy sparse data matrices already exist at {attribute_path}:  loading them")
        y = np.load(y_npy_path, allow_pickle=True)
        X = scipy.sparse.load_npz(X_npy_path)

    
    # 4. Else --> CREATION
    else:
        print(f"\n   Scipy sparse data matrices do not exist: creating them ..")
        if str(attribute_path).endswith("__nature_seeds"):
            y, X = create_sparse_matrices_from_seeds(subreddit_names, use_bool_features,
                                                     attribute_path, attribute_to_label,
                                                     y_npy_path, X_npy_path, y_true_npy_path, save=save)
        else:
            y, X = create_sparse_matrices(subreddit_names, use_bool_features,
                                          attribute_path, unsupervised_path,
                                          y_npy_path, X_npy_path, X_u_npy_path, save=save)
        print(f"\n   .. Done")
    
    print(f"Loaded data X:{X.shape}, y:{y.shape}  ✅")
    if str(attribute_path).endswith("__nature_seeds"):
        return X, y
    return X, y

def get_labeling_map(attribute_path: Path, fullname: bool = False) -> dict:
    '''Returns the labeling convention for gender and partisan (demo_rep)
    '''
    if "gender" in str(attribute_path):
        label2index = {'m': 0, 'f': 1} if not fullname else  {'Male': 0, 'Female': 1}
    elif "demo_rep" in str(attribute_path):
        label2index = {'democrat': 0, 'republican': 1}  if not fullname else  {'Democrat': 0, 'Republican': 1}
    else:
        raise Exception(f"{attribute_path} does not contain a known attribute for labeling")
    return label2index

def create_sparse_matrices(subreddit_names: list, use_bool_features: bool,
                           attribute_path: Path, unsupervised_path: Path = None,
                           y_npy_path: Path = None, X_npy_path: Path = None, 
                           X_u_npy_path: Path = None, save: bool = False, verbose: bool = False) -> (csr_matrix, csr_matrix):
    """Creates, processes and saves the X, and y matrices for the classification task.
      unsupervised_path: is not currently used
    """
    data_stats = {}

    # Create subreddit to index mapping
    subreddit2index = {sub: i for i, sub in enumerate(subreddit_names)}

    # Load and process y data
    y_df = pd.read_csv(attribute_path / "y.csv")

    if y_df.label.dtype == np.int64: # Year of birth or Age
        _threshold = y_df['label'].median()
        y_df['label_num'] = (y_df['label'] > _threshold).astype(int)
        index2label = {0: "old", 1: 'young'} if "year" in str(attribute_path) else {1: "old", 0: 'young'}
    elif y_df.label.dtype in (str, 'object'): # Gender and Partisan
        y_df['label'] = y_df['label'].str.lower()
        label2index = get_labeling_map(attribute_path)
        index2label = {v: k for k, v in label2index.items()}
        y_df['label_num'] = y_df['label'].map(label2index)
    else:
        raise Exception(f"{y_df.label.dtype} not implemented")
  
    print(f"   number of declarations: {len(y_df)}")
    data_stats["Declarations"] = [len(y_df)]
    print(f"   number of users: {y_df.author.nunique()}")
    data_stats["Users"] = [y_df.author.nunique()]

    # Identify non-active users
    X_df = pd.read_csv(attribute_path / "X.csv")
    non_active_users = set(y_df.author) - set(X_df.author.unique())
    print(f"   number of non active users: {len(non_active_users)}")

    data_stats["Non-active users"] = [len(non_active_users)]

    # Identify ambiguous users
    ambiguous_users = (
        y_df
        .assign(label_count=y_df.groupby('author')['label_num'].transform('count'))
        .loc[lambda df: df['label_count'] > 1]
        .drop(columns='label_count')
        )
    ambiguous_users = set(ambiguous_users.author.unique())
    print(f"   number of ambiguous users: {len(ambiguous_users)}")
    
    data_stats["Ambiguous users"] = [len(ambiguous_users)]


    # In age we can indentify users that are outside our studied distribution range 
    if "age" in str(attribute_path) or "year" in str(attribute_path):
        min_range, max_range = (y_df['label'].quantile(.5), y_df['label'].quantile(.5)) # prev .25, .75
        outsiders = set(y_df[(y_df.label > min_range) & (y_df.label < max_range)].author)
    else:
        outsiders = set()

    # Filter y data
    y_df = y_df[~y_df.author.isin(non_active_users.union(ambiguous_users))].drop_duplicates() #removing non-active users and duplicates
    if outsiders:
        y_df = y_df[~y_df.author.isin(outsiders)].drop_duplicates() #removing users outside the studied label range

    author2index = {auth: i for i, auth in enumerate(y_df.author)}
    y = y_df.label_num.to_numpy()
    assert len(y_df.author) == len(y), (len(y_df.author), len(y))


    # Compute class imbalance
    value_counts = y_df.label_num.value_counts(normalize=True) * 100
    data_stats['Class Proportion'] = [f"{index2label[0]}: {value_counts[0]:.2f}%, {index2label[1]}: {value_counts[1]:.2f}%"]

    # Create sparse matrices for X data
    print("    creating scipy matrices")
    for name, path in [("supervised", attribute_path),]: #  ("unsupervised", unsupervised_path)
        if path:
            X_df = pd.read_csv(path / "X.csv")
            
            # Filter authors that are not in the mapping
            X_df = X_df[X_df['author'].isin(set(author2index.keys()))]
            
            # Map author and subreddit names to their indices
            X_df['author_idx'] = X_df['author'].map(author2index)
            X_df['subreddit_idx'] = X_df['subreddit'].map(subreddit2index)

            # Create the sparse matrix
            __X = coo_matrix((np.ones(len(X_df)), (X_df['author_idx'].values, X_df['subreddit_idx'].values)), 
                                          shape=(len(y_df.author), len(subreddit_names)))
            # Boolean features
            print(f"      creating boolean X matrix: {use_bool_features}")
            if use_bool_features:
                __X = (__X > 0.).astype(int)

            if name == "unsupervised":
                X_u = __X.tocsr()
            else:
                X = __X.tocsr()

    assert X.shape[0] == len(y), (f"len(authors)={len(y_df.author)}, X.shape[0]={X.shape[0]}, X.shape[1]={X.shape[1]}")


    # Count active subreddits
    data_stats['Active subreddits'] = [np.sum(X.sum(axis=0) > 0)]

    # Save y data and sparse matrices
    if save:
        np.save(y_npy_path, y)
        scipy.sparse.save_npz(X_npy_path, X)
        if 'unsupervised' in locals():
            scipy.sparse.save_npz(X_u_npy_path, X_u)


    if not verbose:
        return y, X
    return y, X, data_stats
    
def create_sparse_matrices_from_seeds(subreddit_names: list, 
                                      use_bool_features: bool,
                                      attribute_path: Path, 
                                      attribute_to_label: Path,
                                      y_seeds_npy_path: Path = None, 
                                      X_npy_path: Path = None, 
                                      y_true_npy_path: Path = None,
                                      save: bool = False, 
                                      k: int = 4) -> (csr_matrix, np.array, np.array):
    """Creates, processes and saves the X, and y matrices for the classification task.
        Starts from matrices created using declarations and produce the X, y versions using the same
        labeling of the Nature paper, which labels users using their partecipation to popular discriminative subreddits:
        ['AskMen', 'AskWomen'], ['democrats', 'Conservative'], ['teenagers', 'RedditForGrownups'].

        NB: You can label the activity of users in attribute_path using the seeds of attribute_to_label; where attribute_path != attribute_to_label
    """

    # Load numpy data
    _y, X = create_sparse_matrices(subreddit_names, use_bool_features, attribute_path)
    
    # 1. Label y with Seeds (subreddits) from nature paper 
    y_seed_labeled, X_seed_labeled = label_with_seeds(X, subreddit_names, attribute_to_label, k=k)



    # Save
    if save:
        np.save(y_seeds_npy_path, y_seed_labeled)
        scipy.sparse.save_npz(X_npy_path, X_seed_labeled)

    return y_seed_labeled, X_seed_labeled

def label_with_seeds(X, subreddit_names, attribute_to_label, k=4, subselect=True):
    # 1. Definig Seeds (subreddits) from nature paper 
    # "Quantifying social organization and political polarization in online platforms"
    columns1, columns2 = (None, None)
    if "gender" in str(attribute_to_label):
        columns1 = ['AskMen', 'OneY', 'ROTC', 'predaddit']
        columns2 = ['AskWomen', 'women', 'USMilitarySO', 'BabyBumps']
    elif "demo_rep" in str(attribute_to_label):
        columns1 = ['democrats', 'GamerGhazi', 'AskAnAmerican', 'lastweektonight']
        columns2 = ['Conservative', 'KotakuInAction', 'askaconservative', 'CGPGrey']
    elif "age" in str(attribute_to_label):
        columns1 = ['teenagers', 'AskMen', 'trackandfield', 'RedHotChiliPeppers']
        columns2 = ['RedditForGrownups', 'AskMenOver30', 'trailrunning', 'pearljam']
    elif "year" in str(attribute_to_label): # birth year and age have opposite labels
        columns1 = ['RedditForGrownups', 'AskMenOver30', 'trailrunning', 'pearljam']
        columns2 = ['teenagers', 'AskMen', 'trackandfield', 'RedHotChiliPeppers']
    assert columns1 and columns2

    return_index = lambda s: np.where(subreddit_names == s)[0][0]
    columns_idx1 = list(map(return_index, columns1)) # class 1
    columns_idx2 = list(map(return_index, columns2)) # class 2

    if subselect:
        threshold = 3
        print("Columns selected:", columns_idx1, columns_idx2)
        X1 = X[:, columns_idx1[:1]].toarray()
        X2 = X[:, columns_idx2[:1]].toarray()

        # Compute the difference metric (sum of element-wise differences)
        diff_metric = np.abs(X1 - X2).sum(axis=1)

        # Select rows where the difference metric is greater than the threshold
        selected_row_indices = np.where(diff_metric > threshold)[0]
        print(f" Filtering {len(selected_row_indices):_}/{X.shape[0]}")
        
        X = X[selected_row_indices]


    # 2. Values used to label with seeds (selected subreddits)
    col1 = X[:, (columns_idx1[:k])].A.sum(axis=1)
    col2 = X[:, (columns_idx2[:k])].A.sum(axis=1)

    # filter ambiguous rows
    if subselect:
        selected_row_indices = np.where(col1 != col2)[0]
        _X = X[selected_row_indices]
    else:
        selected_row_indices = np.arange(X.shape[0])
        _X = X

    # 3. Create y
    col1 = _X[:, (columns_idx1[:k])].A.sum(axis=1)
    col2 = _X[:, (columns_idx2[:k])].A.sum(axis=1)
    y_seed_labeled = (col2 > col1).astype(int)

    return y_seed_labeled, _X



    
################################################################################################
####################################        General              ###############################
################################################################################################  


def remove_multilabeled_rows(__X, __y):
    print("    Removing multilabeled rows .. ")
    from collections import defaultdict
    import hashlib
    def hash_row(row):
        """ Hash a sparse matrix row to a unique identifier. """
        return hashlib.md5(row.todense().tobytes()).hexdigest()

    # Initialize a dictionary to track unique rows and their corresponding y_train values
    row_dict = defaultdict(set)

    # Iterate over each row in the sparse matrix
    for i in tqdm(range(__X.shape[0]), desc="Hashing rows"):
        row = __X.getrow(i)
        row_hash = hash_row(row)
        row_dict[row_hash].add(__y[i])

    # Identify rows with unique y_train values
    unique_rows = []
    for row_hash, labels in tqdm(row_dict.items(), desc="Identifying rows with unique y_train values"):
         if len(labels) == 1:
            unique_rows.append(row_hash)

    # Filter the original matrix and y_train based on unique rows
    indices_to_keep = []
    for i in tqdm(range(__X.shape[0]), desc='Filtering indices to keep'):
        if hash_row(__X.getrow(i)) in unique_rows:
            indices_to_keep.append(i)

    filtered_X_train = __X[indices_to_keep]
    filtered_y_train = __y[indices_to_keep]

    # Display the filtered results
    
    print("     filtered X_train (shape):", filtered_X_train.shape, '   vs ', __X.shape)
    print("     filtered y_train (length):", len(filtered_y_train), '   vs ', __y.shape)
    return filtered_X_train, filtered_y_train

def printlog(text, filename="./logging.txt"):
    with open(filename, "a") as file:
        file.write(text)

def count_gzip_lines(myfile):
    import gzip
    n_lines = 0
    with gzip.open(myfile, 'rb') as f:
        for _ in f:
            n_lines += 1
    return n_lines
   

################################################################################################
####################################        Threads              ###############################
################################################################################################
def set_mkl_threads(n):
    """Sets all the env variables to limit numb of threads
    """
    try:
        import mkl
        mkl.set_num_threads(n)
        return 0
    except:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:  # pylint: disable=bare-except
            pass
    v = f"{n}"
    os.environ["OMP_NUM_THREADS"] = v  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = v  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = v  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = v  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = v  # export NUMEXPR_NUM_THREADS=6