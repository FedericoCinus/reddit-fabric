import numpy as np

class Majority():
    def __init__(self):
        pass
    def fit(self, _, y):
        self.majority_class = np.bincount(y).argmax()
    def predict(self, X):
        return np.repeat(self.majority_class, X.shape[0])
    def predict_proba(self, X):
        return np.vstack([np.repeat(1-self.majority_class, X.shape[0]),
                          np.repeat(self.majority_class, X.shape[0])]).T

class RawActivityModel():
    """Socio-demographic attribute inference using raw subreddit actvity 
    """
    def __init__(self, subreddit_names, attribute, numb_seed_subreddits=5):
        # 3. Load Nature model
        attribute = attribute.split("__")[0]
        self.attribute = attribute if attribute != "demo_rep" else "partisan"
        self.subreddit_names = subreddit_names
        self.k = numb_seed_subreddits
      
    def fit(self, _, y):
        """Loads subreddits
        """        
        # filter and sort subreddits in scores df
        self.classes_ = np.unique(y)
        
        if self.attribute == "gender":
            columns1 = ['AskMen', 'OneY', 'ROTC', 'predaddit']
            columns2 = ['AskWomen', 'women', 'USMilitarySO', 'BabyBumps']
        elif self.attribute == "partisan":
            columns1 = ['democrats', 'GamerGhazi', 'AskAnAmerican', 'lastweektonight']
            columns2 = ['Conservative', 'KotakuInAction', 'askaconservative', 'CGPGrey']
        elif self.attribute == "age" or self.attribute == "year":
            columns1 = ['teenagers', 'AskMen', 'trackandfield', 'RedHotChiliPeppers']
            columns2 = ['RedditForGrownups', 'AskMenOver30', 'trailrunning', 'pearljam']
        elif self.attribute == "year":
            columns1 = ['teenagers', 'AskMen', 'trackandfield', 'RedHotChiliPeppers']
            columns2 = ['RedditForGrownups', 'AskMenOver30', 'trailrunning', 'pearljam']


        return_index = lambda s: np.where(self.subreddit_names == s)[0][0]
        self.columns_idx1 = list(map(return_index, columns1)) # class 1
        self.columns_idx2 = list(map(return_index, columns2)) # class 2


   
    def predict_proba(self, X_test):
        """Returns classification probabilities 
        """
        col1 = X_test[:, (self.columns_idx1[:self.k])].A.sum(axis=1)
        col2 = X_test[:, (self.columns_idx2[:self.k])].A.sum(axis=1)
        y_seed_labeled = (col2 > col1).astype(int).reshape(-1, 1)
        
        y_proba = np.hstack((1 - y_seed_labeled, y_seed_labeled))
        
        return y_proba
    
    def predict(self, X_test):
        """Returns classification 
        """
        probas = self.predict_proba(X_test)
        
        return np.argmax(probas, axis=1)