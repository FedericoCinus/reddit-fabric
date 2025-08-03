import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import norm

class Nature_model():
    """Socio-demographic attribute inference class from the paper:
      Waller, Isaac, and Ashton Anderson.
      Quantifying social organization and political polarization in online platforms. 
      Nature 600.7888 (2021): 264-268.
    """
    def __init__(self, score_file, subreddit_names, attribute):
        # 3. Load Nature model
        assert attribute != "compass", "<compass> attribute contains multiple political ideologies which are not studied in this model"
        self.invert_label = False
        if attribute == "demo_rep":
            self.attribute =  "partisan"
        elif attribute == "year":
            self.attribute =  "age"
            self.invert_label = True # age and birth year have opposite label
        else:
            self.attribute =  attribute
        self.scores = pd.read_csv(score_file, usecols=["community", self.attribute])
        self.subreddit_names = subreddit_names
        self.subreddits2index = {sub: i for i, sub in enumerate(self.subreddit_names)}
      
    def fit(self, _, y):
        """Retrieves 
        :scores_attribute --> score weights for the given attribute
        :subreddits_in_nature --> subreddits titles that are also in Nature paper
        :subreddits_in_nature_indices --> subreddits indices corresponding to the input list (subreddit_names)
        """        
        # filter and sort subreddits in scores df
        self.classes_ = np.unique(y)
        self.scores = (
            self.scores.assign(subreddit_index=self.scores['community'].map(self.subreddits2index))
                .loc[self.scores['community'].isin(self.subreddit_names)]
                .sort_values(by='subreddit_index')
                .astype({'subreddit_index': 'int'})
        )
        self.scores_attribute = self.scores[self.attribute].values.reshape((len(self.scores) , 1))
        self.subreddits_in_nature = self.scores['community'].values
        self.subreddits_indices_in_nature = self.scores['subreddit_index'].values
   
    def predict_proba(self, X_test):
        """Returns classification probabilities 
        """
        num = X_test[:, self.subreddits_indices_in_nature] @ self.scores_attribute
        den = X_test[:, self.subreddits_indices_in_nature].sum(axis=1)
        
        y = np.divide(num, den, out=np.zeros_like(num), where=(den!=0))
        
        # from z-score to proba
        if self.invert_label:
            y_proba = np.hstack((norm.cdf(y), 1- norm.cdf(y)))
        else:
            y_proba = np.hstack((1 - norm.cdf(y), norm.cdf(y)))

        return y_proba
    
    def predict(self, X_test):
        """Returns classification 
        """
        probas = self.predict_proba(X_test)
        
        return np.argmax(probas, axis=1)