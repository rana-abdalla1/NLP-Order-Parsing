from sklearn.feature_extraction.text import CountVectorizer # type: ignore
import sys
import os

# Add the parent directory of 'FeatureExtraction' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import FeatureExtractor
from FeatureExtractor import FeatureExtractor

class BagOfWordsExtractor(FeatureExtractor):
    def extract(self):
        """
        Perform Bag of Words extraction.
        """
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.corpus)
        return vectorizer.get_feature_names_out(), X.toarray()
