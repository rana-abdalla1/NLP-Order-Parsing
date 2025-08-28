from abc import ABC, abstractmethod

# Abstract FeatureExtractor class
class FeatureExtractor(ABC):
    def __init__(self, corpus):
        self.corpus = corpus

    @abstractmethod
    def extract(self):
        """
        Abstract method for extracting features.
        Must be implemented by subclasses.
        """
        pass

    # Test Feature Extraction Function
    def test_feature_extraction(self):
        """
        Function to test feature extraction using the current FeatureExtractor object.
        """
        feature_names, representation = self.extract()
       
        return feature_names, representation