import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt

import gc
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

class DataClusterer:
    """
    A class to cluster suborders to avoid overfitting.
    """

    def cluster_sentences_large(
        self,
        sentences, 
        labels, 
        model_name='all-MiniLM-L6-v2', 
        method='hdbscan', 
        eps=0.3, 
        min_samples=2, 
        embedding_batch_size=512,  # Adjust this value based on your hardware capabilities
        clustering_batch_size=10000, 
        pca_components=25, #! Change 25 - 50 for Drink Dev and both tests
        visualize=False
    ):
        """
        Cluster sentences based on semantic similarity using DBSCAN or HDBSCAN in a memory-efficient way.
        
        Args:
            sentences (list of list of str): List of tokenized sentences to cluster.
            labels (list): List of corresponding labels for the sentences.
            model_name (str): Name of the Sentence-BERT model to use for embedding.
            method (str): Clustering method ('dbscan' or 'hdbscan').
            eps (float): The maximum distance between points for DBSCAN.
            min_samples (int): The minimum number of points required to form a cluster.
            embedding_batch_size (int): Batch size for generating embeddings.
            clustering_batch_size (int): Batch size for clustering.
            pca_components (int): Number of PCA components for dimensionality reduction.
            visualize (bool): If True, visualizes the cosine distance matrix.
        
        Returns:
            dict: A dictionary where keys are cluster labels and values are lists of sentences in each cluster.
        """
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Flatten tokenized sentences into strings
        sentence_texts = [" ".join(sentence) for sentence in sentences]

        # Step 2: Load the pre-trained Sentence-BERT model
        model = SentenceTransformer(model_name)
        model = model.to(device)

        # Step 3: Generate embeddings in smaller batches
        print("Generating embeddings...")
        sentence_embeddings = []
        num_batches = (len(sentence_texts) + embedding_batch_size - 1) // embedding_batch_size
        for i in tqdm(range(num_batches), desc="Embedding batches", unit="batch"):
            batch = sentence_texts[i * embedding_batch_size:(i + 1) * embedding_batch_size]
            
            # Generate embeddings for the current batch with a larger batch size
            batch_embeddings = model.encode(batch, batch_size=embedding_batch_size, show_progress_bar=False, device=device)
            sentence_embeddings.append(batch_embeddings)
            
            # Free memory for this batch
            del batch
            del batch_embeddings
            gc.collect()
        sentence_embeddings = np.vstack(sentence_embeddings)
        
        # Step 4: Apply PCA for dimensionality reduction
        if pca_components:
            print(f"Reducing dimensions to {pca_components} using PCA...")
            pca = PCA(n_components=pca_components)
            sentence_embeddings = pca.fit_transform(sentence_embeddings)
            del pca
            gc.collect()

        # Optional: Visualize cosine distance matrix
        if visualize:
            print("Visualizing cosine distance matrix...")
            distance_matrix = cosine_distances(sentence_embeddings)
            sns.heatmap(distance_matrix, annot=False, cmap="coolwarm")
            plt.title("Cosine Distance Matrix")
            plt.show()
            del distance_matrix
            gc.collect()

        # Step 5: Perform clustering in chunks
        print("Clustering sentences...")
        clusters = {}
        clusterer = None

        # Initialize clusterer based on the chosen method
        if method.lower() == 'dbscan':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        elif method.lower() == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(min_samples=min_samples, metric='euclidean', cluster_selection_epsilon=eps)
        else:
            raise ValueError("Unsupported clustering method. Choose 'dbscan' or 'hdbscan'.")

        num_clustering_batches = (len(sentence_embeddings) + clustering_batch_size - 1) // clustering_batch_size
        for i in tqdm(range(num_clustering_batches), desc="Clustering batches", unit="batch"):
            batch_embeddings = sentence_embeddings[i * clustering_batch_size:(i + 1) * clustering_batch_size]
            
            # Perform clustering on the batch
            batch_labels = clusterer.fit_predict(batch_embeddings)
            
            # Organize sentences and labels into clusters
            for label, sentence, label_list in zip(batch_labels, sentences[i * clustering_batch_size:(i + 1) * clustering_batch_size], labels[i * clustering_batch_size:(i + 1) * clustering_batch_size]):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((sentence, label_list))
            
            # Free memory for this clustering batch
            del batch_embeddings
            del batch_labels
            gc.collect()

        # Clear large variables no longer needed
        del sentence_embeddings
        del sentence_texts
        del clusterer
        gc.collect()

        print("Clustering completed.")
        return clusters

    def filter_clusters(self, clusters, filter_ratio=0.05):
        """
        Filter clusters to keep all elements of the noise cluster (-1) and a fraction of other clusters.
        
        Args:
            clusters (dict): A dictionary of clusters.
            filter_ratio (float): The fraction of elements to keep from non-noise clusters.
            
        Returns:
            dict: A filtered dictionary of clusters.
        """
        filtered_clusters = {}
        for label, sentences in clusters.items():
            if label == -1:
                filtered_clusters[label] = sentences
            else:
                num_sentences = len(sentences)
                num_to_keep = int(num_sentences * filter_ratio)
                filtered_clusters[label] = sentences[:num_to_keep]
        return filtered_clusters