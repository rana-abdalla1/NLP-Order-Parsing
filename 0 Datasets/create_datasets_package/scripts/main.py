import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_datasets.data_loader import DataLoader
from create_datasets.data_sampler import DataSampler
from create_datasets.data_statistics import DataStatistics
from create_datasets.data_preprocessing import DataPreprocessor
from create_datasets.data_writer import DataWriter
from create_datasets.data_clusterer import DataClusterer

import gc

def process_and_split_dataset(df, dataset_type, data_preprocessor, data_writer, base_path, batch_size):
    """
    This function preprocesses the data and writes the splitter dataset to a Parquet file.
    This function is called three times: for the train, dev, and test datasets.
    """

    # To handle that the dev data is split into dev and test
    print(f"Preprocessing data for {dataset_type} dataset...")

    if dataset_type == "test":
        temp_dataset_type = "dev"
    else:
        temp_dataset_type = dataset_type
    
    # Preprocess data
    df_preprocessed = data_preprocessor.preprocess_orders(df, temp_dataset_type)

    print(f"Data preprocessed for {dataset_type} successfully")

    # Write splitter dataset
    print(f"Writing splitter dataset for {dataset_type}...")

    file_path_splitter = os.path.join(base_path, f"{dataset_type}_splitter.parquet")
    data_writer.write_splitter_dataset_to_parquet(
        df_preprocessed['SRC_list'],
        df_preprocessed['cleaned_pizza_orders_src'],
        df_preprocessed['cleaned_drink_orders_src'],
        file_path_splitter,
        batch_size
    )

    print(f"Data written to {file_path_splitter} successfully")

    return df_preprocessed


def annotate_and_cluster_and_filter(df_preprocessed, dataset_type, order_type, data_preprocessor, data_writer, data_clusterer, base_path, batch_size):
    """
    This function annotates the data, clusters the sentences, filters the clusters, and writes the W2NER dataset to a Parquet file.
    This function is called six times: for the train, dev, and test datasets for both pizza and drink orders.
    """
    
    # Annotator Preprocessing
    print(f"Annotating data for {order_type} {dataset_type} dataset...")

    df_annotate_orders_pizza, df_annotate_orders_drink = data_preprocessor.extract_values_and_labels_from_suborder_list(df_preprocessed)

    df_annotate_orders = df_annotate_orders_pizza if order_type == "pizza" else df_annotate_orders_drink
    
    del df_annotate_orders_pizza, df_annotate_orders_drink  # Free memory
    gc.collect()

    print(f"Data annotated for {order_type} {dataset_type} successfully")

    # Clustering
    print(f"Clustering data for {order_type} {dataset_type} dataset...")

    clusters = data_clusterer.cluster_sentences_large(
        df_annotate_orders[f"{order_type}_values"], 
        df_annotate_orders[f"{order_type}_labels"]
    )

    print(f"Data clustered for {order_type} {dataset_type} successfully")

    del df_annotate_orders  # Free memory
    gc.collect()

    # Filtering clusters
    print(f"Filtering clusters for {order_type} {dataset_type} dataset...")

    filtered_clusters = data_clusterer.filter_clusters(clusters)

    print(f"Clusters filtered for {order_type} {dataset_type} successfully")

    del clusters  # Free memory
    gc.collect()

    # Convert to W2NER format
    print(f"Converting data to W2NER format for {order_type} {dataset_type} dataset...")

    list_w2ner = data_preprocessor.convert_clusters_to_grouped_labels(filtered_clusters)

    print(f"Data converted to W2NER format for {order_type} {dataset_type} successfully")

    del filtered_clusters  # Free memory
    gc.collect()

    # Write W2NER dataset
    print(f"Writing W2NER dataset for {order_type} {dataset_type}...")

    file_path_w2ner = os.path.join(base_path, f"{dataset_type}.parquet")
    data_writer.write_grouped_labels_to_parquet(list_w2ner, file_path_w2ner, batch_size)

    print(f"Data written to {file_path_w2ner} successfully")
    
    file_path_w2ner_txt = os.path.join(base_path, f"{dataset_type}.txt")
    data_writer.write_grouped_labels_to_text_file(list_w2ner, file_path_w2ner_txt)

    print(f"Data written to {file_path_w2ner_txt} successfully")
    
    del list_w2ner  # Free memory
    gc.collect()


def main():
    batch_size = 10000
    file_path_train_dataset = "/kaggle/input/pizza-train/PIZZA_train.parquet"
    file_path_dev_dataset = "/kaggle/input/pizza-dev/PIZZA_dev.parquet"

    base_path_splitter = "/kaggle/working/NLP-Project/1 Models/suborder_splitter/data"

    base_path_pizza_annotator = "/kaggle/working/NLP-Project/1 Models/suborder_annotator/data/project/pizza"
    base_path_drink_annotator = "/kaggle/working/NLP-Project/1 Models/suborder_annotator/data/project/drink"

    if not os.path.exists(base_path_pizza_annotator):
        os.makedirs(base_path_pizza_annotator)
    if not os.path.exists(base_path_drink_annotator):
        os.makedirs(base_path_drink_annotator)

    data_preprocessor = DataPreprocessor()
    data_writer = DataWriter()
    data_clusterer = DataClusterer()

    # Load data
    data_loader = DataLoader(file_path_train_dataset)
    df_train_loaded = data_loader.load_parquet_to_dataframe()
    print("Train data loaded successfully")

    data_loader_dev = DataLoader(file_path_dev_dataset)
    df_dev_loaded = data_loader_dev.load_parquet_to_dataframe()
    print("Dev data loaded successfully")

    df_dev, df_test = train_test_split(df_dev_loaded, test_size=0.5, random_state=42)
    del df_dev_loaded  # Free memory
    gc.collect()

    # Process and split datasets
    preprocessed_data = {}
    for dataset_type, df in zip(["train", "dev", "test"], [df_train_loaded, df_dev, df_test]):
        df_preprocessed = process_and_split_dataset(df, dataset_type, data_preprocessor, data_writer, base_path_splitter, batch_size)
        preprocessed_data[dataset_type] = df_preprocessed

    # Annotate, cluster, and filter datasets
    for dataset_type in ["train", "dev", "test"]:
        for order_type, base_path in zip(["pizza", "drink"], [base_path_pizza_annotator, base_path_drink_annotator]):
            df_preprocessed = preprocessed_data[dataset_type]
            annotate_and_cluster_and_filter(df_preprocessed, dataset_type, order_type, data_preprocessor, data_writer, data_clusterer, base_path, batch_size)
        del preprocessed_data[dataset_type]  # Free memory for the current dataset
        gc.collect()


if __name__ == "__main__":
    main()