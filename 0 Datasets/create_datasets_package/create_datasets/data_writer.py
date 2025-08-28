import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

class DataWriter:
    """
    A class to write data to Parquet files.
    """
    def write_cleaned_orders_to_parquet(self, pizza_orders_src, pizza_orders_top, drink_orders_src, drink_orders_top, parquet_file_path_orders, batch_size):
        """
        Writes raw order data into a Parquet file in batches, preserving the original list format with square brackets.
        
        Args:
            pizza_orders_src (list or pandas.Series): Source pizza orders.
            pizza_orders_top (list or pandas.Series): Top pizza orders.
            drink_orders_src (list or pandas.Series): Source drink orders.
            drink_orders_top (list or pandas.Series): Top drink orders.
            parquet_file_path_orders (str): Path to the output Parquet file.
            batch_size (int): Number of rows to write in each batch.
        """

        # Convert inputs to lists if they are pandas.Series
        pizza_orders_src = list(pizza_orders_src)
        pizza_orders_top = list(pizza_orders_top)
        drink_orders_src = list(drink_orders_src)
        drink_orders_top = list(drink_orders_top)

        # Ensure all lists have the same length
        assert len(pizza_orders_src) == len(pizza_orders_top) == len(drink_orders_src) == len(drink_orders_top), \
            "All input lists must have the same length."

        # Combine the lists into rows, preserving the original list format
        rows = [
            (
                str(pizza_orders_src[i]),
                str(pizza_orders_top[i]),
                str(drink_orders_src[i]),
                str(drink_orders_top[i])
            )
            for i in range(len(pizza_orders_src))
        ]
        
        # Define column names
        columns = ['cleaned_pizza_orders_src', 'cleaned_pizza_orders_top', 'cleaned_drink_orders_src', 'cleaned_drink_orders_top']
        
        # Create a schema for the data
        schema = pa.schema([
            ('cleaned_pizza_orders_src', pa.string()),
            ('cleaned_pizza_orders_top', pa.string()),
            ('cleaned_drink_orders_src', pa.string()),
            ('cleaned_drink_orders_top', pa.string())
        ])
        
        # Initialize the ParquetWriter with the schema
        with pq.ParquetWriter(parquet_file_path_orders, schema=schema) as writer:
            # Write the data to Parquet in batches
            for start in range(0, len(rows), batch_size):
                
                # Prepare the batch data
                batch_rows = rows[start:start + batch_size]
                batch_df = pd.DataFrame(batch_rows, columns=columns)

                # Convert the batch to an Arrow table and write to the Parquet file
                batch_table = pa.Table.from_pandas(batch_df, schema=schema)
                writer.write_table(batch_table)

        # Print the number of rows in the Parquet file
        pq_file_orders = pq.ParquetFile(parquet_file_path_orders)
        print(f"Number of rows in the Parquet file: {pq_file_orders.metadata.num_rows}")

    def write_splitter_dataset_to_parquet(self, train_SRC, cleaned_pizza_orders_src, cleaned_drink_orders_src, parquet_file_path_selected, batch_size):
        """
        Writes selected columns (train_SRC, cleaned_pizza_orders_src, cleaned_drink_orders_src) to a Parquet file in batches.

        Args:
            train_SRC (list or pandas.Series): Training source data.
            cleaned_pizza_orders_src (list or pandas.Series): Cleaned pizza source orders.
            cleaned_drink_orders_src (list or pandas.Series): Cleaned drink source orders.
            parquet_file_path_selected (str): Path to the output Parquet file.
            batch_size (int): Number of rows to write in each batch.
        """

        # Convert inputs to lists if they are pandas.Series
        train_SRC = list(train_SRC)
        cleaned_pizza_orders_src = list(cleaned_pizza_orders_src)
        cleaned_drink_orders_src = list(cleaned_drink_orders_src)

        # Ensure all lists have the same length
        assert len(train_SRC) == len(cleaned_pizza_orders_src) == len(cleaned_drink_orders_src), \
            "All input lists must have the same length."

        # Combine the lists into rows
        rows = [
            (
                str(' '.join(train_SRC[i])),
                str(cleaned_pizza_orders_src[i]),
                str(cleaned_drink_orders_src[i])
            )
            for i in range(len(train_SRC))
        ]

        # Define column names
        columns = ['train_SRC', 'cleaned_pizza_orders_src', 'cleaned_drink_orders_src']

        # Create a schema for the data
        schema = pa.schema([
            ('train_SRC', pa.string()),
            ('cleaned_pizza_orders_src', pa.string()),
            ('cleaned_drink_orders_src', pa.string())
        ])

        # Initialize the ParquetWriter with the schema
        with pq.ParquetWriter(parquet_file_path_selected, schema=schema) as writer:
            # Write the data to Parquet in batches
            for start in range(0, len(rows), batch_size):
                # Prepare the batch data
                batch_rows = rows[start:start + batch_size]
                batch_df = pd.DataFrame(batch_rows, columns=columns)

                # Convert the batch to an Arrow table and write to the Parquet file
                batch_table = pa.Table.from_pandas(batch_df, schema=schema)
                writer.write_table(batch_table)

        # Print the number of rows in the Parquet file
        pq_file_selected = pq.ParquetFile(parquet_file_path_selected)
        print(f"Number of rows in the Parquet file: {pq_file_selected.metadata.num_rows}")
    
    def write_dataset_to_parquet(self, values, labels, parquet_file_path, batch_size):
        """
        Writes selected columns (values, labels) to a Parquet file in batches.

        Args:
            values (list or pandas.Series): Values.
            labels (list or pandas.Series): Labels.
            parquet_file_path (str): Path to the output Parquet file.
            batch_size (int): Number of rows to write in each batch.
        """

        # Convert inputs to lists if they are pandas.Series
        values = list(values)
        labels = list(labels)

        # Ensure both lists have the same length
        assert len(values) == len(labels), "Values and labels lists must have the same length."

        # Combine the lists into rows
        rows = [
            (str(values[i]), str(labels[i]))
            for i in range(len(values))
        ]

        # Define column names
        columns = ['values', 'labels']

        # Create a schema for the data
        schema = pa.schema([
            ('values', pa.string()),
            ('labels', pa.string())
        ])

        # Initialize the ParquetWriter with the schema
        with pq.ParquetWriter(parquet_file_path, schema=schema) as writer:
            # Write the data to Parquet in batches
            for start in range(0, len(rows), batch_size):
                # Prepare the batch data
                batch_rows = rows[start:start + batch_size]
                batch_df = pd.DataFrame(batch_rows, columns=columns)

                # Convert the batch to an Arrow table and write to the Parquet file
                batch_table = pa.Table.from_pandas(batch_df, schema=schema)
                writer.write_table(batch_table)

        # Print the number of rows in the Parquet file
        pq_file = pq.ParquetFile(parquet_file_path)
        print(f"Number of rows in the Parquet file: {pq_file.metadata.num_rows}")

    def write_dataframe_to_text_file(self, df, file_path):
        """
        Write a DataFrame to a text file using the tabulate library, left-aligned.

        Args:
            df (pandas.DataFrame): The DataFrame to write.
            file_path (str): The file path to write the DataFrame to.
        """
        from tabulate import tabulate

        # Format the DataFrame into a left-aligned table
        table = tabulate(
            df,
            headers="keys",               # Include DataFrame column names as headers
            tablefmt="grid",              # Grid table format
            showindex=True,              # Exclude row indices
            stralign="left",              # Align all text to the left
        )

        # Write the formatted table to a file
        with open(file_path, 'w') as f:
            f.write(table)

        print(f"DataFrame written to text file: {file_path}")

    def write_clustered_data_to_parquet(self, clusters, parquet_file_path, batch_size=1000):
        """
        Writes clustered data to a Parquet file in batches.
    
        Args:
            clusters (dict): A dictionary where keys are cluster labels and values are lists of tuples (sentence, labels) in each cluster.
            parquet_file_path (str): Path to the output Parquet file.
            batch_size (int): Number of rows to write in each batch.
        """
        # Create a schema for the data
        schema = pa.schema([
            ('cluster_label', pa.int32()),
            ('sentence', pa.list_(pa.string())),
            ('labels', pa.list_(pa.string()))
        ])
    
        # Initialize the ParquetWriter with the schema
        with pq.ParquetWriter(parquet_file_path, schema=schema) as writer:
            # Write the data in batches
            for cluster_label, items in clusters.items():
                # Prepare the batch data
                for start in range(0, len(items), batch_size):
                    batch_items = items[start:start + batch_size]
                    batch = [{'cluster_label': cluster_label, 'sentence': sentence, 'labels': labels} for sentence, labels in batch_items]
                    df = pd.DataFrame(batch)
                    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                    writer.write_table(table)
    
        # Print the number of rows in the Parquet file
        pq_file = pq.ParquetFile(parquet_file_path)
        print(f"Number of rows in the Parquet file: {pq_file.metadata.num_rows}")

    def write_clustered_data_to_text_file(self, clusters, file_path):
        """
        Write clustered data to a text file.
    
        Args:
            clusters (dict): A dictionary where keys are cluster labels and values are lists of tuples (sentence, labels) in each cluster.
            file_path (str): The file path to write the clustered data to.
        """
        with open(file_path, 'w') as f:
            for cluster_label, items in clusters.items():
                f.write(f"Cluster {cluster_label}:\n")
                for sentence, labels in items:
                    f.write(f"Sentence: {sentence}\n")
                    f.write(f"Labels: {labels}\n")
                    f.write("\n")
    
        print(f"Clustered data written to text file: {file_path}")

    def write_grouped_labels_to_parquet(self, grouped_labels_list, parquet_file_path, batch_size=1000):
        """
        Writes grouped labels to a Parquet file in batches.
    
        Args:
            grouped_labels_list (list): List of dictionaries containing grouped labels.
            parquet_file_path (str): Path to the output Parquet file.
            batch_size (int): Number of rows to write in each batch.
        """
        # Create a schema for the data
        schema = pa.schema([
            ('sentence', pa.list_(pa.string())),
            ('ner', pa.list_(pa.struct([
                ('index', pa.list_(pa.int32())),
                ('type', pa.string())
            ])))
        ])
    
        # Initialize the ParquetWriter with the schema
        with pq.ParquetWriter(parquet_file_path, schema=schema) as writer:
            # Write the data in batches
            for start in range(0, len(grouped_labels_list), batch_size):
                batch = grouped_labels_list[start:start + batch_size]
                df = pd.DataFrame(batch)
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                writer.write_table(table)
    
        # Print the number of rows in the Parquet file
        pq_file = pq.ParquetFile(parquet_file_path)
        print(f"Number of rows in the Parquet file: {pq_file.metadata.num_rows}")

    def write_grouped_labels_to_text_file(self, grouped_labels_list, file_path):
        """
        Write grouped labels to a text file.
    
        Args:
            grouped_labels_list (list): List of dictionaries containing grouped labels.
            file_path (str): The file path to write the grouped labels to.
        """
        with open(file_path, 'w') as f:
            for item in grouped_labels_list:
                f.write(f"Sentence: {item['sentence']}\n")
                f.write("NER:\n")
                for ner_item in item['ner']:
                    f.write(f"    Index: {ner_item['index']}\n")
                    f.write(f"    Type: {ner_item['type']}\n")
                f.write("\n")
    
        print(f"Grouped labels written to text file: {file_path}")