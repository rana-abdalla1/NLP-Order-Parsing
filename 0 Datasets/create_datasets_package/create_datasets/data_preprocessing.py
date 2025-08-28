import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import tabulate
import os
import pandas as pd

class DataPreprocessor:
    """
    A class to preprocess the data  by extracting and cleaning the orders from the SRC and TOP columns.
    """
    def __init__(self):
        self.labels = ["ORDER", "PIZZAORDER", "NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKORDER", "DRINKTYPE", "VOLUME", "NOT", "CONTAINERTYPE"]
    
    def split_with_brackets(self, text):
        """
        Split a string while keeping brackets as separate elements in the output list.
        """
        output_list = re.findall(r'\(|\)|\S+', text)
        return output_list

    def get_orders_indices(self, SRC_list, TOP_list):
        """
        Extract the start and end indices of each PIZZAORDER and DRINKORDER in the SRC and TOP lists.
        """
        element_index = 0
        word_index = 0

        pizza_order_indices_src = []
        drink_order_indices_src = []
        pizza_order_indices_top = []
        drink_order_indices_top = []

        last_order_type = None
        last_order_start_src = None
        last_order_start_top = None

        while element_index < len(TOP_list):
            element = TOP_list[element_index]

            if (element in self.labels and element not in ["PIZZAORDER", "DRINKORDER"]) or (element in ["(", ")"]):
                element_index += 1

            if element not in self.labels and element not in ["(", ")"]:
                element_index += 1
                word_index += 1

            if element in ["PIZZAORDER", "DRINKORDER"]:
                if last_order_type is not None:
                    if last_order_type == "PIZZAORDER":
                        pizza_order_indices_src.append((last_order_start_src, word_index - 1))
                        pizza_order_indices_top.append((last_order_start_top, element_index - 2))
                    if last_order_type == "DRINKORDER":
                        drink_order_indices_src.append((last_order_start_src, word_index - 1))
                        drink_order_indices_top.append((last_order_start_top, element_index - 2))

                last_order_type = element
                last_order_start_src = word_index
                last_order_start_top = element_index - 1
                element_index += 1

        if last_order_type is not None:
            last_index_SRC = len(SRC_list) - 1
            last_index_TOP = len(TOP_list) - 2

            if last_order_type == "PIZZAORDER":
                pizza_order_indices_src.append((last_order_start_src, last_index_SRC))
                pizza_order_indices_top.append((last_order_start_top, last_index_TOP))
            else:
                drink_order_indices_src.append((last_order_start_src, last_index_SRC))
                drink_order_indices_top.append((last_order_start_top, last_index_TOP))

        return pizza_order_indices_src, pizza_order_indices_top, drink_order_indices_src, drink_order_indices_top
    
    def extract_orders(self, pizza_order_indices_src, pizza_order_indices_top, drink_order_indices_src, drink_order_indices_top, SRC_list, TOP_list):
        """
        Extract the pizza and drink orders from the SRC and TOP lists based on the order indices.
        """
        pizza_orders_src = [" ".join(SRC_list[start:end + 1]) for start, end in pizza_order_indices_src]
        pizza_orders_top = [" ".join(TOP_list[start:end + 1]) for start, end in pizza_order_indices_top]
        drink_orders_src = [" ".join(SRC_list[start:end + 1]) for start, end in drink_order_indices_src]
        drink_orders_top = [" ".join(TOP_list[start:end + 1]) for start, end in drink_order_indices_top]

        return pizza_orders_src, pizza_orders_top, drink_orders_src, drink_orders_top

    def clean_and_align_all_orders(self, pizza_orders_src, pizza_orders_top, drink_orders_src, drink_orders_top):
        """
        Aligns the pizza and drink orders lists by trimming excess words after the last closing bracket ')'.
        """
        def clean_orders(src_list, top_list):
            cleaned_src = []
            cleaned_top = []

            for idx, (src, top) in enumerate(zip(src_list, top_list)):
                if not top:
                    cleaned_src.append(src)
                    cleaned_top.append(top)
                    continue

                if isinstance(top, list):
                    top = " ".join(top)

                last_closing_bracket_idx = top.rfind(')')
                cleaned_top_order = top[:last_closing_bracket_idx + 1]
                removed_words_count = len(top[last_closing_bracket_idx + 1:].split())

                if isinstance(src, list):
                    src = " ".join(src)

                src_words = src.split()
                cleaned_src_order = ' '.join(src_words[:-removed_words_count] if removed_words_count > 0 else src_words)

                cleaned_src.append(cleaned_src_order)
                cleaned_top.append(cleaned_top_order)

            cleaned_src = [order for order in cleaned_src if order]
            cleaned_top = [order for order in cleaned_top if order]

            return cleaned_src, cleaned_top

        pizza_orders_src_cleaned, pizza_orders_top_cleaned = clean_orders(pizza_orders_src, pizza_orders_top)
        drink_orders_src_cleaned, drink_orders_top_cleaned = clean_orders(drink_orders_src, drink_orders_top)

        return pizza_orders_src_cleaned, pizza_orders_top_cleaned, drink_orders_src_cleaned, drink_orders_top_cleaned

    def preprocess_orders(self, df, dataset_type):
        """
        Preprocess the orders in the DataFrame.
        """
        df_split_orders = df.copy()
        df_split_orders['SRC_list'] = df_split_orders[f"{dataset_type}.SRC"].apply(lambda x: x.split())
        df_split_orders['TOP_list'] = df_split_orders[f"{dataset_type}.TOP"].apply(self.split_with_brackets)

        df_split_orders['pizza_order_indices_src'], df_split_orders['pizza_order_indices_top'], df_split_orders['drink_order_indices_src'], df_split_orders['drink_order_indices_top'] = zip(
            *df_split_orders.apply(
                lambda row: self.get_orders_indices(row['SRC_list'], row['TOP_list']), axis=1
            )
        )

        df_split_orders['pizza_orders_src'], df_split_orders['pizza_orders_top'], df_split_orders['drink_orders_src'], df_split_orders['drink_orders_top'] = zip(
            *df_split_orders.apply(
                lambda row: self.extract_orders(
                    row['pizza_order_indices_src'],
                    row['pizza_order_indices_top'],
                    row['drink_order_indices_src'],
                    row['drink_order_indices_top'],
                    row['SRC_list'],
                    row['TOP_list']
                ), axis=1
            )
        )

        df_split_orders['cleaned_pizza_orders_src'], df_split_orders['cleaned_pizza_orders_top'], df_split_orders['cleaned_drink_orders_src'], df_split_orders['cleaned_drink_orders_top'] = zip(
            *df_split_orders.apply(
                lambda row: self.clean_and_align_all_orders(
                    row['pizza_orders_src'],
                    row['pizza_orders_top'],
                    row['drink_orders_src'],
                    row['drink_orders_top']
                ), axis=1
            )
        )

        return df_split_orders
    
    def write_orders_to_file(self, df, output_file_path, dataset_type):
        """
        Write the preprocessed data in a tabular format to a text file.
        Write the preprocessed data in a tabular format to a text file.
        """
        n = df.shape[0]
        
        with open(output_file_path, 'w') as f:
            # Original Table
            headers_original = ["SRC String", "TOP String"]
            rows_original = df[[f"{dataset_type}.SRC", f"{dataset_type}.TOP"]].head(n).values.tolist()
            f.write("Original Data:\n")
            f.write(tabulate.tabulate(rows_original, headers=headers_original, tablefmt="grid"))
            f.write("\n\n")

            # Indices Table  
            headers_indices = ["Index Pizza Orders SRC", "Index Pizza Orders TOP", "Index Drink Orders SRC", "Index Drink Orders TOP"]
            rows_indices = df[['pizza_order_indices_src', 'pizza_order_indices_top', 'drink_order_indices_src', 'drink_order_indices_top']].head(n).values.tolist()
            f.write("Indexed Data:\n")
            f.write(tabulate.tabulate(rows_indices, headers=headers_indices, tablefmt="grid"))
            f.write("\n\n")
            
            # Extracted Table
            headers_extracted = ["Pizza Orders SRC", "Pizza Orders TOP", "Drink Orders SRC", "Drink Orders TOP"]
            rows_extracted = df[['pizza_orders_src', 'pizza_orders_top', 'drink_orders_src', 'drink_orders_top']].head(n).values.tolist()
            f.write("Extracted Data:\n")
            f.write(tabulate.tabulate(rows_extracted, headers=headers_extracted, tablefmt="grid"))
            f.write("\n\n")
            
            # Cleaned Table
            headers_cleaned = ["Cleaned Pizza Orders SRC", "Cleaned Pizza Orders TOP", "Cleaned Drink Orders SRC", "Cleaned Drink Orders TOP"]
            rows_cleaned = df[['cleaned_pizza_orders_src', 'cleaned_pizza_orders_top', 'cleaned_drink_orders_src', 'cleaned_drink_orders_top']].head(n).values.tolist()
            f.write("Cleaned Data:\n")
            f.write(tabulate.tabulate(rows_cleaned, headers=headers_cleaned, tablefmt="grid"))
            f.write("\n\n")

    #! ------------------------------------------------------ New Functions ------------------------------------------------------ !#
   
    def remove_outer_label(self, order):
        """
        Remove the PIZZAORDER or DRINKORDER label and parentheses from the TOP string.

        Args:
        order (str): The TOP string containing the order.

        Returns:
        str: The TOP string with the outermost label removed.
        """
    
        order_list = self.split_with_brackets(order)

        # Remove the first two elements (the outer label and the opening parenthesis) and the last element (the closing parenthesis)
        order_list = order_list[2:-1]

        return " ".join(order_list)
    
    def remove_complex_toppings(self, order):
        """
        Remove the COMPLEX_TOPPING label and parentheses from the TOP string.

        Args:
        order (str): The TOP string containing the order.

        Returns:
        str: The TOP string with the COMPLEX_TOPPING label removed.
        """
    
       # Search for the COMPLEX_TOPPING label
       # Remove the label
       # Remove the opening parenthesis directly before the label
       # Use a stack to keep track of the parentheses and remove the closing parenthesis that corresponds to the opening parenthesis of the COMPLEX_TOPPING label

        order_list = self.split_with_brackets(order)

        stack = []
        for i, element in enumerate(order_list):
            if element == "(":
                stack.append(i)
            elif element == ")":
                if stack:
                    start = stack.pop()
                    if order_list[start - 1] == "COMPLEX_TOPPING":
                        order_list = order_list[:start - 2] + order_list[i + 1:] 
                        break

        return " ".join(order_list)
    
    def extract_values_and_labels_from_suborder(self, order):
        """
        Extract the values and labels from a TOP string using BIOES format.

        Args:
        order (str): The TOP string containing the order.

        Returns:
        tuple: Two lists containing the values and their corresponding BIOES labels.
        """

        # As a preprocessing step, replace all NOT labels with NOT-LABEL and remove the closing bracket of the NOT
        # Example: ( NOT ( TOPPING american cheese ) ) -> ( NOT-TOPPING american cheese )
        order = order.replace("( NOT (", "( NOT ").replace(" ) )", " )")

        # Extract all parentheses with their contents from the TOP string
        parentheses = re.findall(r'\([^()]*\)', order)

        values = []
        labels = []

        # Replace all matches with #
        modified_order = order
        for p in parentheses:
            modified_order = modified_order.replace(p, '#', 1)

        # Split the modified order into tokens
        tokens = modified_order.split()

        # Loop over the tokens
        match_index = 0
        for token in tokens:
            if token == '#':
                # Handle the first match in the list of found matches
                p = parentheses[match_index]
                match_index += 1

                # Split the parentheses string into a list of words and remove the parentheses
                p_list = p[1:-1].split()

                if p_list[0] == "NOT":
                    # The first two words are the label
                    label = p_list[0] + "-" + p_list[1]
                    # The rest of the words are the values
                    value_list = p_list[2:]
                else:
                    # The first word is the label
                    label = p_list[0]
                    # The rest of the words are the values
                    value_list = p_list[1:]

                # Assign BIOES tags
                if len(value_list) == 1:
                    # Single value
                    values.append(value_list[0])
                    labels.append(f"S-{label}")
                else:
                    for i, value in enumerate(value_list):
                        values.append(value)
                        if i == 0:
                            # Beginning of a multi-token sequence
                            labels.append(f"B-{label}")
                        elif i == len(value_list) - 1:
                            # End of a multi-token sequence
                            labels.append(f"E-{label}")
                        else:
                            # Inside of a multi-token sequence
                            labels.append(f"I-{label}")
            else:
                # Label the token as 'O'
                values.append(token)
                labels.append("O")

        return values, labels

    def extract_values_and_labels_from_suborder_list(self, df):
        """
        Extract the values and labels from the cleaned pizza and drink orders after removing the outer label.

        Args:
        df (pd.DataFrame): The DataFrame containing the cleaned pizza and drink orders as two columns where each cell contains a list of suborders.

        Returns:
        tuple: Four lists containing pizza values, pizza labels, drink values, and drink labels.
        """

        pizza_values = []
        pizza_labels = []
        drink_values = []
        drink_labels = []

        # Remove the outer label from the TOP strings and extract values and labels
        for index, row in df.iterrows():
            cleaned_pizza_orders = [self.remove_outer_label(order) for order in row['cleaned_pizza_orders_top']]
            cleaned_drink_orders = [self.remove_outer_label(order) for order in row['cleaned_drink_orders_top']]

            # Apply remove_complex_toppings
            cleaned_pizza_orders = [self.remove_complex_toppings(order) for order in cleaned_pizza_orders]
            cleaned_drink_orders = [self.remove_complex_toppings(order) for order in cleaned_drink_orders]

            for order in cleaned_pizza_orders:
                values, labels = self.extract_values_and_labels_from_suborder(order)
                pizza_values.append(values)
                pizza_labels.append(labels)

            for order in cleaned_drink_orders:
                values, labels = self.extract_values_and_labels_from_suborder(order)
                drink_values.append(values)
                drink_labels.append(labels)
                
        # Create a new DataFrame with the extracted values and labels
        df_annotate_orders_pizza = pd.DataFrame({
            'pizza_values': pizza_values,
            'pizza_labels': pizza_labels
        })

        df_annotate_orders_drink = pd.DataFrame({
            'drink_values': drink_values,
            'drink_labels': drink_labels
        })

        return df_annotate_orders_pizza, df_annotate_orders_drink
    
#& ------------------------------------------------------ For the Paper ------------------------------------------------------ &#

    def convert_bioes_to_grouped_labels(self, bioes_tags):
        result = []
        current_group = []
        current_type = None

        # Iterate over the BIOES tags
        for i, tag in enumerate(bioes_tags):
            # Ensure the tag is a string
            tag = str(tag)
            
            # Extract the label type (e.g., NUMBER, TOPPING, NOT-STYLE)
            label_type = tag.split('-')[-1]

            # Check if we're starting a new group
            if tag.startswith('B-'):
                # If we have a previous group, add it to the result
                if current_group:
                    result.append({"index": current_group, "type": current_type})
                
                # Start a new group
                current_group = [i]
                current_type = label_type

            # If the label is 'I-' or 'E-', it's part of the current group
            elif tag.startswith(('I-', 'E-')):
                if label_type == current_type:
                    current_group.append(i)
                else:
                    # If label type changed, add the current group to the result
                    result.append({"index": current_group, "type": current_type})
                    current_group = [i]
                    current_type = label_type

            # If the label is 'S-', it's a single token, treat it as a new group
            elif tag.startswith('S-'):
                if current_group:
                    result.append({"index": current_group, "type": current_type})
                result.append({"index": [i], "type": label_type})
                current_group = []
                current_type = None

            # Handle 'O' (outside) - reset current group if necessary
            if tag == 'O' and current_group:
                result.append({"index": current_group, "type": current_type})
                current_group = []
                current_type = None

        # Add the final group if any
        if current_group:
            result.append({"index": current_group, "type": current_type})

        return result

    def convert_clusters_to_grouped_labels(self, clusters):
        """
        Convert the filtered clusters containing the extracted values and labels to a list of dictionaries with the required format.

        Args:
        clusters (dict): The filtered clusters containing the extracted values and labels.

        Returns:
        list: A list of dictionaries with the required format.
        """

        result = []

        for cluster_label, items in clusters.items():
            for sentence, labels in items:
                ner = self.convert_bioes_to_grouped_labels(labels)
                result.append({"sentence": sentence, "ner": ner})

        return result