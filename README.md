# NLP Order Parsing Project

## Overview

This project aims to develop a function that receives an order and returns its parse tree. The task is divided into three subtasks:
1. Splitting the order into suborders.
2. Annotating each of these suborders.
3. Combining the results into one parse tree.

The first two subtasks require models to perform them, and each model needs a pipeline to fit in, and each pipeline needs a dataset to work on. The project involves creating two datasets appropriate for the two pipelines we will be needing.

## Datasets

### First Dataset
- **Input (X)**: An order.
- **Output (Y)**: A list of pizza suborders and a list of drink suborders.

### Second Dataset
- **Input (X)**: A suborder.
- **Output (Y)**: The labels of this suborder.

## Instructions

1. The raw and processed datasets are not included in the repository. The files are included in the `.gitignore` file.
2. Download these files from the [Google Drive folder](https://drive.google.com/drive/folders/1CJMD1o9aREbs7MajoXcyqF53MNi582rd?usp=drive_link).
3. Check the project structure below to see where the files should be placed.

## Project Structure

```
C:.
│   .gitignore
│   .gitmodules
│   README.md
│   requirements.txt
│
├───0 Datasets
│   │   transforming_json_to_parquet.ipynb
│   │   understand_provided_datasets.ipynb
│   │
│   ├───create_datasets_package
│   │   │   README.md
│   │   │   requirements.txt
│   │   │   __init__.py
│   │   │
│   │   ├───create_datasets
│   │   │   │   data_loader.py
│   │   │   │   data_preprocessing.py
│   │   │   │   data_sampler.py
│   │   │   │   data_statistics.py
│   │   │   │   data_writer.py
│   │   │   │   utils.py
│   │   │   │   __init__.py
│   │   │   │
│   │   │   └───__pycache__
│   │   │
│   │   └───scripts
│   │           main.py
│   │
│   ├───new_datasets
│   │       pizza_train_annotator_dataset_drink.parquet
│   │       pizza_train_annotator_dataset_drink.txt
│   │       pizza_train_annotator_dataset_drink_final.parquet
│   │       pizza_train_annotator_dataset_drink_final.txt
│   │       pizza_train_annotator_dataset_pizza.parquet
│   │       pizza_train_annotator_dataset_pizza.txt
│   │       pizza_train_annotator_dataset_pizza_final.parquet
│   │       pizza_train_annotator_dataset_pizza_final.txt
│   │       pizza_train_splitter_dataset.parquet
│   │       pizza_train_splitter_dataset.txt
│   │
│   ├───provided_datasets
│   │       PIZZA_dev.json
│   │       PIZZA_train.json
│   │       PIZZA_train.parquet
│   │
│   └───sample_input_and_output
│           input.txt
│           output.json
│
├───1 Models
│   ├───suborder_annotator
│   │   │   model.py
│   │   │   pipeline.py
│   │   │   train.py
│   │   │
│   │   └───Attempted Paper Implementation
│   │           1_encoder-layer.ipynb
│   │           2_convolution_layer.ipynb
│   │           3_co_predictor_layer.ipynb
│   │           4_decoding.ipynb
│   │           Expected_Input_Format_Notes.ipynb
│   │           install_packages.py
│   │
│   └───suborder_splitter
│           model.py
│           pipeline.py
│           train.py
│
├───2 Utils
│       data_utils.py
│       evaluation.py
│       model_utils.py
│
├───3 Feature Extraction
│   │   FeatureExtractor.py
│   │
│   ├───BagOfWords
│   │   │   BagOfWords.ipynb
│   │   │   BagOfWords.py
│   │   │   BagOfWords_Notes.ipynb
│   │   │
│   │   └───__pycache__
│   │           BagOfWords.cpython-311.pyc
│   │
│   └───TF-IDF
│       │   TFIDF.ipynb
│       │   TFIDF.py
│       │
│       └───__pycache__
│               TFIDF.cpython-311.pyc
│
├───4 Documents
│   ├───Requirements
│   │       NLP-Project-F24.pdf
│   │
│   └───Research Papers
│           Nested Named Entity Recognition.pdf
│           PIZZA A newbenchmark for complex end-to-end task-oriented parsing.pdf
│           Unified MRC Framework.pdf
│           Unified NER as Word-Word Relation Classification.pdf
│
└───submodules
    └───repository
        │   .gitignore
        │   config.py
        │   data_loader.py
        │   LICENSE
        │   main.py
        │   model.py
        │   README.md
        │   utils.py
        │
        ├───config
        │       ace05.json
        │       cadec.json
        │       conll03.json
        │       example.json
        │       genia.json
        │       resume-zh.json
        │
        ├───data
        │   └───example
        │           dev.json
        │           test.json
        │           train.json
        │
        ├───figures
        │       architecture.PNG
        │       scheme.PNG
        │
        └───log
                placeholder
```

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/FatemaKotb/NLP-Project.git
   cd project_root
   ```

## Usage

1. Create the datasets (incomplete):
   Check the readme file in the `create_datasets_package` folder.

## References
The papers that we are going to implement can be found in the following link: [Unified Named Entity Recognition as Word-Word Relation Classification](https://paperswithcode.com/paper/unified-named-entity-recognition-as-word-word) and its official repository can be found in this link: [w2ner](https://github.com/ljynlp/w2ner).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.