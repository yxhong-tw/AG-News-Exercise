# AG-News-Exercise
This is the National Cheng Kung University (NCKU) Intelligent Knowledge Management (IKM) Lab's exercise for the new incoming students in the Fall 2023 semester.

## Before Start
### Requirements
You can install the requirements by running the following command:
```
pip install -r requirements.txt
```

### Check the Dataset
Before you start, please move the files of the type you choose to the `./data` folder.  

In this repository, there are 3 types of AG-News dataset: `formal`, `origin`, and `pre-test`.
- `formal`
    - It is the dataset for training, validating, and testing.
    - There are 3 files: `train.csv`, `validation.csv`, and `test.csv`.
        - `train.csv` contains 108000 data.
        - `validation.csv` contains 12000 data.
        - `test.csv` contains 7600 data.
- `origin`
    - It is the original dataset.
    - There are 2 files: `train.csv` and `test.csv`.
        - `train.csv` contains 120000 data.
        - `test.csv` contains 7600 data.
- `pre-test`
    - It is the very small dataset for code testing.
    - There are 3 files: `train.csv`, `validation.csv`, and `test.csv`.
        - `train.csv` contains 18 data.
        - `validation.csv` contains 18 data.
        - `test.csv` contains 18 data.


## Information
### Dataset
Original AG-News dataset contains two files: `train.csv` and `test.csv`. Each file contains 3 columns: `Class Index`, `Title`, and `Description`. The `Class Index` is an integer from 1 to 4, representing the category of the news. The `Title` and `Description` are the title and the description of the news, respectively. The `train.csv` contains 120,000 news and the `test.csv` contains 7,600 news.  

The analysis of the dataset is shown below:
```
Shape of train_df: (120000, 3)
title_over_512_data_num of train_df: 0
title_max_num of train_df: 19
title_min_num of train_df: 1
desc_over_512_data_num of train_df: 0
desc_max_num of train_df: 194
desc_min_num of train_df: 1

Shape of test_df: (7600, 3)
title_over_512_data_num of test_df: 0
title_max_num of test_df: 17
title_min_num of test_df: 1
desc_over_512_data_num of test_df: 0
desc_max_num of test_df: 141
desc_min_num of test_df: 4
```

Obviously, the `title` and `description` of the news are short, and the combined of word count of `title` and `description` cannot exceed 512. Therefore, the preprocessing strategy I have implemented is to concatenate the `title` and `description` strings. Then, I insert the `[CLS]` token at the beginning of the string and the `[SEP]` token between the `title` and `description` strings. Finally, I pad the sequence with `[PAD]` tokens until its length reaches 512.  

The sequence format is shown below:
```
string = [CLS] title [SEP] description [PAD]...[PAD]
```

### Model
I choose `bert-base-uncased` as the pretrained model to do this task.  
According to the types of implemented neural networks, there are 2 models:
- `BertWithSingleNN`
    - Utilizes a single neural network, where four classes share one classifier.
    - Each value of last dimesion in output tensors represents the probabilities of each class.
    ![BertWithSingleNN_Arch](https://drive.google.com/uc?export=view&id=1LLHSwNRmP07j_gH2aKUZlUoLRFB3tP9M)

- `BertWithMultiNNs`
    - Utilizes multiple neural networks, with one class and one classifier each, for the classification task.
    - Each classifier is responsible for a class of prediction.
    - The output tensors of all classifiers are concatenated to form the shape `[batch_size, class_num (4), 2]`.
    - In the last dimesion of each tensor, the first value represents the probability of `"is not this class"`; and the second value represents the probability of `"is this class"`.
    ![BertWithMultiNNs_Arch](https://drive.google.com/uc?export=view&id=1LIJFsTygT9XtmGtQZHmZcMuIVwI21C4J)


### Evaluation
I calculate the `Micro` and `Macro` `Precision`, `Recall`, `F1-score`, and `Error Rate` through the confusion matrix to evalute the model performance.  

Below are the formulas of each metric:
- `Precision`
    - `Micro Precision` = `TP` / (`TP` + `FP`)
    - `Macro Precision` = (`Precision of Class 1` + `Precision of Class 2` + `Precision of Class 3` + `Precision of Class 4`) / 4
- `Recall`
    - `Micro Recall` = `TP` / (`TP` + `FN`)
    - `Macro Recall` = (`Recall of Class 1` + `Recall of Class 2` + `Recall of Class 3` + `Recall of Class 4`) / 4
- `F1-score`
    - `Micro F1-score` = 2 * `Micro Precision` * `Micro Recall` / (`Micro Precision` + `Micro Recall`)
    - `Macro F1-score` = 2 * `Macro Precision` * `Macro Recall` / (`Macro Precision` + `Macro Recall`)
- `Error Rate`
    - `Micro Error Rate` = (`FP` + `FN`) / (`TP` + `FP` + `FN` + `TN`)
    - `Macro Error Rate` = (`Micro Error Rate of Class 1` + `Micro Error Rate of Class 2` + `Micro Error Rate of Class 3` + `Micro Error Rate of Class 4`) / 4

#### Loss Function and Data Format
I use `CrossEntropyLoss` as the loss function. With the different models, the format of predictions and labels used to calculate the loss are different.
- `BertWithSingleNN`
    - The format of predictions is `[batch_size, class_num (4)]`.
    - The format of labels is `[class_num (4)]` (using the `one-hot encoding` to represents the labels).
    - There is an example:
        - If the model predicts the given data is `Class 1`, the prediction may be `[0.9, 0.1, 0.1, 0.1]` (prediction.argmax() = 0).
        - If the label of the given data is `Class 2`, the label may be `[0, 1, 0, 0]` (one-hot encoding).
        - The loss of this data will be CrossEntropyLoss(`[0.9, 0.1, 0.1, 0.1]`, `[0, 1, 0, 0]`) = 1.6536.
- `BertWithMultiNNs`
    - The format of final predictions is `[batch_size, class_num (4), 2]`.
    - The format of labels is `[class_num (4), 2]`.
    - As mentioned above, in the last dimesion of each tensor, the first value represents the probability of `"is not this class"`; and the second value represents the probability of `"is this class"`.
    - There is an example:
        - If the model predicts the given data is `Class 1`, the prediction may be `[[0, 1], [1, 0], [1, 0], [1, 0]]`.
        - If the label of the given data is `Class 2`, the label may be `[[1, 0], [0, 1], [1, 0], [1, 0]]`.
        - The loss of this data will be CrossEntropyLoss(`[[0, 1], [1, 0], [1, 0], [1, 0]]`, `[[1, 0], [0, 1], [1, 0], [1, 0]]`) = 0.8133.

### Experiments
**There is no experiment results now.**

Due to the lack of computing resources, I only use the `pre-test` dataset to check whether the code executes as expected.
