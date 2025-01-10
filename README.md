# Spam Classification Project

This README provides instructions to set up, train, run, and evaluate the *Spam Email Classifier*.

## Project Overview
The project implements a spam classifier using stochastic gradient descent (SGD) in Apache Spark, inspired by [source](https://arxiv.org/abs/1004.5168). It includes both a training phase to create a spam classification model (based on a labeled set of data) and a phase to classify new documents.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Installation and Setup](#installation-and-setup)
3. [File Structure](#file-structure)
3. [Running the Classifier](#running-the-classifier)
4. [Evaluation](#evaluation)
5. [Training Data - Assumptions and Important Notes](#training-data-assumptions-and-important-notes)
6. [File Structure](#file-structure)

---

### Dependencies
- **Apache Spark** (with version compatible with Hadoop 3.x)
- **Java Development Kit (JDK)** (v8 or later)
- **Scala** (version compatible with Spark)
- **Maven** (for managing project builds)
- **gcc** (optional; for compiling evaluation tool)
---

### Installation and Setup
1. **Clone or download the project files** into a working directory.
2. **Ensure all dependencies** listed above are installed and available in the system PATH.
3. **Unpack the data**

    ```bash
    bunzip2 train.txt.bz2
    bunzip2 test.txt.bz2
    ```

## File Structure

```src/``` files:
- ```Train.scala```: Spark program for training the classifier.
- ```Classify.scala```: Spark program for applying the classifier to test data.
- ```EnsembleClassify.scala```: Spark program for applying an ensemble classifier to test data (note: requires splitting data into groups).

```eval_tools/``` files:
- ```spam_metrics.c```: C program to compute classification metrics.
- ```spam_eval.sh```: Script to evaluate classification output.

## Running the Classifier

**TrainSpamClassifier**

This program trains a model using the specified input data.

Command:

```bash
spark-submit --driver-memory 2g --class spamclassifier.Train \
 target/spamclassifier.jar --input <input_file> --model <model_output_dir>
```

- Replace ```<input_file>``` with the training data file (e.g., spam.train.group_x.txt).
- Replace ```<model_output_dir>``` with the desired output directory for the model.

**ApplySpamClassifier**

This program applies the trained model to classify test data.

Command:

```bash
spark-submit --driver-memory 2g --class spamclassifier.Classify \
 target/spamclassifier.jar --input <input_file> \
 --output <output_dir> --model <model_dir>
```

- Replace <input_file> with ```test.txt``` or ```train.txt```
- Replace <output_dir> with the directory for classification results.
- Replace <model_dir> with the directory containing the trained model.

## Evaluation

To compile the C program used for evaluation:

```bash
gcc -O2 -o spam_metrics spam_metrics.c -lm
```

Run the following command to evaluate the classifier:

```bash
./spam_eval.sh <test_output_dir>
```

Example output:

```bash
1-ROCA%: 17.25
```

Lower values indicate better classification performance.

## Training Data Assumptions and Important Notes

1. **Training Data Format:** Each line in the training files contains a document ID, label (spam or ham), followed by hashed byte 4-gram features.
