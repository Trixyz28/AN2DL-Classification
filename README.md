# Artificial Neural Networks and Deep Learning 

This repository contains the notebook files and Python scripts for the homeworks 1 and 2 of the course **Artificial Neural Networks and Deep Learning** (Politecnico di Milano, 2022/2023).

The project fully relies on the **TensorFlow** framework, and the working environment is based on **Jupyter Notebook**. 

## Team Composition
Group name: **All Is Well**
- Fatma Hamila
- Kodai Takigawa ([@kodai-t](https://github.com/kodai-t))
- Zheng Maria Yu ([@Trixyz28](https://github.com/Trixyz28))



## Homework 1 - Image Classification

The goal of Homework 1 is to classify images of plants belonging to 8 different species.
It is a supervised task, since all images are labeled with the corresponding class.

The given [dataset](Homework1/dataset.zip) consists of 3542 RGB images (size 96x96), divided into 8 classes.

### Project Structure

- The [main](Homework1/main.ipynb) notebook contains the skeleton of our working procedure, and it keeps trace of the best model we obtained.
- [Training&FineTuning.py](Homework1/Training&FineTuning.py) is the script to train and fine-tune the models that will be used in ensembling.
- [Ensembling.ipynb](Homework1/Ensembling.ipynb) is used to ensemble the chosen models together.
- [specialist_for_species1_result.ipynb](Homework1/specialist_for_species1_result.ipynb): notebook for the training of the binary classification model for Species1 specifically.



## Homework 2 - Time Series Classification

In Homework 2, we were asked to classify multivariate time series sequences in a supervised manner. The total number of classes is 12.

The proposed [data](Homework2/dataset.zip) are in NumPy array format.
The extracted dataset has a shape of 2429x36x6: there are 2429 time series sequences, each one of length 36 steps and 6 channels.

### Project Structure
- [main.ipynb](Homework2/main.ipynb): notebook file for training the best individual model.
- [KFold.ipynb](Homework2/KFold.ipynb): notebook file for performing K-Fold Cross Validation and for ensembling the obtained models.
- [DataAnalysis.ipynb](Homework2/DataAnalysis.ipynb): notebook file to plot some statistics about the dataset.

