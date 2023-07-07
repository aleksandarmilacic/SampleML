# ML.NET Iris Classification

This project demonstrates the usage of ML.NET to build a multiclass classification model using the Iris dataset.

## Dataset

The dataset used in this project is the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). It contains measurements of sepal length, sepal width, petal length, and petal width for three different species of Iris flowers: Iris-setosa, Iris-versicolor, and Iris-virginica.

## Getting Started

To run the program, follow these steps:

1. Clone the repository or download the source code.

2. Ensure that you have .NET and ML.NET installed on your machine.

3. Open the project in your preferred IDE.

4. Download the dataset file [iris-data.txt](https://github.com/OpenAI-User-101/iris-data-ml/blob/main/iris-data.txt) and place it in the project directory.

5. Build and run the program.

## Program Overview

The program performs the following steps:

1. Loads the Iris dataset from the provided text file.

2. Splits the dataset into training and testing sets.

3. Defines a data processing pipeline that includes converting the label to a numeric key, concatenating the feature columns, normalizing the features, and converting the label back to its original value.

4. Trains a multiclass classification model using the training data and the defined pipeline.

5. Evaluates the trained model on the testing data to measure its accuracy.

6. Outputs the evaluation metrics, including accuracy, precision, recall, and F1-score.

## Dependencies

The project depends on the following packages:

- Microsoft.ML

These packages will be automatically restored when you build the project.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and use it according to your needs.

