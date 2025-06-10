# Final project for CMP263

This project focuses on the analysis and classification of Autism Spectrum Disorder (ASD) in children using various machine learning models. It emphasizes reproducibility by utilizing Docker for environment consistency.

## Project Structure

The project is organized into the following main directories:

* `data/`: Contains the dataset (`Autism-Child-Data.arff`) and will store generated graphics.
* `preprocessing/`: Includes scripts for data cleaning and preparation.
* `training/`: Contains scripts for model training, optimization, and utility functions.
* `visualization/`: Includes scripts for data exploration and visualization.
* `src/`: The main application logic.
* `Dockerfile`: Defines the Docker image for the project.
* `Makefile`: Provides convenient commands for building and running the Docker container.
* `README.md`: This file.

## Reproducibility with Docker

To ensure the reproducibility of the machine learning experiments, this project uses Docker. Docker containers encapsulate the application and its dependencies, guaranteeing that the code runs in the same environment every time, regardless of the host system.

## Getting Started

Follow these steps to set up and execute the project:

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/) installed on your system.
* `make` (optional, but recommended for ease of use).

### 1. Build the Docker Image
First, you need to build the Docker image. This process downloads all necessary dependencies and sets up the environment as defined in the `Dockerfile`.

Open your terminal or command prompt in the root directory of this project and run:

```bash
make build
```

### 2. Run the Project in a Docker Container
Once the image is built, you can run the machine learning pipeline within a Docker container using the following command:

```bash
make run
```

## Output and Results
As the main.py script executes, you will see various outputs in your terminal, including:

Data exploration summaries.
Evaluation results for each trained model (kNN, SVM, Gaussian Naive Bayes, Decision Tree, Random Forest).
Additionally in the `data/graphics/evaluating` folder there will be saved every confusion matrix for each model.