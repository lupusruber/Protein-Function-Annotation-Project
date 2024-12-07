# Protein Function Annotation

## Overview

This project implements models for Protein-Protein Interaction (PPI) prediction, focusing on graph-based methods such as the SOTA GNN model call GIPA. 

The project is fully containerized with Docker, so no installation is required. You only need to build and run the Docker container, and everything will be set up for you.

## Project Structure

# Project Structure

  - **data_preparators/**: Contains data preparation scripts and preprocessed graph data.
  - **Dockerfile**: Docker configuration file for setting up the environment.
  - **gat_and_graph_sage/**: Includes scripts for training, evaluating, and experimenting with GraphSAGE and GAT models.
  - **gipa_wide_deep_model/**: Contains code for the Wide & Deep GIPA model.
  - **main_shell.sh**: The main shell script to runs the graph preprocessing, training, and evaluation scripts for the GIPA model.
  - **protein_dataset/**: Stores the dataset and related files for Protein-Protein Interaction (PPI).
  - **README.md**: Project README containing instructions and details.
  - **requirements.txt**: Lists Python dependencies required for the project.
  - **tests/**: Includes test scripts to verify the functionality of the project.

## How to Run

1. **Clone the repository:**

   Clone the project repository to your local machine:

   git clone https://github.com/lupusruber/Protein-Function-Annotation-Project.git
   cd ppi

2. **Build the Docker container:**

   Use the provided Dockerfile to build the Docker container. Ensure Docker is installed and running on your machine.

   docker build -t ppi_project .

3. **Run the Docker container:**

   Once the Docker container is built, you can run it using the following command:

   docker run -it --gpus all ppi_project

4. **Execute scripts within the container:**

   You can now execute the Python scripts for data preparation, model training, or evaluation from within the Docker container. All dependencies and environment configurations are handled inside the container.

5. **Results storage:**

The labels, predictions, and evaluation metrics are stored in the results/ directory. You can find the following files:

- Labels: Stored as .pt files, representing the true labels for the test data.
- Predictions: Stored as .pt files, containing the model's predictions for the test data.
- Metrics: Stored as .json files, containing various evaluation metrics for each experiment.

These results are organized for different configurations (e.g., BP, CC, MF) and can be accessed to assess the model's performance.
