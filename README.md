
**Earthquake Prediction Project**

### Problem Statement

We need to predict earthquakes in order to warn people and try to save their lives, accommodations, and properties.

### Input and Output Data Format

**Input Format:** Date(YYYY/MM/DD) Time Latitude Longitude Depth Mag Magt Nst Gap Clo RMS SRC EventID

**Output Format:** The points on the map with predicted earthquakes.

### Metrics

This is considered a regression task. Therefore, the following metrics will be used:

-   **Mean Absolute Error (MAE)**: The average of the absolute differences between predicted and actual values.
-   **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values, penalizing larger errors more significantly.
-   **Root Mean Squared Error (RMSE)**: The square root of the MSE, providing an error measure in the same units as the target variable.

### Validation

-   70% Training set
-   15% Validation set
-   15% Test set

### Data

-   Example dataset: [http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Earthquakes_Over3.html](http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Earthquakes_Over3.html)
-   Additional data sources needed: Larger and more recent earthquake datasets from European or Russian agencies (e.g., EMSC, USGS, Geophysical Service of the Russian Academy of Sciences).

### Modeling

#### Baseline Models

-   RNN / LSTM

#### Main Model

We plan to use RNN or LSTM to predict the latitude and longitude of future earthquakes. The predicted coordinates will then be used to plot points on a map.

### Deployment

Deployment can be implemented as a web service with an input HTML page for entering data or uploading CSV files.

### Setup

Use **Poetry** for dependency management and packaging in Python. Create a Poetry environment using the pyproject.toml file. When opening a project that contains pyproject.toml but no interpreter is configured, PyCharm will suggest setting up a Poetry environment.

### Training Process

**Problem Formulation:** Define the task as time-series forecasting. RNNs are well-suited for sequential data where the order of elements matters.

**Data Preparation:**

-   Collect sequential dataset (time-series data of earthquakes).
-   Preprocess: Clean the data, handle missing values, and normalize features.
-   Split the data into training, validation, and test sets.
-   Create input-output sequences.
-   Encode/prepare data for the model (numerical format).

**Model Building:**

-   Choose architecture: Use LSTM (or GRU) layers, which are better at capturing long-term dependencies.
-   Define the model using Keras Sequential API.
-   Specify input shape and layer parameters (number of hidden units, etc.).

**Training the Model:**

-   Attach optimizer (Adam) and loss function (MSE for regression).
-   Train using Backpropagation Through Time (BPTT).
-   Use callbacks such as ModelCheckpoint and EarlyStopping to prevent overfitting.

### Production Preparation

1.  **Train the Model** Fit the model on the dataset using the fit() function with specified epochs and batch size. Monitor training metrics and validation loss.
2.  **Evaluate the Model** Test the model on the separate test dataset. Analyze results using MAE, MSE, and RMSE.
3.  **Make Predictions** Use the trained model for inference on new sequences. Interpret the predictions in the context of earthquake locations.

### Inference

Inference involves using the trained model to make predictions on new (unlabeled) data. It can be performed statically (pre-computed and cached) or dynamically.

**LSTM Network Configuration (Example):**

-   Input layer: 1
-   Output layer: 1
-   Hidden neurons: 25
-   Optimizer: Adam
-   Dropout: 0.1
-   Timestep: 240
-   Batch size: 240
-   Epochs: 1000 (parameters can be further optimized)

**Output Files:**

-   Earthquake_data_processed.xlsx (contains both prediction and actual values)
-   Plot file showing actual vs. predicted values