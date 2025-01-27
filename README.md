  # Scooter Demand Forecasting API

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Training the Model](#training-the-model)
7. [Running the Flask API](#running-the-flask-api)
8. [API Endpoints](#api-endpoints)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Version Control with Git](#version-control-with-git)
11. [Next Steps](#next-steps)
12. [Business Impact](#business-impact)
13. [Contributing](#contributing)
14. [License](#license)
15. [Contact](#contact)

---

## Project Overview

The **Scooter Demand Forecasting API** is a machine learning-driven solution designed to predict daily scooter demand in urban areas. Leveraging historical ride data and weather information, the model forecasts the number of scooter rides expected each day, enabling efficient resource allocation and operational planning for scooter-sharing companies.

---

## Features

- **Data Integration**: Combines ride data with corresponding weather information to enrich feature sets.
- **Feature Engineering**: Includes temporal features like day of the week and demand lags, as well as weather-related variables.
- **Model Training**: Utilizes a robust training pipeline with log transformation to handle large-scale demand data.
- **API Deployment**: Provides a RESTful API for real-time demand predictions using a trained model.
- **Evaluation Metrics**: Implements RMSE, MAE, and MAPE to comprehensively assess model performance.
- **Version Control**: Maintains project history and collaboration through Git and GitHub.

---

## Project Structure

scooter-demand-forecasting/ ├── input/ │ ├── voiholm.csv │ └── weather_data.csv ├── src/ │ ├── init.py │ ├── app.py │ ├── data_preparation.py │ ├── model_configs.py │ ├── metrics.py │ ├── train.py │ └── model_selection/ │ ├── init.py │ ├── forward_selection.py │ └── time_series_cv.py ├── .gitignore ├── requirements.txt ├── README.md └── LICENSE


- **input/**: Contains raw CSV files (`voiholm.csv` for ride data and `weather_data.csv` for weather information).
- **src/**: Houses all source code including the Flask API (`app.py`), data preparation scripts, model training pipeline, and model selection modules.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **requirements.txt**: Lists all Python dependencies required for the project.
- **README.md**: Project documentation (this file).
- **LICENSE**: Licensing information.

---

## Installation

### Prerequisites

- **Python 3.7+**
- **pip** (Python package installer)
- **Git** installed on your system

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/scooter-demand-forecasting.git
   cd scooter-demand-forecasting
   ```

2. **Create a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.

 ```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Preparation

Ensure that the input data files are placed in the input/ directory:

- voiholm.csv: Contains historical scooter ride data with at least the following columns:

  - ride_id
  - start_time (timestamp of the ride)
- weather_data.csv: Contains historical weather data with at least the following columns:

  - date (date of the weather record)
  - temperature
  - max_temperature
  - min_temperature
  - precipitation

4. **Training the Model**

## Steps
  1. Navigate to the Project Root
  
  ```bash
  cd scooter-demand-forecasting
  ```
  2. Run the Training Script
  The train.py script handles the entire training pipeline.
  ```bash
  python -m src.train \
  --rides_csv input/voiholm.csv \
  --weather_csv input/weather_data.csv \
  --test_cutoff_date 2020-08-25 \
  --model_out_path final_model.joblib
  ```

  Parameters:
  --rides_csv: Path to the ride data CSV.
  --weather_csv: Path to the weather data CSV.
  --test_cutoff_date: Date to split the training and test data.
  --model_out_path: Output path for the trained model artifact.

  3. Output

     After successful training, a final_model.joblib file will be generated in the project root. This file contains the trained model ready for deployment.

### Running the Flask API

The Flask API serves real-time predictions using the trained model.

## Steps

  1. Ensure the Trained Model Exists

     Verify that final_model.joblib is present in the project root. If not, run the training script as described above.

  2. Start the Flask Server

  ```bash
    python src/app.py
  ```
  The server will start on http://0.0.0.0:5000/.

  3. Environment Variables

    - MODEL_PATH: (Optional) Path to the trained model. Defaults to final_model.joblib in the project root.
    You can set it before running the server if needed:
    
   ```bash
    export MODEL_PATH=path/to/your_model.joblib
    python src/app.py
   ```




     
