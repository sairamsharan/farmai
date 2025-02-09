# Plant Disease and Crop Prediction API

This project is a FastAPI-based web application that predicts plant diseases from leaf images and offers crop prediction based on agricultural parameters. The app uses machine learning models for both crop and disease predictions and provides suggestions for treatment and fertilizers based on identified plant diseases.

## Features

- **Crop Prediction**: Predict the best crop to grow based on agricultural parameters such as nitrogen, phosphorus, potassium levels, temperature, humidity, etc.
- **Plant Disease Detection**: Detect plant diseases from uploaded leaf images and provide disease-specific recommendations.
- **Fertilizer and Treatment Recommendations**: Get personalized fertilizer and treatment recommendations based on the detected plant disease.

## Technologies Used

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **Uvicorn**: ASGI server to serve the FastAPI app.
- **Pickle**: For loading pre-trained machine learning models.
- **InferenceHTTPClient**: For interacting with the external model API.
- **Pillow (PIL)**: For image handling.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Pillow
- Numpy
- Scikit-learn (for model loading)
- Inference SDK (for external model inference)

## Setup and Installation

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/pranayyb/agro.git
cd agro
```

### 2. Create a Virtual Environment (Optional but Recommended)

It is highly recommended to use a virtual environment to isolate the project dependencies.

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Application

```bash
uvicorn main:app --reload
```

## Endpoints

### 1. `/predict-crop`

#### Description:

Predicts the best crop to grow based on soil and environmental parameters.

#### Method:

`POST`

#### Request:

- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "nitrogen": float,      # Nitrogen content in the soil (e.g., 50.0)
    "phosphorus": float,    # Phosphorus content in the soil (e.g., 20.0)
    "potassium": float,     # Potassium content in the soil (e.g., 30.0)
    "temperature": float,   # Temperature in Celsius (e.g., 25.0)
    "humidity": float,      # Relative Humidity as a percentage (e.g., 80.0)
    "ph": float,            # Soil pH value (e.g., 6.5)
    "rainfall": float       # Rainfall in mm (e.g., 200.0)
  }
  ```

#### Response:

- **Content-Type**: `application/json`
- **Response Body**:
  ```json
  {
    "predicted_crop": "crop_name" // Name of the predicted crop (e.g., "rice")
  }
  ```

### 2. `/detect-disease/{plant_type}`

#### Description:

Detects plant diseases from uploaded leaf images and recommends fertilizers and treatments.

#### Method:

`POST`

#### Request:

- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `file`: An image file of a plant leaf.

#### Supported Plant Types (`plant_type`)

The following plant types are supported for disease detection:

- apple
- bell_pepper
- blueberry
- cherry
- corn
- grape
- peach
- potato
- raspberry
- soyabean
- squash
- strawberry
- tomato
- rice
- wheat
- sugarcane
- coffee
- tea

#### Configuration

- **`PLANT_MODEL_MAPPING`**: A dictionary mapping plant types (e.g., "tomato", "potato") to their respective model IDs.
- **`PREDICTED_LABEL_TO_DISEASE_MAPPING`**: A dictionary mapping predicted labels to human-readable disease names.
- **`plant_diseases_to_fertilizers`**: A dictionary mapping plant diseases to their respective fertilizer and treatment recommendations.
- **`CLIENT`**: The client used for inference .

#### Response:

- **Content-Type**: `application/json`
- **Response Body**:
  ```json
  {
    "plant_type": "tea",
    "disease": "Red Spider infested tea leaf",
    "confidence": "53.94%",
    "fertilizer_recommendation": "Potassium-rich fertilizers to improve plant resistance",
    "treatment_recommendation": "Miticides, natural predators like ladybugs, maintain low humidity"
  }
  ```
