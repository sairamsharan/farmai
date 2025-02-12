import pickle
import numpy as np

with open("crop_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    loaded_label_encoder = pickle.load(encoder_file)


X_testing = [90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]
X_testing = np.array([90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]).reshape(
    1, -1
)
X_testing = loaded_scaler.transform(X_testing)
y_testing = model.predict(X_testing.reshape(1, -1))[0]
label = loaded_label_encoder.inverse_transform([y_testing])[0]
print(label)
