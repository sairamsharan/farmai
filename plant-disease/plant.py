from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="yjGS9C041cEmorXMHwZI"
)

result = CLIENT.infer("gg.jpg", model_id="plants-final/1")
print(result["predictions"][0]["class"])
print(f"{result['predictions'][0]['confidence'] * 100:.2f}%")
