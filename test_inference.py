# sends test request to Triton

import math
import numpy as np
import cv2
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

# Check server is live
print("Server live:", client.is_server_live())
print("Model ready:", client.is_model_ready("fundus_classifier"))

# Send test input
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
inputs = [httpclient.InferInput("input__0", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)
outputs = [httpclient.InferRequestedOutput("output__0")]

result = client.infer("fundus_classifier", inputs, outputs=outputs)
output = result.as_numpy("output__0")
print("Output shape:", output.shape)        # should be (1, num_classes)
print("Predicted class:", output.argmax())  # top class index