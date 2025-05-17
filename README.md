# Model 1 - Menggunakan Flash
- Text Prediction Classification. 
- [API Postman](https://.postman.co/workspace/My-Workspace~9779896e-9ea0-40ca-8a08-35c00c022f9c/collection/35358448-7da7eb3b-12c3-4a5f-9b47-67af996bda1b?action=share&creator=35358448&active-environment=35358448-22648637-d5ff-4098-af33-774ba568b29d)

## Log Server Flask:
Error Shape Prediction
```
(mlp_env) C:\Users\ADVAN\Music\python\dicoding\capstone\flask>python app.py
2025-05-17 20:30:05.472524: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-05-17 20:30:10.264619: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-05-17 20:30:25.002514: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

Error Shape Input:
ValueError: Input 0 of layer &#34;functional_1&#34; is incompatible with the layer: expected shape=(None, 3),
        found shape=(1, 20)
```