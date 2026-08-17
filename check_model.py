import pickle
import os

model = pickle.load(open("model.pkl", "rb"))

print("Model Type:")
print(type(model))

print("\nModel Size (MB):")
print(os.path.getsize("model.pkl") / (1024 * 1024))