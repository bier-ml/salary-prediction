import os
import pickle
import sklearn

def load_model_from_pickle(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f'Warning: {file_path} not found.')
        return None


knn_model = load_model_from_pickle("models/knn_model.pkl")
