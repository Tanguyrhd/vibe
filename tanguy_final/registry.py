import os
import pickle

from colorama import Fore, Style
from tensorflow import keras

# def save_results(params: dict, metrics: dict) -> None:

#     # Save params locally
#     if params is not None:
#         params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
#         with open(params_path, "wb") as file:
#             pickle.dump(params, file)

#     # Save metrics locally
#     if metrics is not None:
#         metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
#         with open(metrics_path, "wb") as file:
#             pickle.dump(metrics, file)

#     print("✅ Results saved locally")




def save_model(model: keras.Model = None, type: str = None) -> None:
    """
    Persist trained model locally on the hard drive at f"/home/tanguy/code/Tanguyrhd/vibe//models/{type}.h5"
    """

    # Save model locally
    model_path = os.path.join("/home/tanguy/code/Tanguyrhd/vibe/", "models", type)
    model.save(model_path)

    print("✅ Model saved locally")

    return None





def load_model(type: str = None) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """

    print(Fore.BLUE + f"\nLoad model {type} from local registry...")

    # Get the latest model version name by the timestamp on disk
    local_model_paths = f"/home/tanguy/code/Tanguyrhd/vibe//models/{type}.h5"

    if not local_model_paths:
        return None

    model = keras.models.load_model(local_model_paths)

    print(f"✅ Model {type} loaded from local disk")

    return model
