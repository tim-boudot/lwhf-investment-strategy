from tensorflow import keras
from google.cloud import storage

#import parameters
from lwhf.params import *

#import mlflow
#from mlflow.tracking import MlflowClient



#SAVING MODEL TO GOOGLE CLOUD
def save_model_GCS(model: keras.Model, model_name: str) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(BUCKET_MODEL)

    #If the blob exists, delete it => TODO: store latest/best model of same features, HOW?
    blob = bucket.blob(f"models/{model_name}.keras")
    if blob.exists():
        blob.delete()

    #Create blob to store model
    blob = bucket.blob(f"models/{model_name}.keras")


    #Upload the local file
    blob.upload_from_filename(LOCAL_MODEL_PATH)
    print("✅ Model saved to GCS")

    return None


#check if model exists 
def check_model_GCS(model_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_MODEL)
    blob = bucket.blob(f"models/{model_name}.keras")


    if blob.exists():
        print("✅ Downloading model from GCS")
        blob.download_to_filename(LOCAL_MODEL_PATH)

    return blob.exists()



yf_2000_metrics_all
