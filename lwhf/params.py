import os
import numpy as np


GCP_PROJECT = os.environ.get("GCP_PROJECT")
DATASET = os.environ.get("DATASET")


TABLE = os.environ.get("TABLE")


REGION= os.environ.get("REGION")
BUCKET_MODEL= os.environ.get("BUCKET_MODEL")

LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")
