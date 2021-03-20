from download_model import download_models
from deploy_model import initialize_model, get_modelled_risk_score
import os
from pathlib import Path


model_path = os.path.join(os.path.realpath('.'), download_models()[0])
model = initialize_model(model_path)
print(get_modelled_risk_score(Path('inference_test_images'), model))


