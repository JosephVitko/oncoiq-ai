from fastai.vision import *
from pathlib import Path


def initialize_model(model_path):
    # modelPath should be absolute
    dummy_images_path = Path('inference_test_images')

    # setting up path for training data
    pattern = r'([^/_]+).png$'
    fnames = get_files(dummy_images_path, recurse=True)
    tfms = get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)

    # Data loading for training
    print("loading data")
    data = ImageDataBunch.from_name_re(dummy_images_path, fnames, pattern, ds_tfms=tfms, size=50, bs=1).normalize()

    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.load(model_path)
    return learn


def get_modelled_risk_score(images_path, model):
    # images path should be a path to a directory in which images to score are located
    risk_scores = []
    for f in images_path.iterdir():
        print(f)
        pred_class, _, outputs = model.predict(open_image(f))
        risk_scores.append(float(outputs[1]))
    return risk_scores

