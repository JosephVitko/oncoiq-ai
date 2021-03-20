# https://www.kaggle.com/arbazkhan971/invasive-ductal-carcinoma-classification-89-acc/notebook

from fastai import *
from fastai.vision import *
from images_path import images_path
import numpy as np
from fastprogress.fastprogress import force_console_behavior


def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()


def train():
    # setting up path for training data
    path = Path(images_path)
    pattern = r'([^/_]+).png$'
    fnames = get_files(path, recurse=True)
    tfms = get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)
    path.ls()

    # Data loading for training
    print("loading data")
    np.random.seed(40)
    data = ImageDataBunch.from_name_re(path, fnames, pattern, ds_tfms=tfms, size=50, bs=64, num_workers=4).normalize()

    # data exploration
    data.show_batch(rows=3, figsize=(7, 6), recompute_scale_factor=True)
    plt.savefig('data_batch.png')
    clear_pyplot_memory()

    # print data
    print(data.classes)
    len(data.classes)
    print(data.c)
    print(data)

    # create model
    learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir=Path('../kaggle/working'), path=Path("."))

    # find LR
    learn.lr_find()
    learn.recorder.plot(suggestions=True)
    plt.savefig('loss_vs_learning_plot.png')
    clear_pyplot_memory()

    # train model
    lr1 = 1e-3
    lr2 = 1e-1
    learn.fit_one_cycle(1, slice(lr1, lr2))

    # lr1 = 1e-3
    lr = 1e-1
    learn.fit_one_cycle(1, slice(lr))

    # hyper parameter tuning
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot()
    plt.savefig('loss_plot.png')
    clear_pyplot_memory()
    learn.fit_one_cycle(1, slice(1e-4, 1e-3))

    # interpret the results
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig('interpretation.png')
    clear_pyplot_memory()

    # save and load model
    learn.export()
    learn.model_dir = "/kaggle/working"
    learn.save("stage-1", return_path=True)


if __name__ == '__main__':
    master_bar, progress_bar = force_console_behavior()
    train()

