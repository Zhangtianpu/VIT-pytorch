import Utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import yaml
from Model.VitNet import VitNet
from Dataset.GarbageClsDataset import GarbageDataset
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_curve(x, train, test, title, xlabel, ylabel, is_save=False, path=None):
    plt.figure(figsize=(20, 5))
    plt.plot(x, train)
    plt.plot(x, test)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['train', 'test'])
    plt.title(title)
    plt.grid(visible=True, linestyle='--')

    if is_save:
        if path is None:
            raise Exception("Path is None")
        plt.savefig(path, dpi=600, format='png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    config_file = './Config/GarbageCls_VIT_config.yml'
    showcase_stored_folder = './ExperimentImages/GarbageCls_VIT'
    if os.path.exists(showcase_stored_folder) is False:
        os.makedirs(showcase_stored_folder)
    with open(config_file) as f:
        yaml_config = yaml.safe_load(f)

        data_config = yaml_config['data_configuration']
        model_config = yaml_config['model_configuration']
        train_config = yaml_config['train_configuration']

    logs = Utils.load_logs(folder_path=train_config.get("experiment_result_folder"),
                           filename=train_config.get('experiment_result_file'))

    train_loss = np.array(logs['train_loss'])
    test_loss = [loss for loss in logs['test_loss']]
    test_loss = np.array(test_loss)
    X = range(len(train_loss))

    train_acc = np.array(logs['train_acc'])
    test_acc = [acc for acc in logs['test_acc']]
    test_acc = np.array(test_acc)

    train_miou = np.array(logs['train_f1'])
    test_miou = [miou for miou in logs['test_f1']]
    test_miou = np.array(test_miou)

    plot_curve(X, train_loss, test_loss, "Learning Curve", "Epoch", "Loss", is_save=True,
               path=os.path.join(showcase_stored_folder, 'LearningCurve.png'))
    plot_curve(X, train_acc, test_acc, "Accuracy Curve", "Epoch", "Accuracy", is_save=True,
               path=os.path.join(showcase_stored_folder, 'AccuracyCurve.png'))
    plot_curve(X, train_miou, test_miou, "F1 Score Curve", "Epoch", "F1", is_save=True,
               path=os.path.join(showcase_stored_folder, 'F1.png'))


