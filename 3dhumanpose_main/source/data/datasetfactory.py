import glob
# Load all models
from os.path import dirname, basename, isfile

from source.configuration import Configuration
from source.exceptions.configurationerror import ConfigurationError
from source.data.JointDataset import JointDataset
from source.logcreator.logcreator import Logcreator

modules = glob.glob(dirname(__file__) + "/*.py")
for module in [basename(f)[:-3] for f in modules if
               isfile(f) and not f.endswith('__init__.py') and not f == "datasetfactory"]:
    __import__("source.data." + module)

class DataSetFactory(object):

    @staticmethod
    def load(general_cfg, dataset_folder, image_set, is_train):
        dataset_cfg = general_cfg.dataset
        
        dataset_params = getattr(general_cfg, dataset_cfg + "_params")

        all_datasets = JointDataset.__subclasses__()
        if dataset_cfg:
            dataset = [m(general_cfg, dataset_folder, image_set, is_train, dataset_params.num_joints) for m in all_datasets if m.name.lower() == dataset_cfg.lower()]
            if dataset and len(dataset) > 0:
                return dataset[0]
        raise ConfigurationError('data_collection.dataset')
