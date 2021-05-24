
import os
from comet_ml import Experiment
from torch.utils.tensorboard import SummaryWriter
from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator

def init_comet():
    if not Configuration.get('training.general.log_to_comet'):
        return None

    api_key = os.getenv('COMET_API_KEY')
    project_name = os.getenv('COMET_PROJECT_NAME')
    workspace = os.getenv('COMET_WORKSPACE')
    try:
        if project_name is None:
            raise ValueError
        if workspace is None:
            raise ValueError

        experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        experiment.set_name(os.path.basename(os.path.normpath(Configuration.output_directory)))
        experiment.add_tag(Configuration.get('training.model.name'))

        parameters = Configuration.get('training')
        data_collection_params = Configuration.get('data_collection')

        experiment.log_parameter("Model", parameters.model.name)
        experiment.log_parameter("Training Collections", data_collection_params.dataset)
        experiment.log_parameter("Augmentations", data_collection_params.augmentations)
        experiment.log_parameter("Img size", data_collection_params.image_size)
        experiment.log_parameter("Loss-Function", parameters.loss.loss_function)
        experiment.log_parameter("Optimizer", parameters.optimizer.name)
        experiment.log_parameter("Optimizer-LR", parameters.optimizer.lr)
        experiment.log_parameter("Optimizer-LR", parameters.optimizer.beta1)
        experiment.log_parameter("Optimizer-LR", parameters.optimizer.beta2)
        experiment.log_parameter("lr-sched.", parameters.lr_scheduler.name)

        return experiment
    except ValueError:
        Logcreator.error("Comet initialization was not successful, the following information was missing:")
        if api_key is None:
            Logcreator.error("- COMET_API_KEY")
        if project_name is None:
            Logcreator.error("- COMET_PROJECT_NAME")
        if workspace is None:
            Logcreator.error("- COMET_WORKSPACE")
        Logcreator.error("Pleas add the missing parameters to a file called .env")
        Logcreator.info("Training now continues without logging to Comet")
        return None

