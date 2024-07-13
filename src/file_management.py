import hydra
from omegaconf import DictConfig, OmegaConf
import importlib

def get_class(class_path):
    '''
    Args:
        class_path (str): Path to the class, e.g. "teacher.PPOTeacherHeist"

    Returns:
        class: The class at the given path
    '''
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return class_