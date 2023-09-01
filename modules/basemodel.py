"""base module for all modules."""
import toml #pylint: disable=unused-import
class Model():
    """Abstract class for models."""
    def __init__(self, model_path) -> None:
        """Initialize the Model class."""
        # loading model at `model_path`
        # and it's config, at `model_path.toml`
        raise NotImplementedError()
    def __call__(self, prompt, cfg:dict = None) -> str:
        """Generate text using the model."""
        # generate text
        raise NotImplementedError()
    def generate(self, *args, **kwargs):
        """Generate text using the model."""
        return self(*args, **kwargs)
    def switch_model(self, model_name):
        """Switch to another model."""
        #switch to another model
        raise NotImplementedError()

MODEL_NAME = "Model"
MODEL = [Model]
