import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for instantiating different model architectures dynamically.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            if name in cls._registry:
                logger.warning(f"Model '{name}' is already registered. Overwriting.")
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def build_model(cls, name, config, tokenizer=None):
        """
        Instantiates a model from the registry.
        
        Args:
            name (str): The registered name of the model.
            config (dict): Configuration dictionary.
            tokenizer: (Optional) Tokenizer for models that require vocab resizing.
        """
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found in registry. "
                             f"Available models: {list(cls._registry.keys())}")
                             
        logger.info(f"Building model architecture: {name}")
        model_class = cls._registry[name]
        return model_class(config, tokenizer)
