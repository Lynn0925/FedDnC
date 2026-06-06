from .trainer import TrainerX, TrainerXU, TrainerBase, SimpleTrainer, SimpleNet  # isort:skip

# from .da import *
# from .dg import *
# from .ssl import *

# Lazy import of build to avoid circular imports when trainer modules are
# loaded directly (e.g. importlib.import_module("trainers.promptfl")).
def __getattr__(name):
    if name in ("TRAINER_REGISTRY", "build_trainer"):
        from . import build as _build
        return getattr(_build, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
