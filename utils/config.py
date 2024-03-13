from typing import Union, List, Tuple
import attr
from .object import HalfFrozenObject


@attr.s(frozen=True)
class RunConfiguration(HalfFrozenObject):
    """object specifying possible job configurations"""

    # experiment name (relevant for saving/loading trained models)
    experiment_name: str = attr.ib(default=None)

    # save-flag (e.g. for not saving GridSearch experiments)
    save_model: bool = attr.ib(default=None)

    # gpu
    gpu: int = attr.ib(default=None, validator=lambda i, a, v: v in (0, False))

    # run multiple experiments at one
    num_inits: int = attr.ib(default=None)
    num_splits: int = attr.ib(default=None)


@attr.s(frozen=True)
class DataConfiguration(HalfFrozenObject):
    """object specifying possible dataset configurations"""

    # ranomness
    split_no: int = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and v > 0
    )

    # dataset parameters
    dataset: str = attr.ib(default=None)
    root: str = attr.ib(default=None)
    split: str = attr.ib(
        default=None, validator=lambda i, a, v: v in ("public", "random")
    )
    # note that either the num-examples for the size values
    # must be specified, but not both at the same time!
    train_samples_per_class: Union[int, float] = attr.ib(default=None)
    val_samples_per_class: Union[int, float] = attr.ib(default=None)
    test_samples_per_class: Union[int, float] = attr.ib(default=None)
    train_size: float = attr.ib(default=None)
    val_size: float = attr.ib(default=None)
    test_size: float = attr.ib(default=None)

    # type of feature perturabtion, e.g. bernoulli_0.5
    ood_perturbation_type: str = attr.ib(default=None)
    ood_budget_per_graph: float = attr.ib(default=None)
    ood_budget_per_node: float = attr.ib(default=None)
    ood_noise_scale: float = attr.ib(default=None)
    ood_num_left_out_classes: int = attr.ib(default=None)
    ood_frac_left_out_classes: float = attr.ib(default=None)
    ood_left_out_classes: List[int] = attr.ib(default=None)
    ood_leave_out_last_classes: bool = attr.ib(default=None)


@attr.s(frozen=True)
class ModelConfiguration(HalfFrozenObject):
    """object specifying possible model configurations"""

    # model name
    model_name: str = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and len(v) > 0
    )
    pretrain_mode: str = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and len(v) > 0
    )
    # randomness
    seed: int = attr.ib(default=None, validator=lambda i, a, v: v is not None and v > 0)
    init_no: int = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and v > 0
    )

    # default parameters
    hidden_dim: Union[int, List[int]] = attr.ib(default=None)
    latent_dim: int = attr.ib(default=None)
    radial_layers: int = attr.ib(default=None)
    drop_prob: float = attr.ib(default=None)
    teleport: float = attr.ib(default=None)
    iteration_step: float = attr.ib(default=None)

    reduction: str = attr.ib(
        default=None, validator=lambda i, a, v: v in (None, "sum", "mean", "none")
    )
    emse_loss_weight: float = attr.ib(default=None)
    uce_loss_weight: float = attr.ib(default=None)
    uce_log_loss_weight: float = attr.ib(default=None)
    ce_loss_weight: float = attr.ib(default=None)
    entropy_reg_weight: float = attr.ib(default=None)
    reconstruction_reg_weight: float = attr.ib(default=None)
    tv_alpha_reg_weight: float = attr.ib(default=None)
    tv_vacuity_reg_weight: float = attr.ib(default=None)
    probability_teacher_weight: float = attr.ib(default=None)
    alpha_teacher_weight: float = attr.ib(default=None)


@attr.s(frozen=True)
class TrainingConfiguration(HalfFrozenObject):
    """object specifying possible training configurations"""

    lr: float = attr.ib(default=None)
    weight_decay: float = attr.ib(default=None)
    epochs: int = attr.ib(default=None)
    stopping_patience: int = attr.ib(default=None)
    stopping_metric: str = attr.ib(default=None)
    warmup_epochs: int = attr.ib(default=None)


def configs_from_dict(
    d: dict,
) -> Tuple[
    RunConfiguration, DataConfiguration, ModelConfiguration, TrainingConfiguration
]:
    """utility function converting a dictionary (e.g. coming from a .yaml file) into the corresponding configuration objects

    Args:
        d (dict): dictionary containing all relevant configuration parameters

    Returns:
        Tuple[RunConfiguration, DataConfiguration, ModelConfiguration, TrainingConfiguration]: tuple of corresponding objects for run, data, model, and training configuration
    """
    run = RunConfiguration(**d["run"])
    data = DataConfiguration(**d["data"])
    model = ModelConfiguration(**d["model"])
    training = TrainingConfiguration(**d["training"])

    return run, data, model, training
