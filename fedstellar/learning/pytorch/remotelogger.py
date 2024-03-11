import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor

from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict, _sanitize_callable_params
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

try:
    import wandb
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run, RunDisabled = None, None, None

_WANDB_AVAILABLE = RequirementCache("wandb")
_WANDB_GREATER_EQUAL_0_10_22 = RequirementCache("wandb>=0.10.22")
_WANDB_GREATER_EQUAL_0_12_10 = RequirementCache("wandb>=0.12.10")


class FedstellarWBLogger(Logger):
    LOGGER_JOIN_CHAR = "-"

    def __init__(
            self,
            name: Optional[str] = None,
            save_dir: _PATH = ".",
            version: Optional[str] = None,
            offline: bool = False,
            dir: Optional[_PATH] = None,
            id: Optional[str] = None,
            anonymous: Optional[bool] = None,
            project: str = "lightning_logs",
            log_model: Union[str, bool] = False,
            experiment: Union[Run, RunDisabled, None] = None,
            prefix: str = "",
            checkpoint_name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."  # pragma: no-cover
            )

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        if log_model and not _WANDB_GREATER_EQUAL_0_10_22:
            rank_zero_warn(
                f"Providing log_model={log_model} requires wandb version >= 0.10.22"
                " for logging associated model metadata.\n"
                "Hint: Upgrade with `pip install --upgrade wandb`."
            )

        super().__init__()
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None

        # paths are processed as strings
        if save_dir is not None:
            save_dir = os.fspath(save_dir)
        elif dir is not None:
            dir = os.fspath(dir)

        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = dict(
            name=name,
            project=project,
            dir=save_dir or dir,
            id=version or id,
            resume="allow",
            anonymous=("allow" if anonymous else None),
        )
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        # start wandb run (to create an attach_id for distributed modes)
        if _WANDB_GREATER_EQUAL_0_12_10:
            wandb.require("service")
            _ = self.experiment

        self._checkpoint_name = checkpoint_name
        self.local_step = 0
        self.global_step = 0

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Union[Run, RunDisabled]:
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

                # define default x-axis
                if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
                        self._experiment, "define_metric", None
                ):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        assert isinstance(self._experiment, (Run, RunDisabled))
        return self._experiment

    def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100, log_graph: bool = True) -> None:
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # FL round information
        self.local_step = step
        __step = self.global_step + self.local_step

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            self.experiment.log(dict(metrics, **{"trainer/global_step": __step}))
        else:
            self.experiment.log(metrics)

    @rank_zero_only
    def log_table(
            self,
            key: str,
            columns: Optional[List[str]] = None,
            data: Optional[List[List[Any]]] = None,
            dataframe: Any = None,
            step: Optional[int] = None,
    ) -> None:
        """Log a Table containing any object type (text, image, audio, video, molecule, html, etc).

        Can be defined either with `columns` and `data` or with `dataframe`.
        """

        metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_text(
            self,
            key: str,
            columns: Optional[List[str]] = None,
            data: Optional[List[List[str]]] = None,
            dataframe: Any = None,
            step: Optional[int] = None,
    ) -> None:
        """Log text as a Table.

        Can be defined either with `columns` and `data` or with `dataframe`.
        """

        self.log_table(key, columns, data, dataframe, step)

    @rank_zero_only
    def log_image(self, key: str, images: List[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        """Log images (tensors, numpy arrays, PIL Images or file paths).

        Optional kwargs are lists passed to each image (ex: caption, masks, boxes).
        """
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step)

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.
        """
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        """The project name of this experiment.

        Returns:
            The name of the project the current experiment belongs to. This name is not the same as `wandb.Run`'s
            name. To access wandb's internal experiment name, use ``logger.experiment.name`` instead.
        """
        return self._project

    @property
    def version(self) -> Optional[str]:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if the experiment exists else the id given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else self._id

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # log checkpoints as artifacts
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    @staticmethod
    @rank_zero_only
    def download_artifact(
            artifact: str,
            save_dir: Optional[_PATH] = None,
            artifact_type: Optional[str] = None,
            use_artifact: Optional[bool] = True,
    ) -> str:
        """Downloads an artifact from the wandb server.

        Args:
            artifact: The path of the artifact to download.
            save_dir: The directory to save the artifact to.
            artifact_type: The type of artifact to download.
            use_artifact: Whether to add an edge between the artifact graph.

        Returns:
            The path to the downloaded artifact.
        """
        if wandb.run is not None and use_artifact:
            artifact = wandb.run.use_artifact(artifact)
        else:
            api = wandb.Api()
            artifact = api.artifact(artifact, type=artifact_type)

        save_dir = None if save_dir is None else os.fspath(save_dir)
        return artifact.download(root=save_dir)

    def use_artifact(self, artifact: str, artifact_type: Optional[str] = None) -> "wandb.Artifact":
        """Logs to the wandb dashboard that the mentioned artifact is used by the run.

        Args:
            artifact: The path of the artifact.
            artifact_type: The type of artifact being used.

        Returns:
            wandb Artifact object for the artifact.
        """
        return self.experiment.use_artifact(artifact, type=artifact_type)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status != "success":
            # Currently, checkpoints only get logged on success
            return
        # log checkpoints as artifacts
        if self._checkpoint_callback and self._experiment is not None:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = (
                {
                    "score": s.item() if isinstance(s, Tensor) else s,
                    "original_filename": Path(p).name,
                    checkpoint_callback.__class__.__name__: {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                        ]
                        # ensure it does not break if `ModelCheckpoint` args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
                if _WANDB_GREATER_EQUAL_0_10_22
                else None
            )
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lackluster or custom name)
            self._logged_model_time[p] = t
