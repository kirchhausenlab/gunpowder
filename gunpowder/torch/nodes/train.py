import logging
import numpy as np

from gunpowder.array import ArrayKey, Array
from gunpowder.ext import torch, tensorboardX, NoSuchModule
from gunpowder.nodes.generic_train import GenericTrain

logger = logging.getLogger(__name__)


class Train(GenericTrain):
    """Torch implementation of :class:`gunpowder.nodes.GenericTrain`.

    Args:

        model (subclass of ``torch.nn.Module``):

            The model to train.

        loss:

            The torch loss to use.

        optimizer:

            The torch optimizer to use.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors (argument names of the
            ``forward`` method) in the model to array keys.

        target (:class:`ArrayKey`):

            Array key which will be used as target values during training.

        output (:class:`ArrayKey`):

            Array key for the output array that will be generated by this node
            (if requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (at the moment only
            ``output``). This is useful to set the ``voxel_size``, for example,
            if they differ from the voxel size of the input arrays. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint_basename (``string``, optional):

            The basename used for checkpoint files. Defaults to ``model``.

        save_every (``int``, optional):

            After how many iterations to create a checkpoint to store the
            learnt weights.

        log_dir (``string``, optional):

            Directory for saving tensorboard summaries.

        log_every (``int``, optional):

            After how many iterations to write out tensorboard summaries.
    """

    def __init__(
        self,
        model,
        loss,
        optimizer,
        inputs,
        output,
        target,
        gradients=None,
        array_specs=None,
        checkpoint_basename="model",
        save_every=2000,
        log_dir=None,
        log_every=1,
    ):

        outputs = {"output": output}
        targets = {"output": target}

        # not yet implemented
        gradients = {}

        super(Train, self).__init__(
            inputs, outputs, gradients, array_specs, spawn_subprocess=False
        )

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.targets = targets
        self.checkpoint_basename = checkpoint_basename
        self.save_every = save_every

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.iteration = 0

        if not isinstance(tensorboardX, NoSuchModule) and log_dir is not None:
            self.summary_writer = tensorboardX.SummaryWriter(log_dir)
            self.log_every = log_every
        else:
            self.summary_writer = None
            if log_dir is not None:
                logger.warn("log_dir given, but tensorboardX is not installed")

    def start(self):

        checkpoint, self.iteration = self._get_latest_checkpoint(
            self.checkpoint_basename
        )

        if checkpoint is not None:

            logger.info("Resuming training from iteration %d", self.iteration)
            logger.info("Loading %s", checkpoint)

            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        else:

            logger.info("Starting training from scratch")

        logger.info("Using device %s", self.device)

    def train_step(self, batch, request):

        inputs = self.__collect_provided_inputs(batch)
        targets = self.__collect_provided_targets(batch)
        requested_outputs = self.__collect_requested_outputs(request)

        device_inputs = {
            k: torch.as_tensor(v, device=self.device) for k, v in inputs.items()
        }

        device_targets = {
            k: torch.as_tensor(v, device=self.device) for k, v in targets.items()
        }

        self.optimizer.zero_grad()
        outputs = {"output": self.model(**device_inputs)}

        logger.debug("model output: %s", outputs["output"])
        logger.debug("expected output: %s", device_targets["output"])
        loss = self.loss(outputs["output"], device_targets["output"])
        loss.backward()
        self.optimizer.step()

        for array_key, array_name in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_name].cpu().detach().numpy(), spec
            )

        batch.loss = loss.cpu().detach().numpy()
        self.iteration += 1
        batch.iteration = self.iteration

        if batch.iteration % self.save_every == 0:

            checkpoint_name = self._checkpoint_name(
                self.checkpoint_basename, batch.iteration
            )

            logger.info("Creating checkpoint %s", checkpoint_name)

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_name,
            )

        if self.summary_writer and batch.iteration % self.log_every == 0:
            self.summary_writer.add_scalar("loss", batch.loss, batch.iteration)

    def __collect_requested_outputs(self, request):

        array_outputs = {}

        for output_name, array_key in self.outputs.items():
            if array_key in request:
                array_outputs[array_key] = output_name

        return array_outputs

    def __collect_provided_inputs(self, batch):

        return self.__collect_provided_arrays(self.inputs, batch)

    def __collect_provided_targets(self, batch):

        return self.__collect_provided_arrays(self.targets, batch)

    def __collect_provided_arrays(self, reference, batch):

        arrays = {}

        for array_name, array_key in reference.items():
            if isinstance(array_key, ArrayKey):
                if array_key in batch.arrays:
                    arrays[array_name] = batch.arrays[array_key].data
                else:
                    logger.warn(
                        "batch does not contain %s, array %s will not " "be set",
                        array_key,
                        array_name,
                    )
            elif isinstance(array_key, np.ndarray):
                arrays[array_name] = array_key
            elif isinstance(array_key, str):
                arrays[array_name] = getattr(batch, array_key)
            else:
                raise Exception(
                    "Unknown network array key {}, can't be given to "
                    "network".format(array_key)
                )

        return arrays
