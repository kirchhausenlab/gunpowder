import logging
import os
import numpy as np

from gunpowder.ext import tensorflow as tf
from gunpowder.nodes.generic_train import GenericTrain
from gunpowder.volume import VolumeType, Volume

logger = logging.getLogger(__name__)

class Train(GenericTrain):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Train`.

    Args:

        meta_graph_filename: Filename of a tensorflow meta-graph storing the
            tensorflow graph containing an optimizer. A meta-graph file can be
            created by running::

                # create tensorflow graph
                ...

                # store it
                tf.train.export_meta_graph(filename=meta_graph_filename)

        optimizer: The name of the tensorflow operator performing a training
            iteration.

        loss: The name of the tensorflow tensor containing the loss.

        inputs (dict): Dictionary from the names of input tensors in the
            network to :class:``VolumeType`` or batch attribute name as string.

        outputs (dict): Dictionary from the names of output tensors in the
            network to :class:``VolumeType``. New volumes will be generated by
            this node for each entry (if requested downstream).

        gradients (dict): Dictionary from the names of output tensors in the
            network to :class:``VolumeType``. New volumes containing the
            gradient of an output with respect to the loss will be generated by
            this node for each entry (if requested downstream).

        volume_specs (dict, optional): An optional dictionary of
            :class:`VolumeType` to :class:`VolumeSpec` to set the volume specs
            generated volumes (``outputs`` and ``gradients``). This is useful
            to set the ``voxel_size``, for example, if they differ from the
            voxel size of the input volumes. Only fields that are not ``None``
            in the given :class:`VolumeSpec` will be used.

        save_every (int, optional): After how many iterations to create a
            checkpoint to store the learnt weights.
    '''

    def __init__(
            self,
            meta_graph_filename,
            optimizer,
            loss,
            inputs,
            outputs,
            gradients,
            volume_specs=None,
            save_every=2000):

        super(Train, self).__init__(
            inputs,
            outputs,
            gradients,
            volume_specs,
            spawn_subprocess=False)
        self.meta_graph_filename = meta_graph_filename
        self.optimizer = optimizer
        self.loss = loss
        self.session = None
        self.tf_gradient = {}
        self.graph = None
        self.saver = None
        self.save_every = save_every
        self.iteration = None

    def start(self):

        logger.info("Initializing tf session...")

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.__read_meta_graph()

        # replace names of operations/tensors with actual operations/tensors
        self.optimizer = self.graph.get_operation_by_name(self.optimizer)
        self.loss = self.graph.get_tensor_by_name(self.loss)

        # add symbolic gradients
        for tensor_name in self.gradients:
            tensor = self.graph.get_tensor_by_name(tensor_name)
            self.tf_gradient[tensor_name] = tf.gradients(
                self.loss,
                [tensor])[0]

    def train_step(self, batch, request):

        volume_outputs = self.__collect_requested_outputs(request)
        inputs = self.__collect_provided_inputs(batch)

        to_compute = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'iteration': tf.assign(self.iteration, self.iteration + 1)}
        to_compute.update(volume_outputs)

        # compute outputs, gradients, and update variables
        outputs = self.session.run(to_compute, feed_dict=inputs)

        for volume_type in volume_outputs:
            spec = self.spec[volume_type].copy()
            spec.roi = request[volume_type].roi
            batch.volumes[volume_type] = Volume(
                outputs[volume_type],
                spec)

        batch.loss = outputs['loss']
        batch.iteration = outputs['iteration'][0]

        if batch.iteration%self.save_every == 0:

            checkpoint_name = (
                self.meta_graph_filename +
                '_checkpoint_%i'%batch.iteration)

            logger.info(
                "Creating checkpoint %s",
                checkpoint_name)

            self.saver.save(
                self.session,
                checkpoint_name)

    def stop(self):

        if self.session is not None:

            self.optimizer = self.optimizer.name
            self.loss = self.loss.name

            self.session.close()
            self.graph = None
            self.session = None

    def __read_meta_graph(self):

        logger.info("Reading meta-graph...")

        # read the original meta-graph
        tf.train.import_meta_graph(
            self.meta_graph_filename + '.meta',
            clear_devices=True)

        # add custom gunpowder variables
        with tf.variable_scope('gunpowder'):
            self.iteration = tf.get_variable(
                'iteration',
                shape=1,
                initializer=tf.zeros_initializer,
                trainable=False)

        # create a saver for the current graph
        self.saver = tf.train.Saver()

        # find most recent checkpoint
        checkpoint_dir = os.path.dirname(self.meta_graph_filename)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        # restore model
        if checkpoint is not None:

            logger.info("Restoring model from %s", checkpoint)
            self.saver.restore(self.session, checkpoint)

        else:

            logger.info("No checkpoint found, initializing variables")
            self.session.run(tf.global_variables_initializer())

    def __collect_requested_outputs(self, request):

        volume_outputs = {}

        for output_name, volume_type in self.outputs.items():
            if volume_type in request:
                volume_outputs[volume_type] = output_name

        for output_name, volume_type in self.gradients.items():
            if volume_type in request:
                volume_outputs[volume_type] = self.tf_gradient[output_name]

        return volume_outputs

    def __collect_provided_inputs(self, batch):

        inputs = {}

        for input_name, input_type in self.inputs.items():
            if isinstance(input_type, VolumeType):
                if input_type in batch.volumes:
                    inputs[input_name] = batch.volumes[input_type].data
                else:
                    logger.warn("batch does not contain %s, input %s will not "
                                "be set", input_type, input_name)
            elif isinstance(input_type, np.ndarray):
                inputs[input_name] = input_type
            elif isinstance(input_type, str):
                inputs[input_name] = getattr(batch, input_type)
            else:
                raise Exception(
                    "Unknown network input type {}, can't be given to "
                    "network".format(input_type))

        return inputs
