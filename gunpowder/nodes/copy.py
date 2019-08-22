import copy
import logging

from .batch_filter import BatchFilter
from gunpowder.points import PointsKey
from gunpowder.array import ArrayKey
from gunpowder.batch_request import BatchRequest

from typing import Union

Key = Union[PointsKey, ArrayKey]

logger = logging.getLogger(__name__)


class Copy(BatchFilter):
    """
    Copies data provided by one key into another key

    Args:

        source (:class:`ArrayKey` or :class:`PointsKey`):

            The key of the array or points set to copy.

        target (:class:`ArrayKey` or :class:`PointsKey`):

            The key to copy data to.
    """

    def __init__(self, source: Key, target: Key):
        assert (
            type(source).__name__ == type(target).__name__
        ), "Source is a {} but Target is a {}".format(
            type(source).__name__, type(target).__name__
        )

        self.source = source
        self.target = target

    def setup(self):
        self.enable_autoskip()
        self.provides(self.target, self.spec[self.source])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.source] = request[self.target]
        return deps

    def process(self, batch, request):
        batch[self.target] = copy.deepcopy(batch[self.source])
