import logging
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class TanhSaturate(BatchFilter):
    """Saturate the values of an array to be floats between -1 and 1 by applying the tanh function.
    Args:
        array (:class:`ArrayKey`):
            The key of the array to modify.
        factor (scalar, optional):
            The factor to divide by before applying the tanh, controls how quickly the values
            saturate to -1, 1.
    """

    def __init__(self, array, scale=1, offset=0):

        self.array = array
        self.scale = scale
        self.offset = offset

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        array.data = self.offset + np.tanh(array.data / self.scale)

