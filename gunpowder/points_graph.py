from .freezable import Freezable
from .points import Points, Point
from copy import copy as shallow_copy
from copy import deepcopy
from typing import Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PointsGraph(Points):
    """A subgraph of points that stores the points in a networkx graph.
    
    Differences between PointsGraph and Points:
    - crop() will create new nodes along edges crossing the roi.
        Given nodes `a`,`b` with `a` in the `roi`, and `b` out of the `roi`,
        a new node `c` will be created intersecting the `roi` between
        `a` and `b`. `c` will not use the same id as `b` to avoid problems
        encountered when two `inside` nodes are adjacent to one `outside` node

    Args:

        data (``dict``, ``int`` -> :class:`Point`):

            A dictionary of IDs mapping to :class:`Points<Point>`.

        spec (:class:`PointsSpec`):

            A spec describing the data.
    """

    def __init__(self, data, spec):
        self._graph = self._points_to_graph(data)
        self.spec = spec
        self.freeze()

    @property
    def data(self) -> Dict[int, Point]:
        self._graph_to_points(self.graph)

    @data.setter
    def data(self, data: Dict[int, Point]):
        self.graph = self.points_to_graph(data)

    def crop(self, roi, copy=False):
        """Crop this point set to the given ROI.
        """

        if copy:
            cropped = deepcopy(self)
        else:
            cropped = shallow_copy(self)

        cropped.data = {
            k: v for k, v in cropped.data.items() if roi.contains(v.location)
        }
        cropped.spec.roi = roi

        return cropped

    def merge(self, points, copy_from_self=False, copy=False):
        """Merge these points with another set of points. The resulting points
        will have the ROI of the larger one.

        This only works if one of the two point sets is contained in the other.
        In this case, ``points`` will overwrite points with the same ID in
        ``self`` (unless ``copy_from_self`` is set to ``True``).

        A copy will only be made if necessary or ``copy`` is set to ``True``.
        """

        self_roi = self.spec.roi
        points_roi = points.spec.roi

        assert self_roi.contains(points_roi) or points_roi.contains(
            self_roi
        ), "Can not merge point sets that are not contained in each other."

        # make sure self contains points
        if not self_roi.contains(points_roi):
            return points.merge(self, not copy_from_self, copy)

        # -> here we know that self contains points

        # simple case, self overwrites all of points
        if copy_from_self:
            return self if not copy else deepcopy(self)

        # -> here we know that copy_from_self == False

        # replace points
        if copy:
            merged = shallow_copy(self)
            merged.data.update(points.data)
        else:
            merged = deepcopy(self)
            merged.data.update(deepcopy(points.data))

        return merged
