from .points import Points, Point
from .points_spec import PointsSpec
from .roi import Roi
from .coordinate import Coordinate

from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


class SpatialGraph(nx.DiGraph):
    """
    An extension of a DiGraph that assumes each point has a spatial coordinate.
    Adds utility functions such as cropping to an ROI, shifting all points by
    an offset, and relabelling connected components.
    """

    def crop(self, roi: Roi, copy: bool = False, relabel_nodes=False):
        """
        Remove all nodes not in this roi.
        Does not shift nodes to be relative to the roi.
        """

        # Copy self if needed
        if copy:
            cropped = deepcopy(self)
        else:
            cropped = self

        if len(cropped.nodes) == 0:
            return cropped

        # Group nodes based on location
        all_nodes = set(cropped.nodes.keys())
        to_keep = set(
            key
            for key in cropped.nodes.keys()
            if roi.contains(cropped.nodes[key]["location"])
        )
        to_remove = all_nodes - to_keep

        # Get new boundary nodes and edges
        new_nodes, new_edges = self._handle_boundaries(
            to_keep, roi, next_node_id=max(all_nodes) + 1
        )

        # Handle node and edge changes
        for node in to_remove:
            cropped.remove_node(node)
        for node_id, attrs in new_nodes.items():
            cropped.add_node(node_id, **attrs)
        for u, v in new_edges:
            cropped.add_edge(u, v)

        cropped._relabel_connected_components()

        if relabel_nodes:
            cropped = nx.convert_node_labels_to_integers(cropped)

        return cropped

    def merge(self, other, copy: bool = False):
        """
        Merge this graph with another graph.
        Each Point may recieve a new id.
        Component attributes are recalculated for each Point.
        """
        current = deepcopy(self)
        other = deepcopy(other)
        to_remove = []
        for node, node_attrs in other.nodes.items():
            loc = node_attrs["location"]
            if all(
                np.isclose(
                    current.nodes.get(node, {}).get(
                        "location", np.array([float("inf")] * 3)
                    ),
                    loc,
                )
            ):
                to_remove.append(node)
        for node in to_remove:
            other.remove_node(node)

        combined = nx.disjoint_union(current, other)
        combined = nx.convert_node_labels_to_integers(combined)

        # SpacialGraph does not change any of the attributes of nx.DiGraph
        # so this should be fine.
        combined.__class__ = SpatialGraph

        combined = combined._relabel_connected_components()

        return combined

    def shift(self, offset: Coordinate):
        for point_attrs in self.nodes.values():
            point_attrs["location"] += offset

    def _relabel_connected_components(self):
        for i, connected_component in enumerate(nx.weakly_connected_components(self)):
            for point in connected_component:
                self.nodes[point]["component"] = i
        return self

    def _handle_boundaries(self, to_keep, roi, next_node_id):
        new_points = {}
        new_edges = []
        for u, v in self.edges:
            u_in, v_in = u in to_keep, v in to_keep
            if u_in != v_in:
                in_id, out_id = (u, v) if u_in else (v, u)
                in_attrs, out_attrs = (self.nodes[in_id], self.nodes[out_id])
                new_location = self._roi_intercept(
                    in_attrs["location"], out_attrs["location"], roi
                )
                if not all(np.isclose(new_location, in_attrs["location"])):
                    new_attrs = deepcopy(out_attrs)
                    new_attrs["location"] = new_location
                    new_points[next_node_id] = new_attrs
                    new_edges.append(
                        (in_id, next_node_id) if u_in else (next_node_id, in_id)
                    )
                    next_node_id += 1
        return new_points, new_edges

    def _roi_intercept(
        self, inside: np.ndarray, outside: np.ndarray, bb: Roi
    ) -> np.ndarray:

        offset = outside - inside
        distance = np.linalg.norm(offset)
        assert not np.isclose(distance, 0), "Offset cannot be zero"
        direction = offset / distance

        # `offset` can be 0 on some but not all axes leaving a 0 in the denominator.
        # `inside` can be on the bounding box, leaving a 0 in the numerator.
        # `x/0` throws a division warning, `0/0` throws an invalid warning (both are fine here)
        with np.errstate(divide="ignore", invalid="ignore"):
            bb_x = np.asarray(
                [
                    (np.asarray(bb.get_begin()) - inside) / offset,
                    (np.asarray(bb.get_end()) - inside) / offset,
                ],
                dtype=float,
            )

        with np.errstate(invalid="ignore"):
            s = np.min(bb_x[np.logical_and((bb_x >= 0), (bb_x <= 1))])

        # subtract a small amount from distance to round towards "inside" rather
        # than attempting to round down if too high and up if too low.
        new_location = np.floor(inside + s * (distance - 0.5) * direction)
        if not bb.contains(new_location):
            raise Exception(
                (
                    "Roi {} does not contain point {}!\n"
                    + "inside {}, outside: {}, distance: {}, direction: {}, s: {}"
                ).format(bb, new_location, inside, outside, distance, direction, s)
            )
        return new_location


class GraphPoint(Point):
    """An extension of ``Point`` that allows arbitrary attributes
    to be defined on each point in a networkx friendly way.

    Args:

        location (array-like of ``float``):

            The location of this point.
    """

    def __init__(self, location, **kwargs):
        super().__init__(location)
        self.thaw()
        self.kwargs = kwargs
        self.freeze()

    @property
    def attrs(self):
        attrs = {}
        attrs.update(self.kwargs)
        attrs["location"] = self.location
        return attrs

    def __repr__(self):
        return str(self.location)

    def copy(self):
        return GraphPoint(self.location, **self.kwargs)


class GraphPoints(Points):
    """A subclass of points that supports edges between points.
    Uses a networkx DiGraph to store points and cropping the graph
    generates nodes at the intersection of the boundary box and edges
    in the graph.

    Differences between PointsGraph and Points:
    - crop():
        will create new nodes along edges crossing the roi.
        Given adjacent nodes `a`,`b` with `a` in the `roi`, and `b` out of the `roi`,
        a new node `c` will be created intersecting the `roi` between
        `a` and `b`. `c` will not use the same id as `b` to avoid problems
        encountered when two `inside` nodes are adjacent to one `outside` node

    Args:

        data (``dict``, ``int`` -> :class:`Point`):

            A dictionary of IDs mapping to :class:`Points<Point>`.

        edges (``list``, ``tuple``, ``int``):

            A list of ID pairs (a,b) denoting edges from a to b.

        spec (:class:`PointsSpec`):

            A spec describing the data.
    """

    def __init__(
        self, data: Dict[int, Point], spec: PointsSpec, edges: Tuple[int, int] = None
    ):
        self.spec = spec
        self.data = data
        self.edges = edges
        self.freeze()

    @property
    def graph(self) -> SpatialGraph:
        """
        Every time you want to do something with the graph, it is recalculated
        from the data and edges attributes
        """
        graph = SpatialGraph()
        self._add_points(graph)
        self._add_edges(graph)
        return graph

    @property
    def data(self) -> Dict[int, Point]:
        return self._data

    @data.setter
    def data(self, new_data: Dict[int, Point]):
        self._data = new_data

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self._edges

    @edges.setter
    def edges(self, new_edges: Optional[List[Tuple[int, int]]]):
        self._edges = new_edges if new_edges is not None else []

    def crop(self, roi: Roi, copy: bool = False):
        """
        Crop this point set to the given ROI.
        """

        if copy:
            cropped = deepcopy(self)
        else:
            cropped = self

        # Crop the graph representation of the point set
        cropped_graph = cropped.graph.crop(roi)

        cropped = GraphPoints._from_graph(cropped_graph, spec=cropped.spec)

        # Override the current roi with the original crop roi
        cropped.spec.roi = roi
        return cropped

    def merge(self, points, copy_from_self=False, copy=False):
        """Merge these points with another set of points. The resulting points
        will have the ROI of the larger one.

        This only works if one of the two point sets is contained in the other.
        In this case, ``points`` will overwrite points with the same ID in
        ``self`` (unless ``copy_from_self`` is set to ``True``).

        A copy will only be made if necessary or ``copy`` is set to ``True``.

        TODO: Clear this up:
        This seems to be assuming that the larger roi contains
        a superset of the points in the smaller roi. With graphs
        this may not be the case since cropping creates new nodes
        on the boundaries. There are no guarantees on the IDs of
        these new nodes, so they may coincide with node IDs from
        the larger roi. Thus whether you update the smaller from
        the larger or vice versa, a node will be replaced.
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
            merged = self
            merged.data.update(points.data)
        else:
            merged = deepcopy(self)
            merged.data.update(deepcopy(points.data))

        return merged

    def remove(self, point_id):
        g = self.graph
        pres = g.predecessors(point_id)
        posts = g.successors(point_id)
        g.remove_node(point_id)
        assert len(pres) <= 1, "More than 1 parent!!!"
        for pre in pres:
            for post in posts:
                g.add_edge(pre, post)
        self = type(self)._from_graph(g)

    def disjoint_merge(self, other):
        """
        merges two Point sets that contain points from different
        sources. The Points in each set may or may not share IDs.
        """
        g1, g2 = self.graph, other.graph
        g = g1.merge(g2)
        return GraphPoints._from_graph(g)

    def _add_points(self, graph: SpatialGraph):
        for point_id, point in self.data.items():
            loc = point.location
            if isinstance(point, GraphPoint):
                graph.add_node(point_id, location=loc, **point.kwargs)
            else:
                graph.add_node(point_id, location=loc)
        return graph

    def _add_edges(self, graph: SpatialGraph):
        for u, v in self.edges:
            if u not in graph.nodes or v not in graph.nodes:
                logging.warning(
                    (
                        "{} is{} in the graph, {} is{} in the graph, "
                        + "thus an edge cannot be added between them"
                    ).format(
                        u,
                        "" if u in graph.nodes else " not",
                        v,
                        "" if v in graph.nodes else " not",
                    )
                )
                raise Exception("This should never happen!")
            else:
                graph.add_edge(u, v)
        return graph

    def _graph_to_points(self, graph) -> Dict[int, GraphPoint]:
        point_data = {}
        for point_id, point_attrs in graph.nodes.items():
            # do not deep copy location here. Modifying an attribute on the
            # point needs to modify that attribute on the graph
            loc = point_attrs.pop("location")
            point_data[point_id] = GraphPoint(location=loc, **point_attrs)
        return point_data

    @classmethod
    def _from_graph(cls, g: SpatialGraph, spec: PointsSpec = None):
        points = {
            point_id: GraphPoint(
                point_attrs["location"],
                **{
                    key: value
                    for key, value in point_attrs.items()
                    if key != "location"
                }
            )
            for point_id, point_attrs in g.nodes.items()
        }
        edges = list((u, v) for u, v in g.edges)
        spec = (
            PointsSpec(Roi(Coordinate((None,) * 3), Coordinate((None,) * 3)))
            if spec is None
            else spec
        )
        return cls(points, spec, edges)

