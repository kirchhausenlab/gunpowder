from .provider_test import ProviderTest
import unittest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    BatchFilter,
    Batch,
    GraphPoint as Point,
    GraphPoints as Points,
    PointsSpec,
    PointsKey,
    PointsKeys,
    RandomLocation,
    Crop,
    build,
    Roi,
    Coordinate,
)

import numpy as np


class TestSourcePoints(BatchProvider):
    def __init__(self):

        self.points = Points(
            {1: Point([1, 1, 1]), 2: Point([500, 500, 500]), 3: Point([550, 550, 550])},
            PointsSpec(roi=Roi((None, None, None), (None, None, None))),
        )
        self.edges = [(1, 2), (2, 3)]

    def setup(self):

        self.provides(PointsKeys.TEST_POINTS, self.points.spec)

    def provide(self, request):

        batch = Batch()

        roi = request[PointsKeys.TEST_POINTS].roi

        data = {}

        for point_id, point in self.points.data.items():
            if roi.contains(point.location):
                data[point_id] = point.copy()

        points = Points(
            data,
            PointsSpec(roi),
            [(u, v) for u, v in self.edges if u in data and v in data],
        )

        batch[PointsKeys.TEST_POINTS] = points

        return batch


class GrowFilter(BatchFilter):
    def prepare(self, request):
        grow = Coordinate([50, 50, 50])
        for key, spec in request.items():
            spec.roi = spec.roi.grow(grow, grow)
            request[key] = spec
        return request

    def process(self, batch, request):
        pass


class TestGraphPoints(ProviderTest):
    def test_output(self):

        PointsKey("TEST_POINTS")

        pipeline = TestSourcePoints() + GrowFilter()

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        PointsKeys.TEST_POINTS: PointsSpec(
                            roi=Roi((475, 475, 475), (50, 50, 50))
                        )
                    }
                )
            )

            points = batch[PointsKeys.TEST_POINTS].data
            expected_points = (
                tuple(np.array([500, 500, 500])),
                tuple(np.array([524, 524, 524])),
            )
            seen_points = tuple(
                tuple(np.array(point.location)) for point in points.values()
            )
            self.assertCountEqual(expected_points, seen_points)

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        PointsKeys.TEST_POINTS: PointsSpec(
                            roi=Roi((25, 25, 25), (500, 500, 500))
                        )
                    }
                )
            )

            points = batch[PointsKeys.TEST_POINTS].data
            expected_points = (
                tuple(np.array([25, 25, 25])),
                tuple(np.array([500, 500, 500])),
                tuple(np.array([524, 524, 524])),
            )
            seen_points = tuple(
                tuple(np.array(point.location)) for point in points.values()
            )
            self.assertCountEqual(expected_points, seen_points)
