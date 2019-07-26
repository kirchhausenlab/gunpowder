from .provider_test import ProviderTest
from gunpowder import (
    ArrayKeys,
    ArraySpec,
    Array,
    Roi,
    Coordinate,
    Batch,
    BatchRequest,
    BatchProvider,
    RandomLocation,
    build,
)
import numpy as np

class TestSourceRandomLocation(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((-200, -20, -20), (1000, 100, 100)),
                voxel_size=(20, 2, 2)))

    def provide(self, request):

        batch = Batch()

        spec = request[ArrayKeys.RAW].copy()
        spec.voxel_size = Coordinate((20, 2, 2))

        data = np.zeros(request[ArrayKeys.RAW].roi.get_shape()/(20, 2, 2))
        if request.array_specs[ArrayKeys.RAW].roi.contains((0, 0, 0)):
            data[:] = 1

        batch.arrays[ArrayKeys.RAW] = Array(
            data=data,
            spec=spec)

        return batch

class CustomRandomLocation(RandomLocation):

    # only accept random locations that contain (0, 0, 0)
    def accepts(self, request):
        return request.array_specs[ArrayKeys.RAW].roi.contains((0, 0, 0))

class TestRandomLocation(ProviderTest):

    def test_output(self):

        pipeline = (
            TestSourceRandomLocation() +
            CustomRandomLocation()
        )

        with build(pipeline):

            for i in range(10):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            ArrayKeys.RAW: ArraySpec(
                                roi=Roi((0, 0, 0), (20, 20, 20)))
                        }))

                self.assertTrue(np.sum(batch.arrays[ArrayKeys.RAW].data) > 0)

    def test_random_seed(self):
        pipeline = TestSourceRandomLocation() + CustomRandomLocation

        with build(pipeline):
            seeded_rois = []
            unseeded_rois = []
            for i in range(100):
                batch_seeded = pipeline.request_batch(
                    BatchRequest(
                        {ArrayKeys.RAW: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                        random_seed=10,
                    )
                )
                seeded_rois.append(batch_seeded[ArrayKeys.RAW].spec.roi)
                batch_unseeded = pipeline.request_batch(
                    BatchRequest(
                        {ArrayKeys.RAW: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                        random_seed=10,
                    )
                )
                unseeded_rois.append(batch_unseeded[ArrayKeys.RAW].spec.roi)

            self.assertEqual(len(set(seeded_rois)), 1)
            self.assertGreater(len(set(unseeded_rois)), 1)

