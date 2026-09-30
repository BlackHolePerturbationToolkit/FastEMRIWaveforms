"""FewBackendConsumer resolves to a FEW backend in every case (Task B5).

With no ``force_backend`` the GPUBackendTools base picked the first backend of ANY
package matching FEW's bare names (``gbt_cpu``); children built with that name were then
re-prefixed to the invalid ``few_gbt_cpu``, so ``FastKerrEccentricEquatorialFlux()`` with
no arguments raised.
"""
import unittest


class _Consumer:
    pass


class BackendConsumerTest(unittest.TestCase):
    def setUp(self):
        from few.utils.baseclasses import FewBackendConsumer

        class Toy(FewBackendConsumer):
            @classmethod
            def supported_backends(cls):
                return cls.GPU_RECOMMENDED()

        self.Toy = Toy

    def test_default_resolves_to_a_few_backend(self):
        self.assertTrue(self.Toy().backend.name.startswith("few_"), self.Toy().backend.name)

    def test_foreign_package_name_is_translated(self):
        self.assertEqual(self.Toy(force_backend="gbt_cpu").backend.name, "few_cpu")

    def test_plain_and_few_names(self):
        self.assertEqual(self.Toy(force_backend="cpu").backend.name, "few_cpu")
        self.assertEqual(self.Toy(force_backend="few_cpu").backend.name, "few_cpu")

    def test_child_built_from_parent_name(self):
        parent = self.Toy()
        child = self.Toy(force_backend=parent.backend.name)
        self.assertEqual(child.backend.name, parent.backend.name)


if __name__ == "__main__":
    unittest.main()
