import csv, json, os, unittest

SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "splits")


class HoldoutV1Test(unittest.TestCase):
    def setUp(self):
        self.m = json.load(open(os.path.join(SPLITS, "holdout_v1.json"), encoding="utf-8"))

    def test_uploader_disjoint(self):
        val_users = {d.split("/")[0] for d in self.m["val_datasets"]}
        train_users = {d.split("/")[0] for d in self.m["train_datasets"]}
        self.assertEqual(val_users, set(self.m["val_users"]))
        self.assertFalse(val_users & train_users)
        self.assertEqual(len(self.m["val_datasets"]) + len(self.m["train_datasets"]), 128)

    def test_windows_in_val(self):
        rows = list(csv.DictReader(open(os.path.join(SPLITS, "holdout_v1_val_windows.csv"), encoding="utf-8")))
        self.assertEqual(len(rows), self.m["counts"]["val_windows"])
        val = set(self.m["val_datasets"])
        for r in rows:
            self.assertIn(f"{r['user']}/{r['dataset']}", val)
            self.assertEqual(int(r["num_frames"]), 16)


if __name__ == "__main__":
    unittest.main()
