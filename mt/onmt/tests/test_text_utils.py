import unittest
from onmt.inputters.text_utils import parse_features


class TestTextUtils(unittest.TestCase):
    def test_parse_features(self):
        input_data = "this is a test"
        text, feats = parse_features(input_data)
        self.assertEqual(text, "this is a test")

        input_data = "this is a test"
        text, feats = parse_features(input_data, 0, "0")
        self.assertEqual(text, "this is a test")
