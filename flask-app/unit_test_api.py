import unittest
from run import app
import pickle
import logging
import sys

model = pickle.load(open('model.pkl','rb'))

class testapi(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_api_return(self):
        result = self.client.post("/run")
        self.assertIsNotNone(result)

    def test_model(self):
        result_m = model.get_coherence()
        self.assertIsNotNone(result_m)

    def test_logging(self):
        check = False
        with open("api.log", 'r') as f:
            for line in f:
                if ("app running") in line:
                    check = True
        f.close()
        self.assertEqual(check,True)
if __name__ == "__main__":
    unittest.main()