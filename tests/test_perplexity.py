import unittest
from perplexity import PerplexityClient
import json

class TestPerplexityClient(unittest.TestCase):
    def test_send_prompt(self):
        client = PerplexityClient()
        response = client.send_prompt("Hello, what is the capital of France?")
        self.assertIn('choices', response)
        self.assertIsInstance(response['choices'], list)
        print(json.dumps(response, indent=2))

if __name__ == '__main__':
    unittest.main()
