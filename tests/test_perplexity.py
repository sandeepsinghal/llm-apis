import unittest
from llm_client.perplexity import PerplexityClient
import json
from pydantic import BaseModel
from typing import List



class CountryCapital(BaseModel):
    country : str
    capital : str

class CountryCapitalsList(BaseModel):
    results : List[CountryCapital]

class TestPerplexityClient(unittest.TestCase):

    def test_send_prompt(self):
        client = PerplexityClient()
        response = client.send_prompt("Hello, what is the capital of France?")
        self.assertIn('choices', response)
        self.assertIsInstance(response['choices'], list)
        print(json.dumps(response, indent=2))

    def test_send_prompt_with_json_schema(self):
        client = PerplexityClient()

        response = client.send_prompt(
            "What is the capital of Germany, France and India?",
            json_schema=CountryCapitalsList.model_json_schema()
        )
        print(json.dumps(response, indent=2))
        structured_content = response['choices'][0]['message']['content']
        country_list = CountryCapitalsList.model_validate_json(structured_content)
        print(country_list)
        #print(json.dumps(structured_content, indent=2))

if __name__ == '__main__':
    unittest.main()
