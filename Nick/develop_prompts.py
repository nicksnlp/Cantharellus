## Nikolay Vorontsov, 11.11.2024
## Code to feed training data to llms and ask them to find spans.
##
## Input. json datapoints with at least keys "model_input", "model_output_text"

import google.generativeai as genai
import configparser
import json


