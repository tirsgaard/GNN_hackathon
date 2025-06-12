from tdc import utils
from tdc.benchmark_group import admet_group
import requests
import json

# Name for model to be submitted
model_name = "Mean Predictor"
author = "rhti"

# The ADMET group benchmarks
problem_names = utils.retrieve_benchmark_names('ADMET_Group')
group = admet_group(path = 'data/')

results = {}
for problem_name in problem_names:
    benchmark = group.get(problem_name)

    predictions = {}
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']

    ## --- train your model --- ##
    mean = train_val["Y"].mean()
    y_pred = [mean] * len(test["Y"])
    ## --- end of training --- ##

    predictions[name] = y_pred
    result = group.evaluate(predictions)
    results[name] = result
    
# Format of the output

""" Example output dictionary
{
    "results": {
        caco2_wang' : {'caco2_wang': {'mae': 0.565}}
        'hia_hou' : {'hia_hou': {'roc-auc': 0.5}}
        ...
    },
    "model_name": "Mean Predictor",
    "author": "rhti",
    "extra_data": {} <- This can be used to store any additional information for bookkeeping
}
"""
output = {"results": results,
          "model_name": model_name,
          "author": author,
          "extra_data": {}}


# Define the endpoint URL
adress = "" # <- I will send this to you later
url = adress + '/submit'  # Replace with the actual URL if different

# Send the POST request
try:
    # Set the headers for JSON content
    headers = {'Content-type': 'application/json'}
    output_json = json.dumps(output)
    response = requests.post(url, data=output_json, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        print("Submission successful!")
        print("Response:", response.text)  # Print the response from the server
    else:
        print("Submission failed.")
        print("Status code:", response.status_code)
        print("Response:", response.text)  # Print the response from the server

except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
