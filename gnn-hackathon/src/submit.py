import requests
import json

def send_result(output):
    # Define the endpoint URL
    adress = "https://only-enormous-sparrow.ngrok-free.app" # <- I will send this to you later
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
