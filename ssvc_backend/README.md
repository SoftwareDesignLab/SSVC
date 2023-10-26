# SSVC API README

(Enter goal here)


## Getting Started

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.9.17
- Flask 2.2.2
- Virtual Environment (highly recommended)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/SoftwareDesignLab/SSVC.git
   ```

2. Navigate to the project directory:

   ```bash
   cd SSVC
   ```
   
3. Install the required dependencies:

Install requirements through pip:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

To start the Flask API, use the following command:

NOTE: Running this server like this, should NOT be for production servers.
If you want to deploy this on a production server, please read the Flask
documentation on deployment here:
https://flask.palletsprojects.com/en/2.2.x/deploying/

For those who wish to run it in a docker container, the docker files are there for you to run. Just make sure to have your ports exposed and proper python pathing, when making changes and running Docker files.

```bash
python ssvc_backend/web_api/server.py
```

By default, the API runs on `http://localhost:5000`.

## API Endpoints

### SSVC

- Endpoint: `/ssvc`
- Description: The SSVC endpoint determines all values of the SSVC scoring tree, and determines how you should treat a CVE.
This endpoint also allows for assisting details that would allow for scoring of non-public CVEs that would assist in internal disclosures or fixes. You would just have to supply the CVE-ID you created, along with the description minimally.

**Required Parameters**:
- `cveId`: Id of the CVE you are determining

**Other Parameters**
- `description`: description of the vulnerability that you are scoring. (Optional: if not provided, it will check NVD's database). This is a recommended supplied parameter due to faster response times and avoiding being rate limited by NVD and preventing delays in scoring.
- `exploitStatus`: current status of the exploitability of the vulnerability. If you provide the value of POC, it will assume you have done exploit collection and take it at face value. However, it will still check the Known Exploited Vulnerabilities (KEV) database from NVD for active exploitation. 
- `mission` and `wellbeing`: these are the most unique parameters out of the parameters, because they depend on each other. Both must be provided to accurately score for the scenario relating to **YOUR** software, project and company. To know what to send into these values, please refer to [here](https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide%20508c.pdf) for specific values to submit. 
If these values are not submitted, the api will return a decision for each case of mission and wellbeing impact on the ssvc score. 


- Example Request: `GET /quickssvc?technicalImpact=TOTAL&automatable=True&exploitStatus=POC`
- Example Response:
  ```json
    {
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK*"
    }
  ```

- Response Explanation: Quick SSVC's response is given three values that relate to the severity of impact the vulnerability 
has on mission and wellbeing of the system. `ssvcScoreLow` refers to low mission impact, `ssvcScoreMedium` refers to medium mission impact, and so on.


### QuickSSVC

- Endpoint: `/quickssvc`
- Description: Retrieve an SSVC score quickly based off of already determined values of the SSVC decision tree.

- **Parameters**:
- `technicalImpact`: Technical Impact of CVE (TOTAL or PARTIAL)
- `automatable`: Whether an exploit is able to be automatically exploited (True or False)
- `exploitStatus`: The status of if the CVE is exploited (NONE, POC, ACTIVE) based off of SSVCs definitions.


- Example Request: `GET /quickssvc?technicalImpact=TOTAL&automatable=True&exploitStatus=POC`
- Example Response:
  ```json
    {
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK*"
    }
  ```

- Response Explanation: Quick SSVC's response is given three values that relate to the severity of impact the vulnerability 
has on mission and wellbeing of the system. `ssvcScoreLow` refers to low mission impact, `ssvcScoreMedium` refers to medium mission impact, and so on.



## Usage

To use the API, make HTTP requests to the desired endpoints. You can use tools like `curl`, `Postman`, or any programming language's HTTP client libraries to interact with the API.

## Error Handling

The API provides appropriate error responses for invalid requests. You will receive a JSON response with an error message in case of errors.

## Conclusion

