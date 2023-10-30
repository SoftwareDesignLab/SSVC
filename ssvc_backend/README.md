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
- `mission` and `wellbeing`: these are the unique parameters out of the parameters, because they depend on each other. Both must be provided to accurately score for the scenario relating to **YOUR** software, project and company. To know what to send into these values, please refer to [here](https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide%20508c.pdf) for specific values to submit. 
If these values are not submitted, the api will return a decision for each case of mission and wellbeing impact on the ssvc score. 

#### Example Requests (SSVC)

**Example 1: Just CVE-ID:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: SSVC's goal is to give you the most information about your CVE automatically, so we run a varierty of algorithms and research methods to automatically score these cves in the most accurate way we can. This response is what you would get if you just provided an already published CVE that was found in the NVD. It would grab the description and then process it into our Machine Learning models to determine its technical impact on a system and whether it is able to be automatically exploited. It also will determine its exploit status based off the information we have, either KEV or CWEtoPOC, or Nothing, and we will let you know via the `exploitStatusRationale` response. Since we did not provide the mission and wellbeing you will notice that there is 3 scores  that correlate to the level of impact an exploit has on your company or system. 

**Example 2: CVE-ID and Exploit Status given due to collection:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381&exploitStatus=POC`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "POC",
      "exploitStatusRationale": "Exploit Collection",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: This response is the same as the previous response with two main things that are different. The `exploitStatus` is updated to POC and the `exploitStatusRationale` is updated to "Exploit Collection". Exploit Collection is the value given when you do your own research of the cve and have some sort of proof of concept exploitation of the vulnerability and let the api know that.

**Example 3: CVE-ID and Mission and Well Being:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381&mission=MINIMAL&wellbeing=MINIMAL`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "missionAndWellbeing": "LOW",
      "ssvcScore": "TRACK",
      "technicalImpact": "partial"
    }  
  ```

- Response Explanation: This response is the same as the first response, however there are less `ssvcScore` values and the inclusion of the severity of impact on `missionAndWellbeing` which is returned in the response. The ssvc score value returned is the specific value for this provided instance.

**Example 4: Private/Undisclosed CVE-ID and Provided Description:**
- Example Request: `GET /ssvc?cveId=CVE-2150-1000&description=There is an internal module that can be exploited to cause reomte code execution within the kernel.`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2150-1000",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "total"
    } 
  ```

- Response Explanation: This response is the same as the previous ones, but the main thing this response shows is, SSVC automatic scoring can be used for internal and hypothetical situations for you and your team, you don't have to wait for NVD disclosures if you need to know.


**Example 4: Invalid Response (Improper CVE / Rate Limit)**
- Example Request: `GET /ssvc?cveId=CVE-2100-3381`
- Example Response:
  ```json
    {
      "cveId": "CVE-2100-3381",
      "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
    }
  ```

- Response Explanation: This is the response you will get if something goes wrong, either you requested an invalid CVE or you're requesting too fast (NVD Description Pulling Limitations)

#### NOTE
You can use any and all combinations of the parameters above in your requests. The above examples are showing common examples of what could be requests made to the api.

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

#### Example Requests (SSVC)

**Example 1: Just CVE-ID:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: SSVC's goal is to give you the most information about your CVE automatically, so we run a varierty of algorithms and research methods to automatically score these cves in the most accurate way we can. This response is what you would get if you just provided an already published CVE that was found in the NVD. It would grab the description and then process it into our Machine Learning models to determine its technical impact on a system and whether it is able to be automatically exploited. It also will determine its exploit status based off the information we have, either KEV or CWEtoPOC, or Nothing, and we will let you know via the `exploitStatusRationale` response. Since we did not provide the mission and wellbeing you will notice that there is 3 scores  that correlate to the level of impact an exploit has on your company or system. 

**Example 2: CVE-ID and Exploit Status given due to collection:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381&exploitStatus=POC`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "POC",
      "exploitStatusRationale": "Exploit Collection",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: This response is the same as the previous response with two main things that are different. The `exploitStatus` is updated to POC and the `exploitStatusRationale` is updated to "Exploit Collection". Exploit Collection is the value given when you do your own research of the cve and have some sort of proof of concept exploitation of the vulnerability and let the api know that.

**Example 3: CVE-ID and Mission and Well Being:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381&mission=MINIMAL&wellbeing=MINIMAL`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "missionAndWellbeing": "LOW",
      "ssvcScore": "TRACK",
      "technicalImpact": "partial"
    }  
  ```

- Response Explanation: This response is the same as the first response, however there are less `ssvcScore` values and the inclusion of the severity of impact on `missionAndWellbeing` which is returned in the response. The ssvc score value returned is the specific value for this provided instance.

**Example 4: Private/Undisclosed CVE-ID and Provided Description:**
- Example Request: `GET /ssvc?cveId=CVE-2150-1000&description=There is an internal module that can be exploited to cause reomte code execution within the kernel.`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2150-1000",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "total"
    } 
  ```

- Response Explanation: This response is the same as the previous ones, but the main thing this response shows is, SSVC automatic scoring can be used for internal and hypothetical situations for you and your team, you don't have to wait for NVD disclosures if you need to know.


**Example 4: Invalid Response (Improper CVE / Rate Limit)**
- Example Request: `GET /ssvc?cveId=CVE-2100-3381`
- Example Response:
  ```json
    {
      "cveId": "CVE-2100-3381",
      "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
    }
  ```

- Response Explanation: This is the response you will get if something goes wrong, either you requested an invalid CVE or you're requesting too fast (NVD Description Pulling Limitations)

#### NOTE
You can use any and all combinations of the parameters above in your requests. The above examples are showing common examples of what could be requests made to the api.

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

#### Example Requests (SSVC)

**Example 1: Just CVE-ID:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: SSVC's goal is to give you the most information about your CVE automatically, so we run a varierty of algorithms and research methods to automatically score these cves in the most accurate way we can. This response is what you would get if you just provided an already published CVE that was found in the NVD. It would grab the description and then process it into our Machine Learning models to determine its technical impact on a system and whether it is able to be automatically exploited. It also will determine its exploit status based off the information we have, either KEV or CWEtoPOC, or Nothing, and we will let you know via the `exploitStatusRationale` response. Since we did not provide the mission and wellbeing you will notice that there is 3 scores  that correlate to the level of impact an exploit has on your company or system. 

**Example 2: CVE-ID and Exploit Status given due to collection:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381&exploitStatus=POC`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "POC",
      "exploitStatusRationale": "Exploit Collection",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: This response is the same as the previous response with two main things that are different. The `exploitStatus` is updated to POC and the `exploitStatusRationale` is updated to "Exploit Collection". Exploit Collection is the value given when you do your own research of the cve and have some sort of proof of concept exploitation of the vulnerability and let the api know that.

**Example 3: CVE-ID and Mission and Well Being:**
- Example Request: `GET /ssvc?cveId=CVE-2022-3381&mission=MINIMAL&wellbeing=MINIMAL`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2022-3381",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "missionAndWellbeing": "LOW",
      "ssvcScore": "TRACK",
      "technicalImpact": "partial"
    }  
  ```

- Response Explanation: This response is the same as the first response, however there are less `ssvcScore` values and the inclusion of the severity of impact on `missionAndWellbeing` which is returned in the response. The ssvc score value returned is the specific value for this provided instance.

**Example 4: Private/Undisclosed CVE-ID and Provided Description:**
- Example Request: `GET /ssvc?cveId=CVE-2150-1000&description=There is an internal module that can be exploited to cause reomte code execution within the kernel.`
- Example Response:
  ```json
    {
      "automatable": true,
      "cveId": "CVE-2150-1000",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found",
      "ssvcScoreHigh": "ATTEND",
      "ssvcScoreLow": "TRACK",
      "ssvcScoreMedium": "TRACK",
      "technicalImpact": "total"
    } 
  ```

- Response Explanation: This response is the same as the previous ones, but the main thing this response shows is, SSVC automatic scoring can be used for internal and hypothetical situations for you and your team, you don't have to wait for NVD disclosures if you need to know.


**Example 4: Invalid Response (Improper CVE / Rate Limit)**
- Example Request: `GET /ssvc?cveId=CVE-2100-3381`
- Example Response:
  ```json
    {
      "cveId": "CVE-2100-3381",
      "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
    }
  ```

- Response Explanation: This is the response you will get if something goes wrong, either you requested an invalid CVE or you're requesting too fast (NVD Description Pulling Limitations)

#### NOTE
You can use any and all combinations of the parameters above in your requests. The above examples are showing common examples of what could be requests made to the api.

### Technical Impact

- Endpoint: `/technicalimpact`
- Description: Retrieve the technical impact of a CVE based on provided public CVE-ID or provided CVE-ID and description.

**Required Parameter**:
- `cveId`: id of the CVE you are classifying

- **Optional Parameter**:
- `description`: description of cve you are determining. Good to supply if using non-published CVEs along with avoiding NVD rate limiting.

- Example Request: `GET /technicalimpact?cveId=CVE-2022-3281`
- Example Response:
  ```json
    {
      "cveId": "CVE-2022-3281",
      "technicalImpact": "partial"
    }
  ```

- Response Explanation: The response is very simple and do the point, it allows for you to determine the impact on the system due to machine learning processes. If you would like to know what the results truly mean please read [here](https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide%20508c.pdf).

### Automatability

- Endpoint: `/automatability`
- Description: Determine if the vulnerability you are scoring can be automatically exploited.

**Required Parameter**:
- `cveId`: id of the CVE you are classifying

- **Optional Parameter**:
- `description`: description of cve you are determining. Good to supply if using non-published CVEs along with avoiding NVD rate limiting.
- Example Request: `GET /automatability?cveId=CVE-2022-3281`
- Example Response:
  ```json
    {
      "automatable": false,
      "cveId": "CVE-2022-3281"
    }
  ```

- Response Explanation: The response is very simple and lets you know if the machine learning models that the cve's description ran against, thinks that this cve is able to be automatically exploited.

### Exploit Status

- Endpoint: `/exploitstatus`
- Description: Determine the exploit status of a CVE.

**Required Parameter**:
- `cveId`: id of the CVE you are determining exploit status of.

- Example Request: `GET /exploitstatus?cveId=CVE-2022-3281`
- Example Response:
  ```json
    {
      "cveId": "CVE-2022-3281",
      "exploitStatus": "NONE",
      "exploitStatusRationale": "Nothing Found"
    }
  ```

- Response Explanation: This response will give you the `exploitStatus` of the CVE bsaed off the public information available and the research that has went into automatic SSVC scoring. It also comes with a rationale of why it determined the value it did. 
- The `exploitStatusRationale` can be 1 of 3 values in this endpoint.
- `In KEV`: The CVE is found in the NVD's known exploited vulnerabilities database.
- `CWEtoPOC`: The CVEs mapped POCs from NVD are within our research and mapped information relating to POCs found with CWEs relating to them.
- `Nothing Found`: nothing was found automatically. This does **not** mean that there aren't any exploits. It just means there are none that are publicly accessible.

### Mission and Well Being

- Endpoint: `/wellbeing`
- Description: Determine severity of impact that a vulnerability will have on mission and wellbeing based on your own provided values. These are self-determined values and can be found [here](https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide%20508c.pdf)

**Required Parameters**:
- `mission`: Impact on a company's mission the vulnerability has.
- `wellbeing`: Impact on the well-being of others the vulnerability has.

  - Example Request: `GET /wellbeing?mission=SUPPORT&wellbeing=MINIMAL`
- Example Response:
  ```json
  {
    "missionAndWellbeing": "MEDIUM"
  }
  ```

- Response Explanation: This response is very short, and it correlates to the value that is needed to compute an SSVC score, and is the value that you could see in the ssvc related endpoints when mission and wellbeing was not provided, it has an impact on the score, so its good to know where vulnerabilites impact your scores.

## Usage

To use the API, make HTTP requests to the desired endpoints. You can use tools like `curl`, `Postman`, or any programming language's HTTP client libraries to interact with the API.

## Error Handling

The API provides appropriate error responses for invalid requests. You will receive a JSON response with an error message in case of errors.

### Machine Learning / Tensorflow Errors:
Dealing with Machine Learning models and TensorFlow specifically can prove to be challenging, and at some points there can be sporadic issues where Tensorflow cannot access the temporary files it creates. If this happens, and there are internal server errors, a lot of times you can clear the temporary file directory on your machine, and specifically TensorFlow's, and that tends to fix most of the issues that are created. 

## Conclusion

