import requests
import time
import pandas as pd

"""
Script to utilize local deployment of api to score a csv file of sves
CSV format expected is:
CVEID,DESCRIPTION
"""
def main():
    data = input("enter csv file with cves to score: ")
    use_descriptions = input("Are descriptions provided (y/n): ")
    if use_descriptions.lower() == "y":
        use_descriptions = True
    else:
        use_descriptions = False

    df = pd.read_csv(data)


    scores = list()
    for index, row in df.iterrows():
        id = row["CVEID"]
        if use_descriptions:
            description = row["DESCRIPTION"]

        score_url = f"http://localhost:5000/ssvc?cveId={id}&description={description}" if use_descriptions else f"http://localhost:5000/ssvc?cveId={id}"
        req = requests.get(score_url)
        while not req.status_code == 200:
            # if you want to not deal with waiting loops, set sleep time to 30
            print(f"Got error code {req.status_code}, waiting 10s for potential NVD reset and trying again for {id}")
            time.sleep(10)
            req = requests.get(score_url)

        json = req.json()
        technical_impact = json["technicalImpact"]
        automatable = json["automatable"]
        exploit_status = json["exploitStatus"]
        exploit_status_rationale = json["exploitStatusRationale"]
        ssvc_high = json["ssvcScoreHigh"]
        ssvc_medium = json["ssvcScoreMedium"]
        ssvc_low = json["ssvcScoreLow"]
        print(f"Got scoring information for for {id}")
        scores.append((id, technical_impact, automatable, exploit_status, exploit_status_rationale, ssvc_high, ssvc_medium, ssvc_low))

    print("Exporting results")
    export = pd.DataFrame(columns=["CVE ID", "TECHNICAL IMPACT", "AUTOMATABLE", "EXPLOIT STATUS", "E.S. RATIONALE", "SSVC HIGH", "SSVC MED", "SSVC LOW"], data=scores)
    export.to_csv(data.replace(".csv", "_scored.csv"), index=False)


if __name__ == '__main__':
    main()
