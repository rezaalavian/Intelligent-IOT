import requests
from pathlib import Path
import sys

schema_file = Path("infrastructure/kafka/topics/pm25.avsc")
if not schema_file.exists():
    print("Schema file not found", schema_file)
    sys.exit(2)

schema = schema_file.read_text()
url = "http://127.0.0.1:8081/subjects/pm25-value/versions"
headers = {"Content-Type": "application/vnd.schemaregistry.v1+json", "Accept": "application/vnd.schemaregistry.v1+json"}
resp = requests.post(url, json={"schema": schema}, headers=headers)
print(resp.status_code)
print(resp.text)
resp.raise_for_status()
