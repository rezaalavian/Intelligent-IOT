import urllib.request
import sys

url = "http://127.0.0.1:8081/subjects"
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print(r.status)
        print(r.read().decode())
except Exception as e:
    print("ERR", e)
    sys.exit(2)
