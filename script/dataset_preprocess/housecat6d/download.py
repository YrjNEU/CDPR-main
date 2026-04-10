import os
import requests

url = "http://www.campar.in.tum.de/public_datasets/2023_housecat6d_new/test_scene.zip"
output = "/home/neu/test_scene.zip"

pos = os.path.getsize(output) if os.path.exists(output) else 0
headers = {"Range": f"bytes={pos}-"}

with requests.get(url, headers=headers, stream=True) as r:
    r.raise_for_status()
    with open(output, "ab") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

print("Done")
