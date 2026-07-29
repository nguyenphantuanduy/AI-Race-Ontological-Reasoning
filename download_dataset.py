import requests
import zipfile
import os


FILE_ID = "1-tPzZjZsOuVwjln7cADAWG1PxorO8jo3"

url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"

zip_path = "dataset.zip"

r = requests.get(url)

print(r.status_code)
print(r.headers.get("content-type"))

with open(zip_path, "wb") as f:
    f.write(r.content)


with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("dataset")

print("Done")