import os
import tarfile
import urllib.request

# Create data directory
os.makedirs("data", exist_ok=True)

# URL and destination
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "data/aclImdb_v1.tar.gz"

# Download
if not os.path.exists(filename):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")

# Extract
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path="data")
    print("Extraction complete.")
