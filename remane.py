import os


for file in os.listdir("/Users/chen/Desktop/github/TEM_emittance/enittance_1_06262020/"):
    if file.endswith(".raw"):
        os.rename("/Users/chen/Desktop/github/TEM_emittance/enittance_1_06262020/" + file, "/Users/chen/Desktop/github/TEM_emittance/enittance_1_06262020/" + '0' + file)