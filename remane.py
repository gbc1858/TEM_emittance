import os


for file in os.listdir("/Users/chen/Desktop/github/TEM_emittance/emittance_2/"):
    if file.endswith(".raw"):
        os.rename("/Users/chen/Desktop/github/TEM_emittance/emittance_2/" + file, "/Users/chen/Desktop/github/TEM_emittance/emittance_2/" + '0' + file)