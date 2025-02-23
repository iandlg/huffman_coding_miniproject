import numpy as np
import matplotlib.pyplot as plt
import helper as helper
import sys
import os

sys.setrecursionlimit(2000)

# set file names
databaseDirName = "database"
encodedDirName = "encoded_files"
decodedDirName = "decoded_wavs"

myData = helper.Data(databaseDirName)
dictQuant, refArr, usageCount = myData.uniformQuantizer(2*10**3,[("F1", "SA1")])

# save all audio to bin with their respective code and sample rate
helper.huffmanEncoder(dictQuant,usageCount,encodedDirName)

try:
    helper.pause()
except ImportError as e:
    print(f"Error: {e}")

# decode the bin files
decodedDict = helper.huffmanDecoder(encodedDirName)

# save numpy arrays to .wav files for verification
helper.saveDataDictToWav(decodedDict, decodedDirName)

# compression ratio
uncompressedDirSize =  helper.getSize(databaseDirName)
compressedDirSize = helper.getSize(encodedDirName)
print(f"The compression ratio is {compressedDirSize/uncompressedDirSize}")

# Evaluate reconstructed audio quality
RMSEDict = helper.getRMSE(myData.dictOfArrays, decodedDict)
