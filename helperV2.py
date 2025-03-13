import os
import numpy as np
import soundfile as sf
import heapq
import bitarray as bitarray
import keyboard
import pickle
import bisect
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value  # The quantized value
        self.freq = freq    # The frequency of occurrence
        self.left = None    # Left child (0)
        self.right = None   # Right child (1)

    # Compare nodes based on frequency (for priority queue)
    def __lt__(self, other):
        return self.freq < other.freq

class Data:
    def __init__(self, database_dir: str,  keySubKeyList: list):
        """
        Processes .wav files from the database directory and stores them as dictionnary of numpy arrays.
        Also saves the sizes of each file in a dictionnary for later metrics.
        
        Parameters:
            database_dir (str): Path to the database directory.
            keySubKeyList (list): List of audio files used in creating the usage count.
        """
        self.dictOfArrays = {}
        self.arrayCount = 0
        self.dictOfFileSizes = {}
        
        # Get all subdirectories (F1, F2, ..., M1, M2, ...)
        self.subdirs = [d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d))]
        
        for subdir in self.subdirs:
            subdirPath = os.path.join(database_dir, subdir)
            wav_files = [f for f in os.listdir(subdirPath) if f.endswith(".wav")]
            
            subdirData = {}
            subdirSizes = {}
            for wav_file in wav_files:
                self.arrayCount +=1
                file_path = os.path.join(subdirPath, wav_file)
                audio_data, samplerate = sf.read(file_path)  # Read .wav file as numpy array
                fileSize = os.path.getsize(file_path)
                key = wav_file.split(".")[0]
                subdirData[key] = (np.array(audio_data, dtype=np.float64), samplerate)
                subdirSizes[key] = fileSize
            
            self.dictOfArrays[subdir] = subdirData
            self.dictOfFileSizes[subdir] = subdirSizes
        print(f"Database saved as class dictionnary")
        self.usageCount = self.getTimeDomUsageCount(keySubKeyList)
        self.usageFreqPhaseCount = self.getFrequencyDomUsageCount(keySubKeyList)
        print("Usage counts found")
    
    def getListOfArrays(self):
        """
        Uses the preexisting dictionnary of arrays to construct a list of arrays for easier computation
        Parameters :
            self (Data) : custom Data class
        """
        listOfArrays = []
        for key, value in self.dictOfArrays.items():
            for subKey, tuple in value.items():
                audioData, _ = tuple
                listOfArrays.append(audioData)
        return listOfArrays
    
    def getTimeDomUsageCount(self, keySubKeyList: list):
        """
        Counts the number of occurences of each quantized values in the audio files specified to explore.
        Parameters :
            self (Data) : custom Data class
            keySubKeyList (list): List of audio files used in creating the usage count.
        """
        usageCount = {}
        count = 0
        for key, value in self.dictOfArrays.items():
            for subKey, (audioData, samplerate) in value.items():
                # Update usage count only if the file is in keySubKeyList or if the list is empty
                if not keySubKeyList or (key, subKey) in keySubKeyList:
                    for val in audioData:
                        usageCount[val] = usageCount.get(val,0) + 1
                        count += 1
        print(f"There are {count} elements in the time domain")
        sortedUsageCount = {}
        for key in sorted(usageCount.keys(), key=float):
            sortedUsageCount[key] = usageCount[key]
        return sortedUsageCount
    
    def getFrequencyDomUsageCount(self, keySubKeyList: list):
        """
        Counts the occurrences of (magnitude, phase) pairs in the DFT of each audio file.
            self (Data) : custom Data class
            keySubKeyList (list): List of audio files used in creating the usage count.
        """
        usageFreqPhaseCount = {}  # Dictionary to store (magnitude, phase) counts
        count = 0
        for key, value in self.dictOfArrays.items():
            for subKey, (audioData, samplerate) in value.items():
                # Process only specified files or all if keySubKeyList is empty
                if not keySubKeyList or (key, subKey) in keySubKeyList:
                    X_k = np.fft.fft(audioData)  # Compute DFT
                    magnitudes = np.abs(X_k).tolist() # Get magnitude
                    phases = np.angle(X_k).tolist() # Get phase
                    for mag, phase in zip(magnitudes, phases):
                        key_pair = (mag, phase) # Round for better counting
                        usageFreqPhaseCount[key_pair] = usageFreqPhaseCount.get(key_pair, 0) + 1
                        count += 1
        print(f"There are {count} elements in the frequency domain")
        return usageFreqPhaseCount

##############################################################################################################"
# Encoding
##############################################################################################################"

def buidHuffmanTree(usageCount):
    """
    Builds a Huffman Tree based on usage count dictionary.
    
    Parameters:
        usageCount (dict): A dictionary where keys are values and values are their frequencies.

    Returns:
        HuffmanNode: The root node of the Huffman Tree.
    """
    # Step 1: Create priority queue (min-heap)
    heap = [HuffmanNode(value, freq) for value, freq in usageCount.items()]
    heapq.heapify(heap)  # Convert to priority queue

    # Step 2: Construct the Huffman Tree
    while len(heap) > 1:
        # Extract two nodes with the smallest frequencies
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Create a new node with combined frequency
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        # Push the merged node back into the heap
        heapq.heappush(heap, merged)

    return heap[0]  # The root of the Huffman Tree

def generateHuffmanTable(root, prefix="", huffmanTable={}):
    """
    Generates a Huffman encoding table from a Huffman Tree.

    Parameters:
        root (HuffmanNode): The root of the Huffman Tree.
        prefix (str): The current binary prefix (used in recursion).
        huffmanTable (dict): Dictionary to store Huffman codes.

    Returns:
        dict: A dictionary mapping values to Huffman codes.
    """
    if root is None:
        return

    # If it's a leaf node, store the Huffman code
    if root.value is not None:
        huffmanTable[root.value] = prefix
        return huffmanTable

    # Recursively generate codes for left (0) and right (1) children
    generateHuffmanTable(root.left, prefix + "0", huffmanTable)
    generateHuffmanTable(root.right, prefix + "1", huffmanTable)
    return huffmanTable

def getHuffmanCode(audioData, huffmanTable):
    """
    Generates the huffman code of the given audio data.
    
    Parameters :
        audioData (np.array) : vector of real values
        huffmanTable (dict) : dictionnary mapping the values to the Huffman codes.
    
    Returns:
        (str) : string of bits that encode the audio data."""
    # Ensure we use the closest values from the Huffman table keys
    closeAudioData = closestValues(audioData, np.array(list(huffmanTable.keys())))

    # Generate Huffman code
    code = "".join(huffmanTable[x] for x in closeAudioData)
    return code

def closestValues(arr, reference_arr):
    """
    Find the closest value in reference_arr for each element in arr.

    Parameters:
        arr (np.array) : audio data
        reference_array (list) : list of values to which the arr has to be mapped
    
    Returns:
        (np.array) : list of values
    """
    closest_values = []
    for x in arr:
        closest = reference_arr[np.argmin(np.abs(reference_arr - x))]
        closest_values.append(closest)
    return np.array(closest_values)

def saveUsageCount(usageCount: dict, outputDir: str):
    """
    Save the usage count to a binary file for transmission and decoding.

    Parametrs:
        usageCount (dict): A dictionary where keys are values and values are their frequencies.
        outputDir (str): Directory where the file will be saved.
    """
    # Initialize bitarray
    bitData = bitarray.bitarray(endian='big')

    filePath = os.path.join(outputDir, "usageCount.b")
    
    with open(filePath, 'wb+') as f:
        for val, count in usageCount.items():
            if count != 0:  # Store only non-zero counts
                # Convert float (4bytes) and int16 (2bytes) to binary
                packed_data = struct.pack('fh', val, count)
                bitData.frombytes(packed_data)

        # Write the binary data to file
        bitData.tofile(f)


def huffmanEncoder(dicOfQuantData: dict, usageCount: dict, outputDir: str):
    """
    Function that successively calls the actions to encode the dictionnary of audio files to binary files in the specified directory.
    
    Parameters:
        usageCount (dict): A dictionary where keys are values and values are their frequencies.
        outputDir (str): Directory where the file will be saved.

    Returns:
        (dict) : dictionnary mapping the values to the Huffman codes. (used for debugging)
    """
    # build huffman tree
    root = buidHuffmanTree(usageCount)

    # Build huffman table
    huffmanTable = generateHuffmanTable(root)

    # ensure output directory exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # save the audio files to their compressed version
    for key, val in dicOfQuantData.items():
        for subKey, tuple in val.items():
            audioData, samplerate = tuple
            # get samplerate binary representation
            # Convert sample rate to a 16-bit binary string
            sampleRateBits = format(samplerate, '016b')

            # get code for array
            #print(f"Getting Huffman Code for {key}, {subKey}.")
            code = getHuffmanCode(audioData, huffmanTable)
            
            # encode to efficient bitarray
            bitData = bitarray.bitarray(sampleRateBits + code)

            # Define output filename
            filename = os.path.join(outputDir, f"{key}_{subKey}.bin")

            # Store bit length to avoid extra bits when reading back
            bit_length = len(sampleRateBits + code)

            # Write to file
            with open(filename, "wb+") as f:
                f.write(bit_length.to_bytes(4, 'big'))  # Store bit length as 2 bytes
                bitData.tofile(f)
    print(f"Files encoded to {outputDir}")
    saveUsageCount(usageCount,outputDir)  # Use the custom function
    print("The Usage Count was successfully saved to usageCount.b")
    return huffmanTable

##########################################################################
# Decode
##########################################################################
def loadUsageCount(inputDir):
    """
    Load symbol counts from binary format.

    Parameters:
        inputDir (str): Directory where the usageCount file is located.

    Returns:
        (dict): A dictionary where keys are values and values are their frequencies.
    """
    usageCountFilePath = os.path.join(inputDir,"usageCount.b")
    usageCount = {}
    with open(usageCountFilePath, 'rb') as f:
        packedData = f.read()    
        for i in range(0, len(packedData), 6):  # 4 byte float + 2 bytes int16
            symbol, count = struct.unpack("fh", packedData[i:i+6])
            usageCount[symbol] = count
    
    return usageCount

def decodeStringToArray(root: HuffmanNode, code: str):
    """
    Decodes the given string to an array using the Huffman Tree.
    
    Parameters:
        root (HuffmanNode): root to the Huffman Tree
        code (str): string of bits to be decoded
    
    Returns:
        (np.array): decoded array"""
    ans = []
    curr = root
    n = len(code)
    for i in range(n):
        if code[i]=='0':
            curr = curr.left
        else :
            curr = curr.right

        # reach leaf node
        if curr.left == None and curr.right == None :
            ans.append(curr.value)
            curr = root
    return np.array(ans)

def readEncodedFile(filename: str):
    """
    Reads a binary Huffman-encoded file and converts it to a binary string.
    
    Parameters:
        filename (str): Path to the encoded .bin file.
    
    Returns:
        str: The binary string representation of the Huffman-encoded data.
    """
    bitData = bitarray.bitarray()

    # Read binary data from file
    with open(filename, "rb") as f:
        bit_length = int.from_bytes(f.read(4), 'big')  # Read the stored bit length
        bitData.fromfile(f)  # Load the bitarray from file

    return bitData.to01()[:bit_length]  # Convert to a binary string


def huffmanDecoder(inputDir: str):
    """
    Function that calls the consecutive actions to perform the Huffman decoding.
    
    Parameters:
        inputDir (str): name of the directory where the binary files are saved.
    
    Returns:
        (dict): dictonnary where the audio arrays and sampling rates are saved.
        huffmanTable (dict) : dictionnary mapping the values to the Huffman codes.
    """

    # Read Usage Count:
    usageCount = loadUsageCount(inputDir) 
    
    # Build huffman tree from usage count
    root = buidHuffmanTree(usageCount)

    # Build huffman table
    huffmanTable = generateHuffmanTable(root)

    # Fetch all files to decode
    binFiles = [f for f in os.listdir(inputDir) if f.endswith(".bin")]

    decodedDict = {}
    for file in binFiles:
        filePath = os.path.join(inputDir,file)

        # Extract the string of bits
        bitData = readEncodedFile(filePath)

        # Extract the first 16 bits for the sample rate
        sampleRateBits = bitData[:16]
        sampleRate = int(sampleRateBits, 2)   # Convert binary string to integer

        # Extract the key and subkey to be used in the structure recontruction
        key = file.split("_")[0]  # Extract key (e.g., "F1" from "F1_0.bin")
        subKey = file.split('_')[1].split(".")[0]
        if key not in decodedDict:
            decodedDict[key] = {}  # Initialize list for the key
        #print(f"Decoding string to array : {key}_{subKey}")
        decodedDict[key][subKey] = (decodeStringToArray(root, bitData[16:]), sampleRate)
    print(f".bin files succesfully decoded.")
    return decodedDict, huffmanTable

#####################################################################################################################"
# Metrics
###################################################################################################################"
def getSize(startPath : str):
    """
    Computes the size of a given directory.
    
    Parameters:
        startPath (str): directory to be considered
    
    Returns:
        (float): Size of the directory
    """
    totalSize = 0
    for dirpath, _, filenames in os.walk(startPath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp): # skip if link
                totalSize += os.path.getsize(fp)

    return totalSize

def getRMSE(originalDict: dict, decodedDict: dict):
    """
    Computes the RMSE of each file in the original and decoded dictionnaries.
    
    Parameters:
        originalDict (dict): dictionnary of audio files from the database.
        decodedDict (dict): dictionnary of the decoded audio files.
    
    Returns:
        (dict): A dictionnary of the RMSE of each file pair
        (float): The average RMSE over all the files
    """
    RMSEDict = {}

    for key, val in originalDict.items():
        RMSEDict[key] = {}
        RMSEsum = 0
        count = 0

        for subKey, tuple in val.items():
            ogAudioData, _ = tuple
            try:
                decodedAudioData, _ = decodedDict[key][subKey]
                
                # Check if lengths match
                if ogAudioData.size != decodedAudioData.size:
                    print(f"Warning: The audio data of {key}_{subKey} do not have the same length. Trimming the longer one.")

                # Trim the longer array to match the shorter one's length
                min_length = min(ogAudioData.size, decodedAudioData.size)
                ogAudioData = ogAudioData[:min_length]
                decodedAudioData = decodedAudioData[:min_length]

                # Compute RMSE
                rmse = np.sqrt(((decodedAudioData - ogAudioData) ** 2).mean())
                print(f"The RMSE of {key}_{subKey} is : {rmse}")
                RMSEDict[key][subKey] = rmse
                RMSEsum += rmse
                count += 1
                    
            except KeyError:
                print(f"Error: Fetching audio data for {key}_{subKey} from decoded dictionary was not successful")
    
    return RMSEDict, RMSEsum/count

def getCompression(fileSizeDict: dict, encodedDirName : str):
    """
    Computes the compression ratio for per file (ignoring the usage count file)
    
    Parameters:
        fileSizeDict (dict): dictionnary containing the file sizes of the original .wav files.
        encodedDirName (str): directory containing the encoded binary files.
    
    Returns:
        (np.array): list of compression ratios for each file.
        (float): average compression ratio.
        (float): compression ratio variance.
    """
    compRatios = []
    for key, val in fileSizeDict.items():
        for subKey, ogfileSize in val.items():
            fileName = key + "_" + subKey + ".bin"
            filePath = os.path.join(encodedDirName,fileName)
            encodedFileSize = os.path.getsize(filePath)
            ratio = encodedFileSize/fileSizeDict[key][subKey]
            compRatios.append(ratio)
    compRatios = np.array(compRatios)
    return compRatios, np.mean(compRatios), np.std(compRatios)

def saveDataDictToWav(dataDict: dict, outputDir: str):
    """
    Takes the dictionnary of numpy arrays and saves them to .wav in respective directories

    Parameters:
        dataDict (dict): dictionnary containing the audio arrays and their respective sampling frequencies.
        outputDir (str): name of the directory to save the .wav files to.
    """
    # ensure output directory exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for key, val in dataDict.items():
        for subKey, tuple in val.items():
            audioData, samplerate = tuple
            filePath = os.path.join(outputDir, f"{key}_{subKey}.wav")

            # write wav file
            sf.write(filePath, audioData, samplerate)
            # print(f"Saved {key}_{subKey}.wav with sample rate {samplerate}")
    print(f"The audio files were saved to {outputDir}")