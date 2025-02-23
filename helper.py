import os
import numpy as np
import soundfile as sf
import heapq
import bitarray as bitarray
import keyboard
import pickle 

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
    def __init__(self, database_dir: str):
        """
        Processes .wav files from the database directory and stores them as dictionnary of numpy arrays.
        
        Parameters:
            database_dir (str): Path to the database directory.
            output_filename (str): Name of the output .npz file.
        """
        self.dictOfArrays = {}
        self.arrayCount = 0
        
        # Get all subdirectories (F1, F2, ..., M1, M2, ...)
        self.subdirs = [d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d))]
        
        for subdir in self.subdirs:
            subdirPath = os.path.join(database_dir, subdir)
            wav_files = [f for f in os.listdir(subdirPath) if f.endswith(".wav")]
            
            subdirData = {}
            for wav_file in wav_files:
                self.arrayCount +=1
                file_path = os.path.join(subdirPath, wav_file)
                audio_data, samplerate = sf.read(file_path)  # Read .wav file as numpy array

                key = wav_file.split(".")[0]
                subdirData[key] = (np.array(audio_data, dtype=object), samplerate)
            
            self.dictOfArrays[subdir] = subdirData  # Store as an object array to handle varying lengths 
        print(f"Database saved as class dictionnary")
    
    def getListOfArrays(self):
        """
        Uses the preexisting dictionnary of arrays to construct a list of arrays for easier computation
        """
        listOfArrays = []
        for key, value in self.dictOfArrays.items():
            for subKey, tuple in value.items():
                audioData, _ = tuple
                listOfArrays.append(audioData)
        return listOfArrays
    
    def uniformQuantizer(self, NOfQuanta: int, keySubKeyList: list):
        """
        Performs uniform quantization on all files but updates the usage count 
        only for the specified key-subKey pairs.

        Parameters:
            NOfQuanta (int): Number of quantization levels.
            keySubKeyList (list): List of (key, subKey) tuples for which to update usage count.
                                If empty, update usage count for all files.

        Returns:
            dictOfQuantData (dict): Dictionary containing quantized audio data.
            referenceArray (np.array): Array of quantized reference values.
            usageCount (dict): Dictionary mapping quantized values to their occurrence count 
                            (only for specified key-subKey pairs).
        """
        listOfArrays = self.getListOfArrays()
        max_value = max([np.max(arr) for arr in listOfArrays])
        min_value = min([np.min(arr) for arr in listOfArrays])

        self.dictOfQuantData = {}
        self.referenceArray = np.linspace(min_value, max_value, NOfQuanta)

        # Initialize a counter for reference values
        self.usageCount = {ref_val: 0 for ref_val in self.referenceArray}

        # Create dictionary for the quantized data (process all files)
        for key, value in self.dictOfArrays.items():
            self.dictOfQuantData[key] = {}
            for subKey, (audioData, samplerate) in value.items():
                # Quantize the audio data
                quantizedArr = closestValues(audioData, self.referenceArray)
                self.dictOfQuantData[key][subKey] = (quantizedArr, samplerate)

                # Update usage count only if the file is in keySubKeyList or if the list is empty
                if not keySubKeyList or (key, subKey) in keySubKeyList:
                    for val in quantizedArr:
                        self.usageCount[val] += 1

        print("Quantization performed successfully")
        return self.dictOfQuantData, self.referenceArray, self.usageCount


"""def uniformQuantizer(listOfArrays, N):
    max_value = max([np.max(arr) for arr in listOfArrays])
    min_value = min([np.min(arr) for arr in listOfArrays])

    quantizedData = []
    referenceArray = np.linspace(min_value, max_value, N)

    # Initialize a counter for reference values
    usageCount = {ref_val: 0 for ref_val in referenceArray}
    for arr in listOfArrays:
        quantizedArr = closestValues(arr, referenceArray)
        quantizedData.append(quantizedArr)
        # Count the occurrences of each reference value
        for val in quantizedArr:
            usageCount[val] += 1
    print("Quantization performed successfully")
    return quantizedData, referenceArray, usageCount"""

def closestValues(arr, reference_arr):
    # Find the closest value for each element in arr 
    return np.array([reference_arr[np.argmin(np.abs(reference_arr - x))] for x in arr])

def saveDataDictToWav(dataDict: dict, outputDir: str):
    """
    Takes the dictionnary of numpy arrays and saves them to .wav in respective directories
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
            print(f"Saved {key}_{subKey}.wav with sample rate {samplerate}")
    return 0


def buidHuffmanTree(UsageCount):
    """
    Builds a Huffman Tree based on usage count dictionary.
    
    Parameters:
        usage_count (dict): A dictionary where keys are values and values are their frequencies.

    Returns:
        HuffmanNode: The root node of the Huffman Tree.
    """
    # Step 1: Create priority queue (min-heap)
    heap = [HuffmanNode(value, freq) for value, freq in UsageCount.items()]
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
        huffman_table (dict): Dictionary to store Huffman codes.

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

def huffmanEncoder(dicOfQuantData: dict, usageCount: dict, outputDir: str):
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
            code = "".join(huffmanTable[x] for x in audioData)
            
            # encode to efficient bitarray
            bitData = bitarray.bitarray(sampleRateBits + code)

            # Define output filename
            filename = os.path.join(outputDir, f"{key}_{subKey}.bin")

            # Store bit length to avoid extra bits when reading back
            bit_length = len(sampleRateBits + code)

            # Write to file
            with open(filename, "wb") as f:
                f.write(bit_length.to_bytes(4, 'big'))  # Store bit length as 2 bytes
                bitData.tofile(f)
                print(f"Saved compressed code to {key}_{subKey}.bin")

    # save the usage count for prior use
    filePath = os.path.join(outputDir, "usageCount.pkl")
    with open(filePath, 'wb') as f:
        pickle.dump(usageCount, f) 
        f.close()
        print("The Usage Count was successfully saved to usageCount.pkl")



def decodeStringToArray(root: HuffmanNode, code: str):
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

    # Read Usage Count
    usageCountFilePath = os.path.join(inputDir,"usageCount.pkl")
    with open(usageCountFilePath, 'rb') as f :
        usageCount = pickle.load(f)
    
    # Build huffman tree from usage count
    root = buidHuffmanTree(usageCount)

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
        print(f"Decoding string to array : {key}_{subKey}")
        decodedDict[key][subKey] = (decodeStringToArray(root, bitData[16:]), sampleRate)
    print(".bin files succesfully decoded")
    return decodedDict

# pause() function definition.
def pause():
    while True:
        if keyboard.read_key() == 'space':
            # If you put 'space' key
            # the program will resume.
            break

def getSize(startPath : str):
    totalSize = 0
    for dirpath, _, filenames in os.walk(startPath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp): # skip if link
                totalSize += os.path.getsize(fp)

    return totalSize

def getRMSE(originalDict: dict, decodedDict: dict):
    RMSEDict = {}
    for key, val in originalDict.items():
        RMSEDict[key] = {}
        for subKey, tuple in val.items():
            ogAudioData, _ = tuple
            try:
                decodedAudioData, _ = decodedDict[key][subKey]
                if ogAudioData.size == decodedAudioData.size:
                    rmse = np.sqrt(((decodedAudioData - ogAudioData) ** 2).mean())
                    print(f"The RMSE of {key}_{subKey} is : {rmse}")
                    RMSEDict[key][subKey] = rmse
                else :
                    print(f"Error : the audio data of {key}_{subKey} does not have the same length")
            except:
                print("Error : Fetching audio data from decoded dictionnary was not successful")
    return RMSEDict




