import os
import re
from struct import unpack
import numpy as np

def read_filenames(n, d) -> list:
    file_names = os.listdir(d)
    first_n_files = file_names[:n]
    return first_n_files

def get_date(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_", filename)
    return match.group(1) if match else None

def get_time(filename):
    match = re.search(r"_(\d{2}-\d{2}-\d{2})__", filename)
    return match.group(1) if match else None

def get_message(filename):
    match = re.search(r"_m=([a-f0-9]+)_", filename)
    return match.group(1) if match else None

def get_cipher(filename):
    match = re.search(r"_c=([a-f0-9]+)\.", filename)
    return match.group(1) if match else None

def get_key(filename):
    match = re.search(r"_k=([a-f0-9]+)_", filename)
    return match.group(1) if match else None

def hex_to_byte_array(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(0, 16, 2)]

def byte_array_to_hex(byte_array):
    return ''.join([format(b, '02x') for b in byte_array])

def get_data(n, d):
    traces_filenames = read_filenames(n, d)
    date = []
    time = []
    messages = []
    ciphertexts = []
    keys = []
    for filename in traces_filenames:
        d = get_date(filename)
        t = get_time(filename)
        m = hex_to_byte_array(get_message(filename))
        c = hex_to_byte_array(get_cipher(filename))
        k = hex_to_byte_array(get_key(filename))
        date.append(d)
        time.append(t)
        messages.append(m)
        ciphertexts.append(c)
        keys.append(k)
    return traces_filenames, date, time, np.array(messages), np.array(ciphertexts), keys


def parse_binary(raw_data: bytes) -> tuple:
	"""
	Takes a raw binary string containing data from our oscilloscope.
	Returns the corresponding float vector.
	"""
	ins =  4   # Size of int stored
	cur =  0   # Cursor walking in the string and getting data
	cur += 12  # Skipping the global raw binary string header
	whs =  unpack("i", raw_data[cur:cur+ins])[0] # Storing size of the waveform header
	cur += whs # Skipping the waveform header
	dhs =  unpack("i", raw_data[cur:cur+ins])[0] # Storing size of the data header
	cur += dhs # Skipping the data header
	bfs =  unpack("i", raw_data[cur-ins:cur])[0] # Storing the data size
	sc  =  bfs//ins # Samples Count - How much samples compose the wave
	dat =  unpack("f"*sc, raw_data[cur:cur+bfs])
	return dat


def get_traces(traces_filenames):
    traces = []
    for filename in traces_filenames:
        with open(f'secmatv1_2006_04_0809\\{filename}', 'rb') as f:
            raw_data = f.read()
        trace = list(parse_binary(raw_data))
        traces.append(trace)
    return np.array(traces)
