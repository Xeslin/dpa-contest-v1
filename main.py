import get_data as gd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Standard DES Initial Permutation (IP) table
IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

# Final permutation (IP^-1)
FP = [
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25,
    32, 0, 40, 8, 48, 16, 56, 24
]

# Expansion table (E)
E = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

# Permutation (P)
P = [
    15, 6, 19, 20, 28, 11, 27, 16,
    0, 14, 22, 25, 4, 17, 30, 9,
    1, 7, 23, 13, 31, 26, 2, 8,
    18, 12, 29, 5, 21, 10, 3, 24
]

# DES S-Boxes
SBOX = [
    # S1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # S2
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    # S3
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    # S4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    # S5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # S6
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    # S7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # S8
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

# Permuted choice 1 (PC-1)
PC1 = [
    56, 48, 40, 32, 24, 16, 8,
    0, 57, 49, 41, 33, 25, 17,
    9, 1, 58, 50, 42, 34, 26,
    18, 10, 2, 59, 51, 43, 35,
    62, 54, 46, 38, 30, 22, 14,
    6, 61, 53, 45, 37, 29, 21,
    13, 5, 60, 52, 44, 36, 28,
    20, 12, 4, 27, 19, 11, 3
]

# Permuted choice 2 (PC-2)
PC2 = [
    13, 16, 10, 23, 0, 4,
    2, 27, 14, 5, 20, 9,
    22, 18, 11, 3, 25, 7,
    15, 6, 26, 19, 12, 1,
    40, 51, 30, 36, 46, 54,
    29, 39, 50, 44, 32, 47,
    43, 48, 38, 55, 33, 52,
    45, 41, 49, 35, 28, 31
]


def trace_plot(trace):
    plt.figure(figsize=(12, 4))
    plt.plot(trace)
    plt.title("Осциллограмма")
    plt.ylabel("Напряжение (В)")
    plt.xlabel("Отсчеты")
    plt.grid(True)
    plt.show()

def plot_corr(traces, pcc):
    fig, ax1 = plt.subplots()

    ax1.plot(traces[1, :], 'b-')

    ax1.set_ylabel('Voltage', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    
    ax2 = ax1.twinx()
    ax2.plot(pcc, 'r-', linewidth=2)
    ax2.set_ylabel('Pearson Correlation Coefficient', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    len_trace = np.size(traces, 1)
    plt.xlim((0, len_trace))
    plt.show()

def calc_corr(messages, traces):

    model = np.unpackbits(messages[:, 7].astype('uint8')).reshape(-1, 8).sum(axis=1)

    model_centered = model - np.mean(model)
    traces_centered = traces - np.mean(traces, axis=0)

    numerator = np.dot(model_centered, traces_centered)
    denominator = np.linalg.norm(model_centered) * np.linalg.norm(traces_centered, axis=0)
    pcc = numerator / denominator

    plot_corr(traces, pcc)

def hamming_weight(x):
    return bin(x).count('1')

def permute(block, table):
    return [block[x] for x in table]

def des_sbox(input_val, sbox_idx):
    row = ((input_val & 0x20) >> 4) | (input_val & 0x01)
    col = (input_val & 0x1E) >> 1
    sbox = SBOX[sbox_idx]
    return sbox[row][col]

def cpa_attack(messages, traces, target_byte):
    best_guess = 0
    max_corr = 0
    correlations = np.zeros(256)
    n_traces = len(traces)
    trace_length = len(traces[0])
    
    # For each possible key byte value
    for key_guess in range(256):
        hws = np.zeros(n_traces)
        
        # Calculate hypothetical Hamming weights
        for i in range(n_traces):
            # HW of S-box output
            sbox_out = des_sbox(messages[i, target_byte] ^ key_guess, target_byte)
            hws[i] = hamming_weight(sbox_out)
        
        # Calculate correlation with each point in the trace
        trace_correlations = np.zeros(trace_length)
        for j in range(trace_length):
            r, _ = pearsonr(hws, traces[:, j])
            trace_correlations[j] = abs(r)
        plt.plot(trace_correlations)
        max_r = np.max(trace_correlations)
        correlations[key_guess] = max_r
        
        if max_r > max_corr:
            max_corr = max_r
            best_guess = key_guess
    plt.show()
    # Plot correlations for this byte
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(256), correlations)
    # plt.title(f"Correlations for target byte {target_byte}")
    # plt.xlabel("Key guess")
    # plt.ylabel("Maximum correlation")
    # plt.axvline(x=best_guess, color='r', linestyle='--', label=f'Best guess: {best_guess}')
    # plt.legend()
    # plt.show()
    
    return best_guess, max_corr


if __name__ == "__main__":
    directory = DIRECTORY_PATH
    n = 1000
    filenames, date, time, messages, ciphertexts, keys, traces = [], [], [], [], [], [], []
    filenames, date, time, messages, ciphertexts, keys = gd.get_data(n, directory)
    traces = gd.get_traces(filenames)

    #trace_plot(traces[-1])
    #calc_corr(messages, traces)

    traces_cutted = traces[:, 5000:7000]

    recovered_key = np.zeros(8, dtype=np.uint8)
    for byte_pos in range(8):
        best_guess, max_corr = cpa_attack(messages, traces_cutted, byte_pos)
        recovered_key[byte_pos] = best_guess
        print(f"Byte {byte_pos}: guessed {best_guess}, correlation {max_corr:.4f}")

    print(f"Recovered key: {gd.byte_array_to_hex(recovered_key)}")
    
