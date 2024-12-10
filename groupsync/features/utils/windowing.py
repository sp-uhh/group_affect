import pandas as pd


def split_dataframe(df, window_size): 
    # input - df: a Dataframe, chunkSize: the chunk size
    # output - a list of DataFrame
    # purpose - splits the DataFrame into smaller chunks
    chunks = list()
    # num_chunks = len(df) // window_size + 1
    num_chunks = len(df) // window_size + (1 if len(df) % window_size else 0)
    print(len(df))
    print(window_size)
    for i in range(num_chunks):
        chunk = df[i*window_size:(i+1)*window_size]
        print("chunk size = ", chunk.shape)
        chunks.append(chunk)
    return chunks