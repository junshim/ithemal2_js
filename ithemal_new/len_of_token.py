import argparse
import torch

from ithemal_utils import *
from tqdm import tqdm

def main():
    # type: () -> None
    print("test")
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    args = parser.parse_args()

    data = load_data(args.data)
   
    dic = {}
    dic_instr = {}

    total_data = len(data.data)
    print("total data : {}".format(total_data))
    print(data.hot_idx_to_token)
    for i in range(total_data):
        datum = data.data[i]
        token_inputs = datum.x
        len_instr = 0
        for j in range(len(token_inputs)):
            len_instr +=1
            token_len = len(token_inputs[j])
            try: dic[token_len] += 1
            except: dic[token_len] = 1
        try: dic_instr[len_instr] += 1
        except: dic_instr[len_instr] = 1

    dic = sorted(dic.items())
    #for i in range(len(dic))
    print(dic)
    dic_instr = sorted(dic_instr.items())
    for j in range(len(dic_instr)):
        print("%d, %d"%(dic_instr[j][0],dic_instr[j][1]))

if __name__ == '__main__':
    main()
