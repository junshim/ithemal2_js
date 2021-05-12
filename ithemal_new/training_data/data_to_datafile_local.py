import subprocess
from tqdm import tqdm
import torch

# read code_raw and timing from data.csv
# save it to data list of (code_id, code_raw, timing)
#data_path = "./pp_data1213.csv"
data_path = "./benchmark/throughput/skl.csv"
fr = open(data_path)

i = 1
data = list()
raw_data = list()
oneline = fr.readline()

j=0
while True:
        temp = list()
        for value in oneline.split(','):
                temp.append(value)
        if not temp[0]:
            oneline = fr.readline()
            if not oneline: break
            else : continue
        if float(temp[1].rstrip()) > 10000.:
            j+=1
            oneline = fr.readline()
            if not oneline: break
            else : continue
        item = list()
        item.append(i)
        item.append(float(temp[1].rstrip()))
        raw_data.append(temp[0])
        data.append(item)
        oneline = fr.readline()
        if not oneline: break
        if i>30000:break
        i+=1;

print(j)
print(len(data))
fr.close()

# do tokenizer for code_raw
# save it to data list of (code_id, timing, code_intel,code_xml)
for iterator in tqdm(range(len(data))):
        code_raw = raw_data[iterator]
        intel = subprocess.Popen(['./tokenizer',code_raw,'--intel'],stdout=subprocess.PIPE)
        out,err = intel.communicate()
        data[iterator].append((out.rstrip()).decode())

        tokenizer = subprocess.Popen(['./tokenizer',code_raw,'--token'],stdout=subprocess.PIPE)
        out,err = tokenizer.communicate()
        data[iterator].append((out.rstrip()).decode())

#torch.save(data,'./test_sample_data.data')
torch.save(data,'./skl_without_over10000.data')
#torch.save(data,'./pp_learning1213.data')

#for iterator in range(len(data)):
#    print(data[iterator][0])
#    print(data[iterator][1])
#    print(data[iterator][2])
#    print(data[iterator][3])
