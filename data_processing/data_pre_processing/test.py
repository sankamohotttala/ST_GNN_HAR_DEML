import argparse
from pathlib import Path
data_paths='../data/ntu/xsub/train_data/'
data_path='sd.txt'
label_path = Path(data_paths)
wtf = label_path.exists()
print(wtf)