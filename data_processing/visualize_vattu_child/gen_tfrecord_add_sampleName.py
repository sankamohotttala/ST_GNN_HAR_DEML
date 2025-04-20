#here we attempt to add the sample name to the tfrecord file
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(features, label,name):
    feature = {
        'features' : _bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
        'label'     : _int64_feature(label),
        'name'     : _bytes_feature(bytes(name, 'utf-8'))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def gen_tfrecord_data(num_shards, label_path, data_path, dest_folder, shuffle):
    label_path = Path(label_path)
    #wtf = label_path.exists()
    if not (label_path.exists()):
        print('Label file does not exist')
        return

    data_path = Path(data_path)
    if not (data_path.exists()):
        print('Data file does not exist')
        return

    try:
        with open(label_path) as f:
            names, labels = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            names, labels = pickle.load(f, encoding='latin1')

    # Datashape: Total_samples, 3, 300, 25, 2
    data   = np.load(data_path, mmap_mode='r')
    labels = np.array(labels)
    names = np.array(names)

    if len(labels) != len(data):
        print("Data and label lengths didn't match!")
        print("Data size: {} | Label Size: {}".format(data.shape, labels.shape))
        return -1

    print("Data shape:", data.shape)
    if shuffle:
        p = np.random.permutation(len(labels))
        labels = labels[p]
        data = data[p]
        names = names[p]

    dest_folder = Path(dest_folder)
    if not (dest_folder.exists()):
        os.mkdir(dest_folder)

    step = len(labels)//num_shards
    for shard in tqdm(range(num_shards)):
        tfrecord_data_path = os.path.join(dest_folder, data_path.name.split(".")[0]+"-"+str(shard)+".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
            for i in range(shard*step, (shard*step)+step if shard < num_shards-1 else len(labels)):
                writer.write(serialize_example(data[i], labels[i], names[i]))

if __name__ == '__main__':

    #output_folder=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\processed\xsub'
    
    #output_folder=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\processed_new\xsub'
    #output_folder=r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_2D\xsub'
    
    # output_folder=r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_Comparison\xsub'
    output_folder=r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\shared\xsub'
    
    # above is also the input folder


    trainOrVal='train' # either 'train' or 'val'

    if trainOrVal=='train':
        lable_path=output_folder+'/train_label.pkl'
        data_path=output_folder+'/train_data_joint.npy'
        dest_path= output_folder+'/train_data/' 
    else:
        lable_path=output_folder+'/val_label.pkl'
        data_path=output_folder+'/val_data_joint.npy'
        dest_path= output_folder+'/val_data/' 

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data TFRecord Converter')
    parser.add_argument('--num-shards',
                        type=int,
                        default=40,
                        help='number of files to split dataset into')
    # parser.add_argument('--label-path',
    #                     required=False,default=output_folder+'/val_label.pkl',
    #                     help='path to pkl file with labels')
    parser.add_argument('--shuffle',
                        required=False,default=True,
                        help='setting it to True will shuffle the labels and data together')
    # parser.add_argument('--data-path',
    #                     required=False,default=output_folder+'/val_data_joint.npy',
    #                     help='path to npy file with data')
    # parser.add_argument('--dest-folder',
    #                     required=False,default=output_folder+'/val_data/',
    #                     help='path to folder in which tfrecords will be stored')
    arg = parser.parse_args()

    gen_tfrecord_data(arg.num_shards,
                      lable_path,
                      data_path,
                      dest_path,
                      arg.shuffle)
