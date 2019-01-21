import numpy as np
import random, os
import torch, warnings, glob
import multiprocessing
from pathos.multiprocessing import _ProcessPool as Pool
import tqdm, math, time
from sklearn.externals import joblib
from torch.utils.data import Dataset

class LoadedBballData(Dataset):

    def __init__(self, data_path, target):
        self.meta_info = joblib.load(os.path.join(data_path, 'meta.pkl'))
        self.data_path = data_path
        self.target = target

    def __len__(self):
        return self.meta_info['len']

    def __getitem__(self, idx):
        file_index = math.floor(idx / self.meta_info['traj/file'])
        within_file_index = idx % self.meta_info['traj/file']
        xs, ys, ps = joblib.load(os.path.join(self.data_path, '%d.pkl' % file_index))

        if self.target == 'expected':
            return xs[within_file_index], ys[within_file_index]
        elif self.target == 'points':
            return xs[within_file_index], ps[within_file_index]
        else:
            raise ValueError('Target has to be either expected or points!')

def save_bball_data_helper_helper(dataset, indices, savename):
    # get dataset saved
    xs = []
    ys = []
    ps = []
    for i in indices:
        x, y, p = dataset[i]
        xs.append(x)
        ys.append(y)
        ps.append(p)
    joblib.dump((xs, ys, ps), savename)
    
def save_bball_data_helper(dataset, indices, savedir, base_index, traj_per_file):
    nchunks = math.ceil(len(indices) / traj_per_file)
    for i in tqdm.tqdm(range(nchunks)):
        sub_indices = indices[(i*traj_per_file): ((i+1)*traj_per_file)]
        savename = os.path.join(savedir, '%d.pkl' % (base_index + i))
        save_bball_data_helper_helper(dataset, sub_indices, savename)

def parallel_bball_data_helper(dataset, savedir, cpus=30, traj_per_file=10):
    # todo: deal with invalid data after data transformation
    result_list = []
    pool = Pool(cpus)

    tasks_per_cpu = max(math.ceil(len(dataset) / cpus), traj_per_file)
    # make tasks a multiple of traj_per_file
    tasks_per_cpu = math.ceil(tasks_per_cpu / traj_per_file) * traj_per_file

    # save meta information
    joblib.dump({'len': len(dataset),
                 'traj/file': traj_per_file},
                os.path.join(savedir, 'meta.pkl'))

    index = 0
    i = 0
    while index < len(dataset):
        indices = range(index, min(index + tasks_per_cpu, len(dataset)))
        index = index + tasks_per_cpu

        result_list.append(pool.apply_async(func=save_bball_data_helper,
                                            args=(dataset, indices,
                                                  savedir, i, traj_per_file)))
        i += int(tasks_per_cpu / traj_per_file)
        
    while True:
        try:
            def call_if_ready(result):
                if result.ready():
                    result.get()
                    return True
                else:
                    return False  
            done_list = list(map(call_if_ready, result_list))
            print('{}/{} done'.format(sum(done_list), len(result_list)))
            if np.all(done_list):
                break
            time.sleep(3)
        except:
            pool.terminate()
            raise
    print('finished preprocessing')

def save_bball_data(dataset, savedir, override_existing=False):

    print(savedir)
    file_exist = os.path.exists(savedir)
    if file_exist:
        warnings.warn("%s exist, override: %r" % (savedir, override_existing))

    if not file_exist or override_existing:
        print("==>save data of size %d in %s" % (len(dataset), savedir))
        if file_exist:
            os.system('rm -r %s' % savedir)
        os.system('mkdir -p %s' % savedir)

        # get dataset saved
        parallel_bball_data_helper(dataset, savedir)
        print("==>save data done")

def load_bball_data(load_path, target):
    print('==>load data %s' % load_path)
    dset = LoadedBballData(load_path, target)
    print('==>load data of size %d done' % len(dset))
    return dset
    
class BballDataset(Dataset):
    '''return one possession'''
    def __init__(self, traj_locations, transform):
        ''' 
        transform turns a possession into (x, y), see bball_transform/image_transform.py
        '''
        self.traj_locations = traj_locations
        self.transform = transform

    def __len__(self):
        return len(self.traj_locations)
        
    def __getitem__(self, idx):
        data = joblib.load(self.traj_locations[idx])
        return self.transform(data)

