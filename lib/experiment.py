import torch.nn as nn
import math, torch, os
from lib.train import TrainFeedForward
from lib.model import MODELS
from torch.utils.data import DataLoader
from lib.data import BballDataset, save_bball_data, load_bball_data
from lib.utils import get_traj_locations, shot_only_criterion, shot_length_criterion
from lib.bball_transform.image_transform import transform_producer
from lib.bball_transform.ts_transform import transform_producer as transform_producer_ts
from lib.bball_transform.flat_transform import transform_producer as transform_producer_flat

# exact setting for each experiment ran
SYNTHETIC_EXPERIMENTS = {}

class ImageExperiment(object):

    def get_data(self, args, path):
        raise NotImplementedError()
        
    def run(self, args, train_data, val_data):
        if args.arch == 'MLP':
            net = MODELS[args.arch]([args.input_dim, 30, 1])
            savename = 'mlp.pth.tar'

        elif args.arch == 'LSTM':
            net = MODELS[args.arch](22, args.hidden_size,
                                     args.num_layers, args.dropout, args.use_gpu)
            savename = 'lstm.pth.tar'
        elif args.arch == 'CNN':
            net = MODELS[args.arch]()
            savename = 'cnn.pth.tar'
            
        savename = os.path.join(args.smdir, savename)
        optimizer = torch.optim.Adam(net.parameters())
        criterion = nn.MSELoss()

        if args.arch in ('MLP', 'CNN'):
            trainer = TrainFeedForward(net, train_data,
                                       optimizer=optimizer, criterion=criterion, 
                                       save_filename=savename, val_data=val_data,
                                       use_gpu=args.use_gpu, n_iters=args.niters,
                                       n_save=args.n_save_model,
                                       batch_size=args.batch_size)


        if os.path.exists(savename) and not args.override_model:
            trainer.load_checkpoint(savename)
        trainer.train()

    def get_train_val_test(self, args):

        if args.debug:
            train_path = os.environ['BBALL2018_DEBUG_TRAIN']
            val_path = os.environ['BBALL2018_DEBUG_VAL']
            test_path = os.environ['BBALL2018_DEBUG_TEST']
        else:
            train_path = os.environ['BBALL2018_TRAIN']
            val_path = os.environ['BBALL2018_VAL']
            test_path = os.environ['BBALL2018_TEST']

        train_set = self.get_data(args, train_path)
        val_set = self.get_data(args, val_path)
        test_set = self.get_data(args, test_path)

        return train_set, val_set, test_set

class ExampleExperiment(ImageExperiment):

    def get_data(self, args, path):

        traj_locations = get_traj_locations(path, criterion=shot_length_criterion)

        transform_image_data = transform_producer(1)
        bball_dataset = BballDataset(traj_locations, transform=transform_image_data)
        return bball_dataset

    def wrap_dataset(self, dset, savedir, args):
        # save the data
        save_bball_data(dset, savedir, args.override_data)

        # return dataloader of the saved data
        dset = load_bball_data(savedir, args.target)

        return DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

class ImageShotEventExperiment(ImageExperiment):

    def get_data(self, args, path):
        pass

class TimeSeriesExperiment(ImageExperiment):

    def get_data(self, args, path):

        traj_locations = get_traj_locations(path, criterion=shot_length_criterion)
        transform_ts_data = transform_producer_ts(1)
        bball_dataset = BballDataset(traj_locations, transform=transform_ts_data)
        return bball_dataset

    def wrap_dataset(self, dset, savedir, args):
        # save the data
        save_bball_data(dset, savedir, args.override_data)

        dset = load_bball_data(savedir, args.target)

        #Deal with uneven lengths breaking DataLoader
        def pad_data(x):
            max_len = max([len(sig[0]) for sig in x])
            new_x = []
            for data in x:
                new_data = []
                for datum in data: 
                    datum = np.append(datum, np.zeros(max_len - len(datum)))
                    new_data.append([float(i) for i in datum])
                new_x.append(new_data)
            return new_x

        def my_collate(batch):
            data = [item[0] for item in batch]
            lengths = [len(d[0]) for d in data]
            data = torch.stack([torch.Tensor(x[0]).float() for x in pad_data(data)])
            target = [item[1] for item in batch]
            target = torch.FloatTensor(target)
            lens = torch.LongTensor(lengths)
            return [data, target, lens]

        return DataLoader(dset, batch_size = args.batch_size, collate_fn = my_collate, batch_first = True, shuffle=True, num_workers=args.num_workers)


class FlatInputExperiment(ImageExperiment):

    def get_data(self, args, path):

        traj_locations = get_traj_locations(path, criterion=shot_length_criterion)

        print(path, len(traj_locations))
        transform_flat_data = transform_producer_flat(1, crop_len=args.trajlen)
        bball_dataset = BballDataset(traj_locations, transform=transform_flat_data)
        return bball_dataset

    def wrap_dataset(self, dset, savedir, args):
        # save the data
        save_bball_data(dset, savedir, args.override_data)

        # return dataloader of the saved data
        dset = load_bball_data(savedir, args.target)

        return DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)


SYNTHETIC_EXPERIMENTS['example'] = ExampleExperiment()
SYNTHETIC_EXPERIMENTS['image_shot'] = ImageShotEventExperiment()
SYNTHETIC_EXPERIMENTS['timeseries'] = TimeSeriesExperiment()
SYNTHETIC_EXPERIMENTS['flatinput'] = FlatInputExperiment()
