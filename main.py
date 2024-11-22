import numpy as np
import pandas as pd
from utils import read_data, cluster_variances, setup_seed, initialization, compute_tilted_sse_InEachCluster, get_noised_data, compute_sse
from update import rkm, Fastrkm
from options import args_parser
import json
import time
args = args_parser()
setup_seed(args.seed)
print('random seed =', args.seed)
print('Initialization method: ', args.init)
# csv_file_path = 'data/CleanData/'+args.dataset+'.csv'
num_subsample = args.num_subsample
sample_size = args.sample_size
# dataset_name = args.dataset
# dataset_names = ['athlete','bank','census','diabetes','recruitment','spanish','student','3d']
dataset_names = ['census']
for dataset_name in dataset_names:
    for subsample_id in range(0, num_subsample):
        csv_file_path = '../individually-fair-k-clustering-main/data/'+dataset_name+'_'+str(sample_size)+'_'+str(subsample_id)+'.csv'
        data = read_data(csv_file_path)
        data_with_noise = get_noised_data(data, args.noise_frac)
        t_list = args.t
        k_list = args.num_clusters
        epoch_list = args.epoch_list
        lr_list = args.lr_list
        distances_file = '../individually-fair-k-clustering-main/data/'+dataset_name+'_'+str(sample_size)+'_'+str(subsample_id) + "_distances.csv"

        all_pair_distances = pd.read_csv(distances_file)
        all_pair_distances = all_pair_distances.values
        for k in k_list:
            for t in t_list:
                for num_epoch in epoch_list:
                    for lr in lr_list:
                        output = {}
                        time1 = time.monotonic()
                        centroids, labels = initialization(data_with_noise, k, args)
                        phi = compute_tilted_sse_InEachCluster(data_with_noise, centroids, labels, k, t)
                        if args.solver == 'rkm':
                            centroids, labels, SSE, tilted_SSE, max_violation = rkm(data_with_noise, args, t, k, num_epoch, lr, all_pair_distances)
                        elif args.solver == 'r3km':
                            centroids, labels, SSE, tilted_SSE = Fastrkm(data_with_noise, args, t, k, num_epoch, lr, centroids, labels, phi)
                        else:
                            exit('Not implemented solver')
                        time2 = time.monotonic()
                        SSE_without_noise = compute_sse(data, centroids)
                        # print("Tilted Mini-Batch K-Means center:\n", centroids)

                        print("SSE in each iteration:\n", SSE)
                        print("tilted SSE in each iteration:\n", tilted_SSE)


                        print(f't={t}, k={k}')

                        print('Running time:', time2-time1)
                        print('SSE without noise:', SSE_without_noise)
                        output['dataset'] = dataset_name
                        output['SSE_iteration'] = SSE
                        output['tilted_SSE_iteration'] = tilted_SSE
                        output['SSE'] = np.mean(SSE[-20:])/sample_size
                        output['tilted_SSE'] = min(tilted_SSE)
                        address = 'output/'
                        # comparison among other methods, use this file_name
                        # file_name = address + dataset_name + '_t='+str(t)+'_k='+str(k)+'_id='+str(subsample_id)+'.json'

                        # comparison among different parameters, use this file_name
                        file_name = (address + dataset_name + '_t=' + str(t) + '_k=' + str(k) + '_id=' +
                                     str(subsample_id) + '_lr=' +str(lr)+'_epoch=' + str(num_epoch) + '.json')
                        with open(file_name, "w") as dataf:
                            json.dump(output, dataf)
