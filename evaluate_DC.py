import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
use_gpu = torch.cuda.is_available()



def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    query = np.expand_dims(query, axis=1)
    base_means = np.expand_dims(np.array(base_means), axis=0)
    dist = np.linalg.norm(query - base_means, axis=2)
    index = np.argpartition(dist, k, axis=1)[:, :k]
    mean = np.concatenate([base_means[:, index][0], query], axis=1)
    calibrated_mean = np.mean(mean, axis=1)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=1)+alpha
    return calibrated_mean, calibrated_cov


if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples


    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)

    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- Tukey's transform
        beta = 0.5
        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(750/n_shot)
        
        means, covs = distribution_calibration(support_data, base_means, base_cov, k=2)
        rng = np.random.default_rng()
        for n_lsample in range(n_lsamples):
            sampled_data.append(rng.multivariate_normal(mean=means[n_lsample, :], cov=covs[n_lsample, :], size=num_sampled, method='cholesky'))

        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        sampled_label = np.tile(np.expand_dims(support_label, axis=1), num_sampled).reshape(n_ways * n_shot * num_sampled)
        X_aug = np.concatenate([support_data, sampled_data])
        Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)

        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
    print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))

