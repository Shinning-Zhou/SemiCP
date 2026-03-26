import torch

def cluster_feature(features, group_num, num_iters=100):
    # 记录原始形状
    N = features.shape[0]
    D = features.shape[1:]  # 其余维度的形状
    features_flat = features.view(N, -1)  # (N, D)

    # 随机初始化聚类中心
    indices = torch.randperm(N)[:group_num]
    centroids = features_flat[indices]

    for _ in range(num_iters):
        # 计算距离并分配类别
        distances = torch.cdist(features_flat, centroids)  # (N, group_num)
        cluster_labels = torch.argmin(distances, dim=1)  # (N,)

        # 更新聚类中心
        new_centroids = torch.zeros_like(centroids)
        for i in range(group_num):
            mask = cluster_labels == i
            if mask.sum() > 0:
                new_centroids[i] = features_flat[mask].mean(dim=0)
            else:
                new_centroids[i] = centroids[i]

        # 判断是否收敛
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return cluster_labels, centroids


def get_groups(features, centroids):
    if centroids is None:
        raise ValueError("Centroids are not initialized. Run cluster_feature first.")

    # 计算特征的形状并展平
    N = features.shape[0]
    features_flat = features.view(N, -1)  # (N, D)

    # 计算距离并分配组标签
    distances = torch.cdist(features_flat, centroids)  # (N, group_num)
    group_labels = torch.argmin(distances, dim=1)  # (N,)

    return group_labels
