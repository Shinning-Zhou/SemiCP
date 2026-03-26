import numpy as np
import torch
from sklearn.cluster import KMeans
from .cluster import ClusteredPredictor

class SemiClusteredPredictor(ClusteredPredictor):
    def __init__(self, score_function, unlabeled_score_function, model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random'):
        super().__init__(score_function, model, temperature, ratio_clustering=ratio_clustering, num_clusters=num_clusters, split=split)
        self.unlabeled_score_function = unlabeled_score_function
        
    def calculate_threshold(self, labeled_logits, labeled_labels, unlabeled_logits, alpha):
        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")
        labeled_logits = labeled_logits.to(self._device)
        labeled_labels = labeled_labels.to(self._device)
        unlabeled_logits = unlabeled_logits.to(self._device)
        num_classes = labeled_logits.shape[1]
        
        labeled_scores = self.score_function(labeled_logits, labeled_labels)
        unlabeled_scores, unlabeled_labels = self.unlabeled_score_function(unlabeled_logits, labeled_logits, labeled_labels, self.score_function)
        
        scores = torch.cat([labeled_scores, unlabeled_scores])
        labels = torch.cat([labeled_labels, unlabeled_labels])

        alpha = torch.tensor(alpha, device=self._device)
        classes_statistics = torch.tensor([torch.sum(labels == k).item() for k in range(num_classes)],
                                          device=self._device)

        # 1) Choose necessary parameters for Cluster algorithm
        if self._ratio_clustering == 'auto' or self._num_clusters == 'auto':
            n_min = torch.min(classes_statistics)
            n_thresh = self._get_quantile_minimum(alpha)
            # Classes with fewer than n_thresh examples will be excluded from clustering
            n_min = torch.maximum(n_min, n_thresh)
            num_remaining_classes = torch.sum((classes_statistics >= n_min).float())

            # Compute the number of clusters and the minium number of examples for each class
            n_clustering = (n_min * num_remaining_classes / (75 + num_remaining_classes)).clone().to(
                torch.int32).to(self._device)
            if self._num_clusters == 'auto':
                self._num_clusters = torch.floor(n_clustering / 2).to(torch.int32)
            if self._ratio_clustering == 'auto':
                self._ratio_clustering = n_clustering / n_min

        # 2) Split data
        clustering_scores, clustering_labels, cal_scores, cal_labels = self._split_data(scores,
                                                                                         labels,
                                                                                         classes_statistics)

        # 3)  Filter "rare" classes
        rare_classes = self._get_rare_classes(clustering_labels, alpha, num_classes)
        self.num_clusters = self._num_clusters
        # 4) Run clustering
        if (num_classes - len(rare_classes) > self._num_clusters) and (self._num_clusters > 1):
            # Filter out rare classes and re-index
            remaining_idx, filtered_labels, class_remapping = self._remap_classes(clustering_labels, rare_classes)
            filtered_scores = clustering_scores[remaining_idx]

            # Compute embedding for each class and get class counts
            embeddings, class_cts = self._embed_all_classes(filtered_scores, filtered_labels)
            kmeans = KMeans(n_clusters=int(self._num_clusters), n_init=10, random_state=2023).fit(
                X=embeddings.detach().cpu().numpy(),
                sample_weight=np.sqrt(
                    class_cts.detach().cpu().numpy()),
                )
            nonrare_class_cluster_assignments = torch.tensor(kmeans.labels_, device=self._device)

            cluster_assignments = - torch.ones((num_classes,), dtype=torch.int32, device=self._device)

            for cls, remapped_cls in class_remapping.items():
                cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
        else:
            cluster_assignments = - torch.ones((num_classes,), dtype=torch.int32, device=self._device)
        self.cluster_assignments = cluster_assignments
        # 5) Compute qhats for each cluster

        self.q_hat = self._compute_cluster_specific_qhats(cluster_assignments,
                                                           cal_scores,
                                                           cal_labels,
                                                           alpha)
        