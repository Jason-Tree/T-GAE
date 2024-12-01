# T-GAE: Transferable Graph Autoencoder for Network Alignment

This repository contains code for the paper T-GAE: Transferable Graph Autoencoder for Network Alignment. The paper is accepted by Proceedings of the Third Learning on Graphs Conference (LoG 2024)

Abstract: 

Network alignment is the task of establishing one-to-one correspondences be- tween the nodes of different graphs. Although finding a plethora of applications in high-impact domains, this task is known to be NP-hard in its general form. Existing optimization algorithms do not scale up as the size of the graphs in- creases. While being able to reduce the matching complexity, current GNN approaches fit a deep neural network on each graph and requires re-train on unseen samples, which is time and memory inefficient. To tackle both challenges we propose T-GAE, a transferable graph autoencoder framework that leverages transferability and stability of GNNs to achieve efficient network alignment on out-of-distribution graphs without retraining. We prove that GNN-generated embeddings can achieve more accurate alignment compared to classical spec- tral methods. Our experiments on real-world benchmarks demonstrate that T-GAE outperforms the state-of-the-art optimization method and the best GNN approach by up to 38.7% and 50.8%, respectively, while being able to reduce 90% of the training time when matching out-of-distribution large scale networks. We conduct ablation studies to highlight the effectiveness of the proposed encoder architecture and training objective in enhancing the expressiveness of GNNs to match perturbed graphs. T-GAE is also proved to be flexible to uti- lize matching algorithms of different complexities.

## Code overview

This repository contains the source code that evaluates the performance of Moment-GNN in:

  - Graph Matching
  - Sub-graph Matching

## Dependencies

To run this code please install Pytorch, numpy, munkres, scipy, netrd, networkx.

We have used torch==2.0.1, numpy==1.23.1, munkres==1.1.4, scipy==1.12.0, networkx==3.2.1

## Data

All data used are inclueded in the data.zip file, please run the following command to unzip the data:

```
unzip data.zip
```

## Graph Matching
To run the experiments for graph matching in Section 5.2 please run:
```
python graphMatching.py
```

The above command will run Celegans on 0 perturbation level by default. To choose dataset, perturbation model and level, matching algorithm, or evaluation interval, please follow this example:
```
python graphMatching.py --dataset celegans --model uniform --level 0 --algorithm greedy --eval_interval 1
```
where dataset is one of {celegans, arenas, douban, cora, dblp, coauthor_cs}, perturbation model is one of {uniform, degree}, level is one of {0, 0.01, 0.05}, algorithm is one of {greedy, exact, approxNN}, for small sized graphs (Celegans, Arenas), we suggest to run exact for better performance. For 0 perturbation level, we suggest run approxNN for better efficiency. To sppedup experiments, you can set eval_interval larger, but this may not achieve the best performance. 

## Sub-graph Matching
To run the experiments for sub graph matching in Section 5.3, please run:
```
python subgraphMatching.py
```

To choose dataset, please follow this example:
```
python subgraphMatching.py --dataset ACM_DBLP
```
where ACM_DBLP is one of {ACM_DBLP, Douban Online_Offline}

## License
MIT

## Citation

If you find the code in this repo useful, please consider citing our paper:

@misc{he2024tgaetransferablegraphautoencoder,
      title={T-GAE: Transferable Graph Autoencoder for Network Alignment}, 
      author={Jiashu He and Charilaos I. Kanatsoulis and Alejandro Ribeiro},
      year={2024},
      eprint={2310.03272},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url=https://arxiv.org/abs/2310.03272, 
}
