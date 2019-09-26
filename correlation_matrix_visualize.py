'''
created by xingxiangrui on 2019.5.22
generate and visualize correlation labels to heatmap
'''
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os

from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics


class adj_matrix_gen():

    def __init__(self):
        self.load_dir='/Users/Desktop/code/ML_GAT-master/'
        self.save_dir='/Users/Desktop/code/ML_GAT-master/'
        self.load_corelation_filename='coco_correlations.pkl'
        self.load_names_filename='coco_names.pkl'
        self.save_filename='coco_normalized_adj.pkl'

        self.load_correlation_file_path=self.load_dir+self.load_corelation_filename
        self.load_names_file_path=self.load_dir+self.load_names_filename
        self.save_file_path=self.save_dir+self.save_filename
        self.un_normalized_fig_path=self.save_dir+'un_normalized_adj.jpg'
        self.normalized_fig_path=self.save_dir+'normalized_adj.jpg'

    def run_adj_matrix_gen(self):

        # plot heatmap of matrix
        def plot_cor(mat, names,save_fig_name):
            fig, ax = plt.subplots()
            # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
            # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
            sns.heatmap(
                pd.DataFrame(mat * (mat >= 0), columns=names,
                             index=names),
                xticklabels=True,
                yticklabels=True,cmap="YlGnBu") # cmap="YlGnBu"  'RdYlBu'
            # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
            #            square=True, cmap="YlGnBu")
            ax.set_title('Correlation', fontsize=18)
            # ax.set_ylabel('Attribute', fontsize=18)
            # ax.set_xlabel('Attribute', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
            plt.savefig(save_fig_name)
            plt.close('all')

        def normalize_adj_gen(un_normalized_adj,threshold):
            un_normalized_adj = (un_normalized_adj > threshold) * un_normalized_adj
            # adj[adj>0.4]=1
            D = np.power(np.sum(un_normalized_adj, axis=1), -0.5)
            D = np.diag(D)
            normalized_adj = np.matmul(np.matmul(D, un_normalized_adj), D)
            return normalized_adj


        # load correlation matrix and names
        with open(self.load_correlation_file_path, 'rb') as f:
            print("loading ",self.load_correlation_file_path," from local...")
            correlations = pickle.load(f)
        unnormalized_adj = correlations['pp']
        with open(self.load_names_file_path,'rb') as f:
            print("loading ",self.load_names_file_path," from local...")
            names=pickle.load(f)


        plot_cor(mat=unnormalized_adj,names=names,save_fig_name=self.un_normalized_fig_path)

        normalized_adj=normalize_adj_gen(un_normalized_adj=unnormalized_adj,threshold=0.1)

        plot_cor(mat=normalized_adj, names=names, save_fig_name=self.normalized_fig_path)


if __name__ == '__main__':
    adj_matrix_gen().run_adj_matrix_gen()
    print('finished....')
