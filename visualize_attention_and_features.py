"""
created by xingxiangrui on 2019.5.20
this program is to :
    print and visualize attention maps
    dimention reduction of input feature and visualize them


    code in clsgat_conv.py(save model inner results)
        def forward(self, x):
        # [B,N,C]
        B, N, C = x.size()
        print('B',B,'N',N,'C',C)
        # h = torch.bmm(x, self.W.expand(B, self.in_features, self.out_features))  # [B,N,C]

        # save resnet out feature path
        if self.save_attention_map == True:
            if not os.path.exists(self.save_resout_feature_path):
                feature_np=x.cpu().data.numpy()
                with open(self.save_resout_feature_path, 'wb') as f:
                    print('writing to', self.save_resout_feature_path)
                    pickle.dump(feature_np, f)

        h = torch.matmul(x, self.W)  # [B,N,C]

        # save GALayer in feature path
        if self.save_attention_map == True:
            if not os.path.exists(self.save_feature_in_GATLayer_path):
                feature_np = h.cpu().data.numpy()
                with open(self.save_feature_in_GATLayer_path, 'wb') as f:
                    print('writing to', self.save_feature_in_GATLayer_path)
                    pickle.dump(feature_np, f)


        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                                                                                                  2 * self.out_features)  # [B,N,N,2C]
        # temp = self.a.expand(B, self.out_features * 2, 1)
        # temp2 = torch.matmul(a_input, self.a)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N,N]

        attention = F.softmax(attention, dim=2)  # [B,N,N]
        attention = F.dropout(attention, self.dropout, training=self.training)

        #  save attention maps
        if self.save_attention_map==True:
            if not os.path.exists(self.save_attention_path):
                attention_value=attention.cpu().data.numpy()
                with open(self.save_attention_path, 'wb') as f:
                    print('writing to', self.save_attention_path)
                    pickle.dump(attention_value, f)

        h_prime = torch.bmm(attention, h)  # [B,N,N]*[B,N,C]-> [B,N,C]

        # save attention maps
        if self.save_attention_map == True:
            if not os.path.exists(self.save_GAlayer_out_feature_path):
                feature_np = h_prime.cpu().data.numpy()
                with open(self.save_GAlayer_out_feature_path, 'wb') as f:
                    print('writing to', self.save_GAlayer_out_feature_path)
                    pickle.dump(feature_np, f)
                    self.save_attention_map = False

"""
# import torch.utils.data as data
# import json
import os
# import subprocess
from PIL import Image
import numpy as np
# import torch
import pickle
import seaborn as sns
import scipy.misc as misc
from sklearn import decomposition
# from util import *
import pandas as pd
import matplotlib.pyplot as plt
# import warnings


class visualize_attention():
    def __init__(self):
        # super(self).__init__()
        # loaded features are numpy format,names is list
        # self.pkl_file_dir='/Users/baidu/Desktop/code/chun_ML_GCN/attention_analyse/'
        self.pkl_file_dir = '/Users/baidu/Desktop/code/chun_ML_GCN/attention_analyse/cls_gat_with_supple_loss/'
        self.attention_path=self.pkl_file_dir+'batch_attentions.pkl'
        self.names_files=self.pkl_file_dir+'coco_names.pkl'
        self.resout_feature_path = self.pkl_file_dir + 'resnet_out_feature.pkl'
        self.feature_in_GATLayer_path = self.pkl_file_dir + 'feature_in_BGALayer.pkl'
        self.GAlayer_out_feature_path = self.pkl_file_dir + 'GALayer_output_feature.pkl'
        self.save_correlation_heatmap=True
        self.save_PCA_features=False

    def run_visualize(self):
        # loading batch attention value and feature value
        with open(self.names_files, 'rb') as f:
            print("loading" ,self.names_files)
            names = pickle.load(f)      # 8*80*256
        with open(self.attention_path, 'rb') as f:
            print("loading",self.attention_path)
            attention_value = pickle.load(f)    # batch_size*80*80
        with open(self.feature_in_GATLayer_path, 'rb') as f:
            print("loading" ,self.feature_in_GATLayer_path)
            inGALayer_feature = pickle.load(f)      # batch_size*80*256
        with open(self.resout_feature_path, 'rb') as f:
            print("loading" ,self.resout_feature_path)
            resnet_out_feature = pickle.load(f)      # batch_size*80*256
        with open(self.GAlayer_out_feature_path, 'rb') as f:
            print("loading" ,self.GAlayer_out_feature_path)
            GAlyaer_out_feature = pickle.load(f)      # batch_size*80*256
        # feature
        self.resnet_out_feature,self.inGALayer_feature,self.GAlyaer_out_feature=resnet_out_feature,inGALayer_feature,GAlyaer_out_feature


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

        # print attention map
        if(self.save_correlation_heatmap == True):
            for map_idx in range(attention_value.shape[0]):
                attention_map=attention_value[map_idx,:,:]
                map_name=self.pkl_file_dir+'attention_map_'+str(map_idx)+'.jpg'
                # misc.toimage(attention_map).save(map_name)
                plot_cor(mat=attention_map, names=names, save_fig_name=map_name)

        # save features fig of given feature
        def each_category_pca_features(img_idx,feature_value):
            # feature dimension reduction and visualize
            pca = decomposition.PCA(n_components=2)
            # from 8*80*80 to 8*6400
            # feature_flatten = np.array([feature_value.shape[1], feature_value.shape[2]])
            category_feature=feature_value[img_idx,:,:]

            # PCA features
            X = pca.fit_transform(category_feature)
            dim0 = X[:, 0]
            dim1 = X[:, 1]
            # draw PCA results and save
            plt.scatter(dim0, dim1)
            return pca.explained_variance_ratio_

        def each_img_pca(img_idx,file_name):
            res_var_ratio=each_category_pca_features(img_idx=img_idx,feature_value=self.resnet_out_feature)
            ga_in_var_ratio=each_category_pca_features(img_idx=img_idx, feature_value=self.inGALayer_feature)
            ga_out_var_ratio=each_category_pca_features(img_idx=img_idx, feature_value=self.GAlyaer_out_feature)

            plt.legend(['resnet_out_feature| explain_var_ratio='+str(res_var_ratio),
                        'GALayer_input_feature| '+str(ga_in_var_ratio),
                        'GALayer_output_feature| '+str(ga_out_var_ratio)])
            plt.savefig(file_name)
            plt.close('all')

        if(self.save_PCA_features== True) :
            each_img_pca(img_idx=1,file_name=self.pkl_file_dir+'img_1_class_feature_pca.jpg')
            each_img_pca(img_idx=2, file_name=self.pkl_file_dir+'img_2_class_feature_pca.jpg')
            each_img_pca(img_idx=3, file_name=self.pkl_file_dir+'img_3_class_feature_pca.jpg')
            each_img_pca(img_idx=4, file_name=self.pkl_file_dir+'img_4_class_feature_pca.jpg')
            each_img_pca(img_idx=5, file_name=self.pkl_file_dir+'img_5_class_feature_pca.jpg')
            each_img_pca(img_idx=6, file_name=self.pkl_file_dir+'img_6_class_feature_pca.jpg')

        print('program end...')


if __name__ == '__main__':
    visualize_attention().run_visualize()
    # badcase_analyse().badcase_area_histogram()
    # badcase_analyse().hist_try()
    # badcase_analyse().coco_categories_names()








