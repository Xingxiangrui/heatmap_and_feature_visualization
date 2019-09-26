# heatmap_and_feature_visualization
visualize heatmap and model features

热图可视化，与feature的降维可视化：
https://blog.csdn.net/weixin_36474809/article/details/90370876

背景：运行模型时，经常需要将相应的数据可视化。

目录

一、网络结果存为np

1.1 网络输出存储

1.2 GPU张量转换

1.3 流程

二、heatmap输出

2.1 misc函数

2.2 生成heatmap

2.3 sns.heatmap

2.4 print标签

三、数据可视化

3.1 flatten降维

3.2 PCA降维

3.3 绘出散点图

3.4 三维数组展开二维数组

3.5 清除plt数据

3.6 PCA的expain ration

四、图像格式转换

4.1 jpg转png

五、代码
一、网络结果存为np

注意网络之中，文件为CUDA的torch Tensor，因此需要转化为numpy格式方便运算与读取。
1.1 网络输出存储

网络输出结果为numpy结构，只需np.append即可

            output_data_np,labels_np=self.on_forward_analyse(False, model, criterion, data_loader)
            # output_and_labels={'output_data_np':output_data_np,'labels_np':labels_np}

            all_output_results[i]=output_data_np
            all_labels[i]=labels_np
            # print('all_output_results',all_output_results)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            # self.on_end_batch(False, model, criterion, data_loader)
        # all validate results and labels on coco
        # print('all_output_results',all_output_results)
        # print('all_labels',all_labels)

        # concat all numpy
        total_results = all_output_results[0]
        total_labels = all_labels[0]
        for img_idx in range(len(all_output_results) - 1):
            if img_idx % 1000 == 0:
                print(img_idx, '/', len(all_output_results))
            total_results= np.append(total_results, all_output_results[img_idx + 1], axis=0)
            total_labels = np.append(total_labels, all_labels[img_idx + 1], axis=0)
        with open('checkpoint/coco/weight_decay_cls_gat_on_5_10/model_results_numpy.pkl', 'wb') as f:
            print("writing checkpoint/coco/weight_decay_cls_gat_on_5_10/model_results_numpy.pkl")
            pickle.dump(total_results, f)
        with open('checkpoint/coco/weight_decay_cls_gat_on_5_10/coco_labels_numpy.pkl', 'wb') as f:
            print("writing checkpoint/coco/weight_decay_cls_gat_on_5_10/oco_labels_numpy.pkl")
            pickle.dump(total_labels, f)

1.2 GPU张量转换

网络预测结果往往在GPU上，因此需要做一定的转换

# compute output
self.state['output'] = model(feature_var, inp_var)
# .data-----.cpu()------.numpy
output_data_np=self.state['output'].cpu().data.numpy()
labels_np=target_var.cpu().data.numpy()

1.3 流程

    GPU_Tensor.cpu().data.numpy()从GPU上的张量转为numpy格式
    np.append 将相应的numpy拼接
    pickle.dump 写入相应的文件

二、heatmap输出
2.1 misc函数

值为0到1之间。

https://blog.csdn.net/mtj66/article/details/80178086

import scipy.misc
misc.imsave('out.jpg', image_array)

上面的scipy版本会标准化所有图像，以便min(数据)变成黑色，max(数据)变成白色。如果数据应该是精确的灰度级或准确的RGB通道，则解决方案为：

import scipy.misc
misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
2.2 生成heatmap

另种方法是用函数的方法，直接将相应的结果生成heatmap。生成heatmap可在此代码基础上更改。

        # plot heatmap of matrix
        def plot_cor(mat, names,save_fig_name):
            fig, ax = plt.subplots()
            # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
            # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
            sns.heatmap(
                pd.DataFrame(mat * (mat >= 0), columns=names,
                             index=names),
                xticklabels=True,
                yticklabels=True, cmap="YlGnBu")
            # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
            #            square=True, cmap="YlGnBu")
            ax.set_title('Correlation', fontsize=18)
            # ax.set_ylabel('Attribute', fontsize=18)
            # ax.set_xlabel('Attribute', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
            plt.savefig(save_fig_name)
            plt.close('all')

运行结果：

2.3 sns.heatmap

            sns.heatmap(
                pd.DataFrame(mat * (mat >= 0), columns=names,
                             index=names),
                xticklabels=True,
                yticklabels=True, annot=True,cmap="YlGnBu")

2.4 print标签

设置阈值，阈值为最大值最小值的平均。

        def print_big_correlation_labels(attention_map):
            threshood=(np.max(attention_map)+np.min(attention_map))/2.0
            related_list=[]
            for col_idx in range(attention_map.shape[0]):
                if (np.mean(attention_map[:,col_idx])>threshood):
                    related_list.append(col_idx)
            return related_list

输出相应的labels

        # print attention heatmap and save
        if(self.save_correlation_heatmap == True):
            for map_idx in range(attention_value.shape[0]):
                attention_map=attention_value[map_idx,:,:]
                map_name=self.pkl_file_dir+'attention_map_'+str(map_idx)+'.jpg'
                # misc.toimage(attention_map).save(map_name)
                plot_cor(mat=attention_map, names=names, save_fig_name=map_name)

        def print_big_correlation_labels(attention_map):
            threshood=(np.max(attention_map)+np.min(attention_map))/2.0
            related_list=[]
            for col_idx in range(attention_map.shape[0]):
                if (np.mean(attention_map[:,col_idx])>threshood):
                    related_list.append(col_idx)
            return related_list

        def from_idx_list_to_name_list(idx_list):
            name_list=[]
            for idx in range(len(idx_list)):
                name_list.append(self.names[idx_list[idx]])
            return name_list

        # print attention heatmap labels
        if (self.print_attention_correlation_labels== True):
            for map_idx in range(attention_value.shape[0]):
                attention_map = attention_value[map_idx, :, :]
                # map_name = self.pkl_file_dir + 'attention_map_' + str(map_idx) + '.jpg'
                print(map_idx)
                related_list=print_big_correlation_labels(attention_map=attention_map)
                print(related_list)
                name_list=from_idx_list_to_name_list(idx_list=related_list)
                print(name_list)


三、数据可视化

这个链接里面汇集了多重数据降维的方法：https://www.jianshu.com/p/3bb2cc453df1
3.1 flatten降维

https://blog.csdn.net/brucewong0516/article/details/79185282

.flatten() ： 对数组进行降维，返回折叠后的一维数组，原数组不变
3.2 PCA降维

运用sk learn中的PCA

https://blog.csdn.net/u012162613/article/details/42192293

代码中为，对每个catgory_feature进行PCA降维，然后画出。

将不同的种类的PCA散点图拼接在一起，save。

save后需要 plt.close('all')来清空数据，免得后续受到影响。

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

        def each_img_pca(img_idx,file_name):
            each_category_pca_features(img_idx=img_idx,feature_value=self.resnet_out_feature)
            each_category_pca_features(img_idx=img_idx, feature_value=self.inGALayer_feature)
            each_category_pca_features(img_idx=img_idx, feature_value=self.GAlyaer_out_feature)
            plt.legend(['resnet_out_feature', 'GALayer_input_feature', 'GALayer_output_feature'])
            plt.savefig(file_name)
            plt.close('all')

        if(self.save_PCA_features== True) :
            each_img_pca(img_idx=1,file_name='img_1_class_feature_pca.jpg')
            each_img_pca(img_idx=2, file_name='img_2_class_feature_pca.jpg')
            each_img_pca(img_idx=3, file_name='img_3_class_feature_pca.jpg')
            each_img_pca(img_idx=4, file_name='img_4_class_feature_pca.jpg')
            each_img_pca(img_idx=5, file_name='img_5_class_feature_pca.jpg')
            each_img_pca(img_idx=6, file_name='img_6_class_feature_pca.jpg')

3.3 绘出散点图

直接画出散点图：

import matplotlib.pyplot as plt

year=[1950,1970,1990,2010]

pop=[2.518,3.68,5.23,6.97] #2.散点图,只是用用scat函数来调用即可
plt.scatter(year,pop)

plt.show()

例如：

            # PCA features
            X = pca.fit_transform(category_feature)
            dim0 = X[:, 0]
            dim1 = X[:, 1]
            # draw PCA results and save
            plt.scatter(dim0, dim1)

3.4 三维数组展开二维数组

https://blog.csdn.net/u013044310/article/details/86383162

np.reshape(A,(a,b)) 函数即可。

用到的参数：

    A:需要被重新组合的数组
    (a,b): 各个维度的长度。比如要想展开成二维数组，那么(a,b)就是展开成a行b列。

3.5 清除plt数据

运行结束的时候务必加这个函数，不然plt生成的图像会叠加在一起。

https://cloud.tencent.com/developer/ask/37449

plt.close('all')
3.6 PCA的expain ration

https://blog.csdn.net/qq_36523839/article/details/82558636

除了这些输入参数外，有两个PCA类的成员值得关注。第一个是explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。第二个是explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。

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

四、图像格式转换
4.1 jpg转png


from PIL import Image
import cv2 as cv
import os

def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w), int(h)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            # img.convert('RGB').save(outfile, quality=70)
            img.convert('RGB').save(outfile)
            os.remove(PngPath)
        else:
            # img.convert('RGB').save(outfile, quality=70)
            img.convert('RGB').save(outfile)
            # os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


if __name__ == '__main__':
    PNG_JPG("/Users/baidu/Desktop/工作/personal/成绩单截图.png")
