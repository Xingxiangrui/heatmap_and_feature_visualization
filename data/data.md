# data format
numpy format

# data generation

most like this:
torch.Tensor.GPU---.cpu---.data---.numpy
        # save resnet out feature path
        if self.save_attention_map == True:
            if not os.path.exists(self.save_resout_feature_path):
                feature_np=x.cpu().data.numpy()
                with open(self.save_resout_feature_path, 'wb') as f:
                    print('writing to', self.save_resout_feature_path)
                    pickle.dump(feature_np, f)
