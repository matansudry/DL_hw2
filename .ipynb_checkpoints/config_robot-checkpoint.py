# paths
train_target_path = 'data/cache/train_target.pkl'
val_target_path = 'data/cache/val_target.pkl'
images_path_train = '../../../datashare/train2014'
images_path_validation = '../../../datashare/val2014'
ans2label_path = 'data/cache/trainval_ans2label.pkl'
label2ans_path = 'data/cache/trainval_label2ans.pkl'
q2label_path = 'data/cache/trainval_q2label.pkl'
label2q_path = 'data/cache/trainval_label2q.pkl'
preprocessed_image_train = 'data/cache/train.h5'
image_prefix_train = 'COCO_train2014_'
image_prefix_validation = 'COCO_val2014_'
img2idx_train = 'data/cache/img2idx_train.pkl'

# image preprocessing
image_scale = 256
target_size = 224

# net parameters
# question net
word_embeddings_dim = 300
lstm_hidden_dim = 256
output_dim = 4096
lstm_drop = 0.4
hidden_dim = 100
accumulate_grad_steps = 50

# image net
out_channels1 = 16
out_channels2 = 32
out_channels3 = 72
kernel_size = 3
max_pool_kernel = 2
stride = 2
pad = 1

# learning parameters
parallel = True
epochs = 50
batch_size = 256
num_workers = 8
lr_step_size = 30
lr = 0.001
lr_gamma = 0.1
