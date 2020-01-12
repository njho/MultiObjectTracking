import torch
from util.FeatureExtractor import FeatureExtractor
from torchvision import transforms
import models
from scipy.spatial.distance import cosine, euclidean
from  util.utils import *
from sklearn.preprocessing import normalize
import os
import random
import argparse
import json
from tqdm import tqdm


'''
to use:
    python run_reid_on_scenes \
            --scene_path=./scenes/smallTestVideo/\
            --out_path=./uid_json/\
            --th=0.4
'''


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        print('string', string)
        raise IOError(string)


def file_path(string):
    if os.path.exists(string):
        return string
    else:
        raise IOError(string)


def out_dir(string):
    if os.path.exists(string):
        return string
    else:
        os.mkdir(string)
        return string


parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=dir_path, required = True, help = 'directory that contains seperated scene files')
parser.add_argument('--jsons_path', type=dir_path, required = True, help = 'directory that contains seperated scene json files')
parser.add_argument('--out_path', type=out_dir, required = True, help = 'path to output directory')
parser.add_argument('--th', type=float, default=0.4, required=False, help = 'threshold of distance for similarity')
parser.add_argument('--model_path', type=file_path, default= './models/Market1501_Resnet50_Alignedreid/checkpoint_ep300.pth.tar', required=False, help = 'model path for inference')
parser.add_argument('--file_name_out', required = True, help = 'output file name')

        
def pool2d(tensor, type= 'max'):
    sz = tensor.size()
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    x = x[0].cpu().data.numpy()
    x = np.transpose(x,(2,1,0))[0]
    return x


def get_score(img_path1, img_path2, model):
    
    img_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img1 = read_image(img_path1)
    img2 = read_image(img_path2)
    img1 = img_to_tensor(img1, img_transform)
    img2 = img_to_tensor(img2, img_transform)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()
    model.eval()
    
    exact_list = ['7']
    myexactor = FeatureExtractor(model, exact_list)
    f1 = myexactor(img1)
    f2 = myexactor(img2)
    a1 = normalize(pool2d(f1[0], type='max'))
    a2 = normalize(pool2d(f2[0], type='max'))
    dist = np.zeros((8,8))
    for i in range(8):
        temp_feat1 = a1[i]
        for j in range(8):
            temp_feat2 = a2[j]
            dist[i][j] = euclidean(temp_feat1, temp_feat2)
    score = dtw(dist)
    return score


def dtw(dist_mat):
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]

    return dist[-1,-1]/sum(dist.shape)


def make_patches_dict(img_list):
    patches = dict()
    for img_path in img_list:
        if img_path.split('_')[0] not in patches.keys():
            patches[img_path.split('_')[0]] = [img_path]
        else:
            patches[img_path.split('_')[0]].append(img_path)
    return patches


def make_scene_patches_dict(video_scene_path):
    scenes = os.listdir(video_scene_path)
    scene_patches = dict()
    for scene in tqdm(scenes):
        img_ls = os.listdir(os.path.join(video_scene_path, scene))
        ls_frames = set([x.split('_')[0] for x in img_ls])
        patches = make_patches_dict(img_ls)
        scene_patches[scene] = patches
    return scene_patches


def compare_query_img_to_gallery(q_img, gallery, scene_path, model, threshold):
    uid_frames = [q_img]
    new_gallery = []
    q_img = os.path.join(scene_path, q_img)
    for g_img in gallery:
        g_img_path = os.path.join(scene_path, g_img)
        score = get_score(q_img, g_img_path, model)
        if score < threshold:
            uid_frames.append(g_img)
        else:
            new_gallery.append(g_img)
    return uid_frames, new_gallery


def generate_uid_for_dict(my_dict, uid_dict, uid, scene_path, model, threshold, count):
    count += 1
    my_keys = my_dict.keys()
    random.shuffle(my_keys)
    q_imgs = my_dict[my_keys[0]]
    gallery = []
    #print('generating uid for scene')
    for key in my_keys[1:]:
        for g_img in my_dict[key]:
            gallery.append(g_img)
    #print('gallery:', gallery)
    #print('q_imgs:', q_imgs)
    for q_img in q_imgs:
        uid_ls, gallery = compare_query_img_to_gallery(q_img, gallery, scene_path, model, threshold)
        uid_dict[uid] = uid_ls
        uid += 1
        count += 1
    if gallery:
        next_dict = make_patches_dict(gallery)
        uid_dict = generate_uid_for_dict(next_dict, uid_dict, uid, scene_path, model, threshold, count)
    return uid_dict


def run_uid_match(patches_dict, scene_path, model, threshold):
    uid_dict = dict()
    uid = 0
    count = 0
    uid_dict = generate_uid_for_dict(patches_dict, uid_dict, uid, scene_path, model, threshold, count)
    return uid_dict

if __name__== "__main__":
    args = parser.parse_args()
    scenes_path = args.scene_path
    jsons_path = args.jsons_path
    out_path = args.out_path
    mod_path = args.model_path
    file_name_out = args.file_name_out
    
    th = args.th
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    all_json = []

    print(file_name_out)
    
    use_gpu = torch.cuda.is_available()

    model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=use_gpu,aligned=True)

    checkpoint = torch.load(mod_path)

    model.load_state_dict(checkpoint['state_dict'])
    
    print('loaded model...')

    scenes = os.listdir(scenes_path)

    print('making patch dictionary')
    
    scene_patches_dict = make_scene_patches_dict(scenes_path)

    scene_uid_dict = dict()
    
    #print(scene_patches_dict)
    
    print('detecting unique persons per scene')
    
    print(th)

    for scene in tqdm(scenes):

        scene_path = os.path.join(scenes_path, scene)
        
        json_path = os.path.join(jsons_path, scene)
        
        if len(os.listdir(json_path)) == 0 or json_path.split('/')[-1][0] == '.':
            continue
      
        #print('Running uid match')
        uid_dict = run_uid_match(scene_patches_dict[scene], scene_path, model, th)
        
        scene_uid_dict[scene] = uid_dict
        
        for uid in uid_dict:
            
            for file_names_common in uid_dict[uid]:
            
                txt_file = file_names_common[:-4] + '.txt'
                
                txt_file_link = os.path.join(json_path, txt_file)
            
                #print('txt link:' , txt_file_link)
                with open(txt_file_link, 'r') as json_file:
                    data_txt = eval(json_file.read())
                
                    data_txt['uid'] = uid
                    
                with open(txt_file_link, 'w') as json_file:
                    
                    json_file.write(str(data_txt))
                    
                all_json.append(data_txt)
                
               
                    
            
    with open(out_path + file_name_out +  '_uid_per_scene.json', 'w') as fp:
        json.dump(scene_uid_dict, fp)
        
    with open(out_path + file_name_out + '_all_uid.json', 'w') as fp:
        json.dump(all_json, fp)
