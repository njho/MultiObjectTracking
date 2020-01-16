#-*- coding: UTF-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import eval_widerface
import torchvision_model
import model
import os

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--video_path', type=str, default='video_record.avi', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model')
    parser.add_argument('--save_path', type=str, default='./out/result.avi', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    # Create torchvision model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(args.model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()

    # Read video
    cap = cv2.VideoCapture(args.video_path)

    codec = cv2.VideoWriter_fourcc(*'MJPG')

    # width = int(cap.get(3))
    # height = int(cap.get(4))
    #
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter('args.save_path', fourcc, frames_per_second, (width, height))

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # fps = 24.0
    # out = cv2.VideoWriter('args.save_path', codec, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_file = open("bbox_coords.txt","w+")

    while(True):
        ret, img = cap.read()

        if not ret:
            print('Video open error.')
            break

        img = torch.from_numpy(img)
        img = img.permute(2,0,1)

        if not args.scale == 1.0:
            size1 = int(img.shape[1]/args.scale)
            size2 = int(img.shape[2]/args.scale)
            img = resize(img.float(),(size1,size2))
        current_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        input_img = img.unsqueeze(0).float().cuda()
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(input_img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)

        # np_img = resized_img.cpu().permute(1,2,0).numpy()
        np_img = img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        img = np_img.astype(np.uint8)

        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box,landmark,score in zip(boxes,picked_landmarks[j],picked_scores[j]):

                    # Crop image
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    txt_file.write(" {0}: ({1},{2}), ({3},{4}), ({5},{6}), ({7},{8})".format(current_timestamp,x,y,x+w,y,x+w,y+h,x,y+h))
                    # x1 = width * x
                    # y1 = height * y               # Write
                    # x2 = x1 + width * w   # x, y, w, h = cv2.boundingRect(boxes[idx])
                    # y2 = y1 + height * h
                    # coords = [x1, y1, x2, y2]

                    # Draw
                    print(" {0}: ({1},{2}), ({3},{4}), ({5},{6}), ({7},{8})".format(current_timestamp,x,y,x+w,y,x+w,y+h,x,y+h))
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)

                    # Save Image
                    # cropped_image = img.crop(coords)
                    # cv2.imwrite(os.path.join('output_images', str(current_timestamp) + '.jpg'), cropped_image)
                    # crop_and_save_image(img,[x,y,w,h],current_timestamp,"output_images")
                    # crop_img = img[y:int(box[0])+ int(box[1]), x:int(box[2])+ int(box[3])]
                    # x_crop = ((x,y),(x+w))
                    # y_crop = ((x+w,y+h),(x,y+h))
                    # x, y, w, h = cv2.boundingRect(contours[i])
                    # crop_img = img[y:y+w, x:x+h]
                    # You may need to convert the color.
                    img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img_conv)
                    im_pil.crop((x,y,w,h))
                    im_pil.save(os.path.join('output_images', str(current_timestamp) + '.jpg'))
                    # cv2.imwrite(os.path.join('output_images', str(current_timestamp) + '.jpg'), crop_img)


                    # cv2.circle(img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
                    # cv2.circle(img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
                    # cv2.circle(img,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
                    # cv2.circle(img,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
                    # cv2.circle(img,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
                    # cv2.circle(img,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)
                    # cv2.putText(img, text=str(score.item())[:5], org=(box[0],box[1]), fontFace=font, fontScale=0.5,
                    #             thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))
        out.write(img)
        # cv2.imshow('RetinaFace-Pytorch',img)
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     print('Now quit.')
        #     break
    txt_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()