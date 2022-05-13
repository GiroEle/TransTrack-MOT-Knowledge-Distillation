# Modified by Peize Sun
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import numpy as np
import torch
import torchvision.transforms.functional as F
import glob as gb
import os
import cv2

from models import build_tracktest_model
from models import Tracker
import datasets.transforms as T
from util.misc import nested_tensor_from_tensor_list
from track_tools.colormap import colormap


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='mot')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    #parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # PyTorch checkpointing for saving memory (torch.utils.checkpoint.checkpoint)
    parser.add_argument('--checkpoint_enc_ffn', default=False, action='store_true')
    parser.add_argument('--checkpoint_dec_ffn', default=False, action='store_true')

    # demo
    parser.add_argument('--video_input', default='demo.mp4')
    parser.add_argument('--demo_output', default='demo_output')
    parser.add_argument('--track_thresh', default=0.4, type=float)
    

    #FPS_eval
    parser.add_argument('--max_size', default=1333, type=float)
    return parser


def resize(image, size=800, max_size=1333):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        h, w = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    new_height, new_width = get_size_with_aspect_ratio(image.shape[:2], size, max_size)    
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, new_height, new_width


def main(args):
    f = open("./demo_output/FPS_eval.txt",'w')
    #02-25 between runs clear GPU
    torch.cuda.empty_cache()

    cap = cv2.VideoCapture(args.video_input)

    frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
    demo_images_path = os.path.join(args.demo_output, 'demo_images')
    if not os.path.exists(demo_images_path):
        os.makedirs(demo_images_path)
        
    device = torch.device(args.device)
    model, _, postprocessors = build_tracktest_model(args)    
    model.to(device)
    model.eval()
    tracker = Tracker(score_thresh=args.track_thresh)

    #taking out resume so I can try out different models without having to train for 15 min 1 epoch
    #each time
    """
    checkpoint = torch.load(args.resume, map_location='cpu')
    _, _ = model.load_state_dict(checkpoint['model'], strict=False)
    print("Model is loaded")
    """
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    color_list = colormap()
    
    
    print("Starting inference")
    count = 0
    tracker.reset_all()
    pre_embed = None    
    res, img = cap.read()



    #giro
    #from torchsummary import summary
    ##summary(model, input_size=(3, 750, 1333))
    import gc
    x=0
    obj_list=[]
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_list.append(obj)
                x+=obj.reshape(-1).size()[0]
        except:
            pass
    print("g-inital params in gc:",x)
    print("g-res:",res)


    #03-23 replacing time with timeit

    #giro: video sim
    max_frames = 50
    fps=cap.get(cv2.CAP_PROP_FPS)
    frame_increment=1/fps
    import time,math
    import timeit
        
    fps_list=[]
    #eval_output=np.zeros((max_frames,3)) # inference time as period; FPS of inference time (without previously counted section); time on output image (previously counted section); 

    time_start=timeit.default_timer()
    part4_elasped_start=-1
    #time_first_start=time_start
        

    #while res and timeit.default_timer()-time_first_start < 219:
    #while res and timeit.default_timer()-time_first_start < 100:
    while res and count<max_frames:
        count += 1
        #reduces largest dimension of inputted image to max_size = 1333
        
        #temp_elasped=timeit.default_timer()
        
        #g: altering input image size
        resized_img, nh, nw = resize(img,max_size=args.max_size)
        #resized_img, nh, nw = resize(img,max_size=800)

        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)        
        tensor_img = F.normalize(F.to_tensor(rgb_img), mean, std)
        samples = nested_tensor_from_tensor_list([tensor_img]).to(device)
        
        
        
        #print("-1:{:.3f} included in 4 (pre_model input)".format(timeit.default_timer()-temp_elasped))
        
        part1_3_elasped=timeit.default_timer()
        #02-25 Might want to try using pre_embed again. Was not possible originally because of memory limitiations.
        #outputs, pre_embed = model(samples, pre_embed)
        outputs, pre_embed = model(samples)

        #temp_elasped=timeit.default_timer()

        part1_3_elasped=timeit.default_timer()-part1_3_elasped
        orig_sizes = torch.stack([torch.as_tensor([video_height, video_width])], dim=0).to(device)
        results = postprocessors['bbox'](outputs, orig_sizes)
        if count == 1:
            res_track = tracker.init_track(results[0])
        else:
            res_track = tracker.step(results[0])

        if part1_3_elasped != -1: 
            print("4:{:.3f} post-processing:".format(timeit.default_timer()-time_start-part1_3_elasped))

        ## 02-25 removing task of outputing bounding box images from FPS eval (using curtime listed below instead of other position)
        ## solution reached after res_track found
        fps_list.append(1/(timeit.default_timer()-time_start))
        #eval_output[count-1][0]=timeit.default_timer()-time_start
        #eval_output[count-1][1]=fps_list[count-1]
        curr_time=timeit.default_timer()

        """
        #giro    
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print("total_mem",t,"mem_reserved",r,"mem_allocated",a,"free in reserved",f)
        """

        #print("-1:{:.3f} included in 4 (pre_model input)".format(timeit.default_timer()-temp_elasped))

        """
        for ret in res_track:
            if ret['active'] == 0:
                continue
            bbox = ret['bbox']
            tracking_id = ret['tracking_id']

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[tracking_id%79].tolist(), thickness=2)
            cv2.putText(img, "{}".format(tracking_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[tracking_id%79].tolist(), 2)
        
        cv2.imwrite(os.path.join(demo_images_path, "demo{:0>6d}.png".format(count)), img)
        #print('Frame{:d} of the video is done'.format(count))
        """

        """02-25 removing the task of building images from the inference speed test by moving outside of tracked time
        #giro: video sim - this section takes 0.5 seconds but doesn't delay
        fps_list.append(1/(timeit.default_timer()-time_start))
        curr_time=timeit.default_timer()
        """
        #eval_output[count-1][2]=timeit.default_timer()-curr_time

        #simulate real-time: by "missing frames" that occured during the inference step
        #take floor, assuming that if inbetween frame inputs, the system will use the current
        #latest available frame
        
        for i in range(math.floor( (curr_time-time_start)/frame_increment)):
            res, img = cap.read()

        #giro:video sim: time stop after latest frame update
        time_start=timeit.default_timer()

        """ 02-25 trying to break 6 Hz gap
        #giro: clearing to free up memory
        torch.cuda.empty_cache()
        del samples
        del orig_sizes
        """

    print('Lenth of the video: {:d} frames'.format(count))

    #giro:video sim
    print("(Real-Time Evaluation) Average FPS: ", sum(fps_list)/len(fps_list))
    print("fps_list ",fps_list)
    print("(Real-Time Evaluation) Average FPS: ", sum(fps_list)/len(fps_list), file=f)
    print("fps_list ",fps_list, file=f)

    print("Starting img2video")
    img_paths = gb.glob(os.path.join(demo_images_path, "*.png"))
    size = (video_width, video_height) 
    videowriter = cv2.VideoWriter(os.path.join(args.demo_output, "demo_video.avi"), cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, size)

    for img_path in sorted(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print("img2video is done")            

    #print("\nEval Outputting")
    #for i in eval_output:
    #    print(i[0],i[1],i[2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Demo for TransTrack', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
