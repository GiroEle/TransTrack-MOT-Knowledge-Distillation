from __future__ import annotations
import json
import numpy as np

#with open("./output/KD_eval/score_area_track_results.json",'r') as f:
#    KD_data_train = json.load(f)

# t[0], rename_track_id, t[1], t[2], t[3] - t[1], t[4] - t[2],t[5],t[6],t[3]*t[4]))
#                 bbox = [bbox[0], bbox[1], bbox[2], bbox[3], item['score'], item['active']]
# tracks[tracking_id].append([frame_id] + bbox)

# frame_id, track_id, bbox_TLx, bbox_TLy,bbox_wid,bbox_high, scrore, active, area
#  
paths_KD=[ 
    "2_MOT17-02-FRCNN.txt",
    "2_MOT17-04-FRCNN.txt",
    "2_MOT17-05-FRCNN.txt",
    "2_MOT17-09-FRCNN.txt",
    "2_MOT17-10-FRCNN.txt",
    "2_MOT17-11-FRCNN.txt",
    "2_MOT17-13-FRCNN.txt"   
]

#offset because of train_half setup skipping sections

ID_frame_id_train = np.genfromtxt('./mot/annotations_KD/ID_frame_id_train.csv',delimiter=',',dtype=int)
ID_frame_id_val = np.genfromtxt('./mot/annotations_KD/ID_frame_id_val.csv',delimiter=',',dtype=int)


#these minus -1 to get the map (what needs to be added to frame_id to get overall image_id)
#index by video id
ID_map_train=[
    1,
    601,
    1651,
    2488,
    3013,
    3667,
    4567
]

ID_map_val=[
    302,
    1127,
    2070,
    2751,
    3341,
    4118,
    4943
]

path = ["./output/KD_eval/tracks/",
        "./output/KD_eval_val/tracks/"]

annotations_update_train=[]
data_ann_train=[]
x=0

for i,paths_KD_i in enumerate(paths_KD,start=1):
    print(i,path[0]+paths_KD_i)
    data_train_i=np.loadtxt(path[0]+paths_KD_i, delimiter=',')
    for data_train_i_j in data_train_i:
        #print(i,data_train_i_j[0],ID_map_train[i-1])

        #active modes
        if data_train_i_j[11]==1:
            id_i=data_train_i_j[0] -1 + ID_map_train[i-1]
            x+=1
            data_ann_train.append([id_i,data_train_i_j[1], data_train_i_j[2],data_train_i_j[3],data_train_i_j[4],data_train_i_j[5],data_train_i_j[10]])
            annotations_update_train.append( {
                'id': x,
                'category_id':1,
                'image_id':id_i,
                'track_id':data_train_i_j[1],
                'bbox':list(data_train_i_j[2:6]),
                'conf':data_train_i_j[10],
                'iscrowd':0,
                'area': round(data_train_i_j[4]*data_train_i_j[5],2)
            })
        #print(id_i)

annotations_update_val=[]
for i,paths_KD_i in enumerate(paths_KD,start=1):
    print(i,path[1]+paths_KD_i)
    data_val_i=np.loadtxt(path[1]+paths_KD_i, delimiter=',')
    for data_val_i_j in data_val_i:
        #print(i,data_val_i[0],ID_map_train[i-1])

        #active modes
        if data_val_i_j[11]==1:
            id_i=data_val_i_j[0] -1 + ID_map_val[i-1]
            x+=1
            data_ann_train.append([id_i,data_val_i_j[1], data_val_i_j[2],data_val_i_j[3],data_val_i_j[4],data_val_i_j[5],data_val_i_j[10]])
            annotations_update_val.append( {
                'id': x,
                'category_id':1,
                'image_id':id_i,
                'track_id':data_val_i_j[1],
                'bbox':list(data_val_i_j[2:6]),
                'conf':data_val_i_j[10],
                'iscrowd':0,
                'area': round(data_val_i_j[4]*data_val_i_j[5],2)
            })

#with open("./output/KD_eval_val/score_area_track_results.json",'r') as f:
#    KD_data_val = json.load(f)

with open("./mot/annotations_KD/train_half.json",'r+') as f:
    dataset_train = json.load(f)
    dataset_train['annotations']=annotations_update_train
    f.seek(0)
    json.dump(dataset_train, f, indent=4)
    f.truncate()

with open("./mot/annotations_KD/val_half.json",'r+') as f:
    dataset_val = json.load(f)
    dataset_val['annotations']=annotations_update_val
    f.seek(0) 
    json.dump(dataset_val, f, indent=4)
    f.truncate()

print("******************* IF USED IN FUTURE MAKE SURE TO CHANGE AREA RECALULATION THAT WAS FIXED *********************************")

print("done")