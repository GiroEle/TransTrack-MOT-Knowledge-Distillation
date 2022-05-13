## Description
Files and instructions necessary to use and reproduce the MOT knowledge distillation method I present.
Free Space: 7GB

## Original TrasTrack Documentation
https://github.com/PeizeSun/TransTrack
-	Use my steps below
-	Important: ignore the installation/requirements section. Use below:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
-	My repo comments out certain things that create errors. Mainly to do with torch
-	Lacking explanation of program structure. Will fill in other relevant information below

## Additional Documentation
Setup guide:
1. Install and build libs
```
python>=3.7
Download my TransTrack folder. For the rest of instructions, assumes operating in this folder.
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
(^ later versions probably work)

pip install -r requirements.txt
python models/ops/setup.py build --build-base=models/ops/ install
```
2. Prepare datasets and annotations
```
https://motchallenge.net/
download MOT17, copy contents (test and train) folders into a new “./mot” folder.
Python track_tools/convert_mot_to_coco.py
```
Additional Notes:
-	Large model used is the “671mot17_crowdhuman_mot17.pth” in original documentation.
-	Supplied KD and NO_KD trained small model as "small_KD_checkpoint0149.pth" "small_noKD_checkpoint0149.pth" "small_KD_checkpoint0249.pth"
-	points 4 5 6 of original documentation might be useful.
-	Notes section of original documentation for training time for a specific compute.

## MOT Model
Main_Track.py

  Input: parameters that specific model, notably --KD if training using model generated annotations. --resume to continue checkpoint. Parameters taken from arguments passed in (example with batch size and epochs of 1). Parameters explained in thesis.
  
  Output: in ./output checkpoint of trained model. Also log of training progress. Checkpoints every 5 epochs.
  
  Example:
  - ‘Large’ model’s training call
```
python .\main_track.py  --output_dir ./output --dataset_file mot --coco_path mot --batch_size 1 --with_box_refine --epochs 1 --lr_drop 100 --num_queries 500
```
  - ‘Small’ model’s training call (not KD)
     
    - If KD training add “--KD”. Changes what annotations is used in training.

```
python .\main_track.py  --output_dir ./output --dataset_file mot --coco_path mot --batch_size 1 --with_box_refine --epochs 1 --lr_drop 100 --enc_layers 1 --dec_layers 1 --backbone resnet18 --num_feature_levels 1 --hidden_dim 64 --dim_feedforward 512 --num_queries 250
```
  Explanation (including other files called by main_track.py):

* Main_track.py Structure:

  - build_tracktest_model(args), build_tracktrain_model(args) build model each for different situations (eval vs training). Inside models folder and paths to specific files.
  
    models\deformable_detrtrack_train.py 		function: build(args)
    
    models\deformable_detrtrack_test.py 		function: build(args)
    
  - evaluate, train_one_epoch from engine_track.py

- engine_track.py


- models\deformable_detrtrack_train.py
 models\deformable_detrtrack_test.py 	

These modules each create the overall TransTrack model, that calls relevant modules to produce output. The model is created from the DeformableDETR class.

- DeformableDETR/forward_train 

  explains how the model gets the desired output. Difference with Test version includes the use of a pre_embed (previous inferences information).

  The following modules are incorporated in a nested module list.
  -	models\backbone.py
    -	torch load used to get object detection backbone
    -	intermediate layers are used as output
    -	positional embedding attached
    -	Requires some hardcoding to accommodate different backbone selection (see resnet18 in file; line 77).
  -	models\deformable_transformer_track.py
    -	transformer used
    
-	./datasets/mot.py build function

  - Where dataset is created based on json annotations.

-	./models/Save_track.py

  - How MOT tracks are saved.

-	./models/Tracker.py

  - TransTrack is largely object detection, this section allows a MOT result to be produced and maintained from frame to frame.

## Timing + Image/Video Evaluation

demo_videosim.py 

- input: model parameters and checkpoint (--resume) that specify a trained model. Hyper parameter choice of image downsize. “--video_input” control video input.

- Output: log text file and print out of real-time FPS performance of model on “demo.mp4” video. Over 50 frames of inference. Generates MOT image and video output with desired checkpoint. Stored in “./demo_output” folder.


demo_videosim_testing OUT.py 

- Continuation of previous. Removed features to produce more accurate timing evaluation. This code does not produce the MOT video+image outputs. Computation that delays the model’s inference, even while not counted, still affected its FPS.

Example:
```
python .\demo_videosim_testing OUT.py --max_size 800 --video_input demo.mp4 --resume ./output/small_KD_checkpoint0249.pth --output_dir ./output --dataset_file mot --coco_path mot --batch_size 1 --with_box_refine --epochs 1 --lr_drop 100 --enc_layers 1 --dec_layers 1 --backbone resnet18 --num_feature_levels 1 --hidden_dim 64 --dim_feedforward 512 --num_queries 250
```

## Evaluation for MOT metrics
**./track_tools/eval_motchallenge.py**

- Input: The tracks created after an evaluation operation (--eval) of main_track.py (eg: ./output/val/tracks). Specified which groundtruth annotations to compare to. Original documentation explains with sh mota.sh. 

- Output: MOT performance metrics.

- Complete evaluation with the original dataset regardless of KD (specified by --groundtruths)

```
python ./track_tools/eval_motchallenge.py --groundtruths mot/train --tests output/val/tracks --gt_type _val_half --eval_official --score_threshold -1
```
  - --tests output/val/tracks  is changed to the track output folder

- eval and building videoframes with 
```
python main_track.py  --output_dir . --dataset_file mot --coco_path mot --batch_size 1 --resume ./output/checkpoint.pth --eval --with_box_refine --num_queries 500
```
```
python track_tools/txt2video.py 
```
```
py track_tools/eval_motchallenge.py --groundtruths mot/train --tests val/tracks --gt_type _val_half --eval_official --score_threshold -1
```


## Knowledge Distillation Dataset Creation

**./main_track_AnnGen_Train.py**
**./main_track_AnnGen_Val.py**

Input: Takes a trained model from the “resume” arg and parameters relevant to the model.

Output: New tracking output for training/validation dataset, stored in ./output/KD_eval and ./output/KD_eval_val (opposed to default TransTrack storing validation results in val)

Example:

```
python main_track_AnnGen_Train.py --eval --num_workers 1 --resume ./671mot17_crowdhuman_mot17.pth --output_dir ./output --dataset_file mot --coco_path mot --batch_size 1 --with_box_refine --epochs 1 --lr_drop 100 --num_queries 500
```

```
python main_track_AnnGen_Val.py --eval --num_workers 1 --resume ./671mot17_crowdhuman_mot17.pth --output_dir ./output --dataset_file mot --coco_path mot --batch_size 1 --with_box_refine --epochs 1 --lr_drop 100 --num_queries 500
```

Explanation:

- Uses the training/val dataset and creates MOT annotations. Based on ./main_track.py evaluation operation. This originally takes the trained model and creates MOT annotations for the validation dataset. Altered to match real-time input for train dataset as well (inserted sequentially).

- The normal eval operation outputs tracks in output/val folder. This is changed to KD_eval and KD_eval_val for main_track_AnnGen_Train.py and main_track_AnnGen_Val.py. Both will be called “val” folder for simplicity.

- Inside of output/val/tracks there is text files containing all annotations for each video (set of frames). The contents of each row in the text files are information about each tracked object detection in each frame:

  - frame_id, track_id, bbox_top_left_X, bbox_top_left_Y, bbox_width, bbox_height, -1, -1, -1, -1
![image](https://user-images.githubusercontent.com/56175932/168344516-22b3aa9b-aa6f-4f4f-b275-c2245bea3dbf.png)
![image](https://user-images.githubusercontent.com/56175932/168344520-41861de8-8248-44f4-9220-3377d8d7a7e6.png)

**./building_dataset_annotations.py**

input: Requires 1) tracks inside text files, generated from running the previous code in the location it created “./output/KD_eval/tracks/” and "./output/KD_eval_val/tracks/". 2) Copy the “.\mot\annotations” folder to create a duplicate “.\mot\annotations_KD”.

output: Dataset in format required by TransTrack with new annotations in “.\mot\annotations_KD”. This means updating the "./mot/annotations_KD/val_half.json" and "./mot/annotations_KD/train_half.json" which store the annotations.

Example:
```
python building_dataset_annotations.py
```

Explanation:
- Just overwrites annotation jsons with new information.

**Training KD model**

Use ./main_track.py with --KD parameter to use the KD dataset in .\mot\annotations_KD.
