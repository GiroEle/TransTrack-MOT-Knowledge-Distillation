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
## Knowledge Distillation Dataset Creation


# Welcome to GitHub

Welcome to GitHub—where millions of developers work together on software. Ready to get started? Let’s learn how this all works by building and publishing your first GitHub Pages website!

## Repositories

Right now, we’re in your first GitHub **repository**. A repository is like a folder or storage space for your project. Your project's repository contains all its files such as code, documentation, images, and more. It also tracks every change that you—or your collaborators—make to each file, so you can always go back to previous versions of your project if you make any mistakes.

This repository contains three important files: The HTML code for your first website on GitHub, the CSS stylesheet that decorates your website with colors and fonts, and the **README** file. It also contains an image folder, with one image file.

## Describe your project

You are currently viewing your project's **README** file. **_README_** files are like cover pages or elevator pitches for your project. They are written in plain text or [Markdown language](https://guides.github.com/features/mastering-markdown/), and usually include a paragraph describing the project, directions on how to use it, who authored it, and more.

[Learn more about READMEs](https://help.github.com/en/articles/about-readmes)

## Your first website

**GitHub Pages** is a free and easy way to create a website using the code that lives in your GitHub repositories. You can use GitHub Pages to build a portfolio of your work, create a personal website, or share a fun project that you coded with the world. GitHub Pages is automatically enabled in this repository, but when you create new repositories in the future, the steps to launch a GitHub Pages website will be slightly different.

[Learn more about GitHub Pages](https://pages.github.com/)

## Rename this repository to publish your site

We've already set-up a GitHub Pages website for you, based on your personal username. This repository is called `hello-world`, but you'll rename it to: `username.github.io`, to match your website's URL address. If the first part of the repository doesn’t exactly match your username, it won’t work, so make sure to get it right.

Let's get started! To update this repository’s name, click the `Settings` tab on this page. This will take you to your repository’s settings page. 

![repo-settings-image](https://user-images.githubusercontent.com/18093541/63130482-99e6ad80-bf88-11e9-99a1-d3cf1660b47e.png)

Under the **Repository Name** heading, type: `username.github.io`, where username is your username on GitHub. Then click **Rename**—and that’s it. When you’re done, click your repository name or browser’s back button to return to this page.

<img width="1039" alt="rename_screenshot" src="https://user-images.githubusercontent.com/18093541/63129466-956cc580-bf85-11e9-92d8-b028dd483fa5.png">

Once you click **Rename**, your website will automatically be published at: https://your-username.github.io/. The HTML file—called `index.html`—is rendered as the home page and you'll be making changes to this file in the next step.

Congratulations! You just launched your first GitHub Pages website. It's now live to share with the entire world

## Making your first edit

When you make any change to any file in your project, you’re making a **commit**. If you fix a typo, update a filename, or edit your code, you can add it to GitHub as a commit. Your commits represent your project’s entire history—and they’re all saved in your project’s repository.

With each commit, you have the opportunity to write a **commit message**, a short, meaningful comment describing the change you’re making to a file. So you always know exactly what changed, no matter when you return to a commit.

## Practice: Customize your first GitHub website by writing HTML code

Want to edit the site you just published? Let’s practice commits by introducing yourself in your `index.html` file. Don’t worry about getting it right the first time—you can always build on your introduction later.

Let’s start with this template:

```
<p>Hello World! I’m [username]. This is my website!</p>
```

To add your introduction, copy our template and click the edit pencil icon at the top right hand corner of the `index.html` file.

<img width="997" alt="edit-this-file" src="https://user-images.githubusercontent.com/18093541/63131820-0794d880-bf8d-11e9-8b3d-c096355e9389.png">


Delete this placeholder line:

```
<p>Welcome to your first GitHub Pages website!</p>
```

Then, paste the template to line 15 and fill in the blanks.

<img width="1032" alt="edit-githuboctocat-index" src="https://user-images.githubusercontent.com/18093541/63132339-c3a2d300-bf8e-11e9-8222-59c2702f6c42.png">


When you’re done, scroll down to the `Commit changes` section near the bottom of the edit page. Add a short message explaining your change, like "Add my introduction", then click `Commit changes`.


<img width="1030" alt="add-my-username" src="https://user-images.githubusercontent.com/18093541/63131801-efbd5480-bf8c-11e9-9806-89273f027d16.png">

Once you click `Commit changes`, your changes will automatically be published on your GitHub Pages website. Refresh the page to see your new changes live in action.

:tada: You just made your first commit! :tada:

## Extra Credit: Keep on building!

Change the placeholder Octocat gif on your GitHub Pages website by [creating your own personal Octocat emoji](https://myoctocat.com/build-your-octocat/) or [choose a different Octocat gif from our logo library here](https://octodex.github.com/). Add that image to line 12 of your `index.html` file, in place of the `<img src=` link.

Want to add even more code and fun styles to your GitHub Pages website? [Follow these instructions](https://github.com/github/personal-website) to build a fully-fledged static website.

![octocat](./images/create-octocat.png)

## Everything you need to know about GitHub

Getting started is the hardest part. If there’s anything you’d like to know as you get started with GitHub, try searching [GitHub Help](https://help.github.com). Our documentation has tutorials on everything from changing your repository settings to configuring GitHub from your command line.
