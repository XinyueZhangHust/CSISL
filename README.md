## Rethinking the Influence of Distribution Adjustment in Incremental Segmentation
This is the implementation of CSISL. 

Title of the paper: Rethinking the Influence of Distribution Adjustment in Incremental Segmentation.

## News
December 16, 2024: Submit the paper to TMM，model codes and some results are released 🔥

## Requirements
All experiments in this paper are done with the following environments:

- python 3.6.13
- charset-normalizer  2.0.12
- cycler              0.11.0
- dataclasses         0.8
- decorator           4.4.2
- einops              0.4.1
- filelock            3.4.1
- huggingface-hub     0.4.0
- idna                3.8
- importlib-metadata  4.8.3
- importlib-resources 5.4.0
- Jinja2              3.0.3
- joblib              1.1.1
- jsonpatch           1.32
- jsonpointer         2.3
- kiwisolver          1.3.1
- MarkupSafe          2.0.1
- matplotlib          3.3.4
- networkx            2.5.1
- numpy               1.19.2
- opencv-python       4.3.0.38
- packaging           21.3
- Pillow              8.4.0
- pip                 21.3.1
- pyparsing           3.1.4
- pyproject           1.3.1
- python-dateutil     2.9.0.post0
- PyYAML              6.0.1
- requests            2.27.1
- scikit-learn        0.24.2
- scipy               1.5.4
- setuptools          59.6.0
- six                 1.16.0
- threadpoolctl       3.1.0
- timm                0.6.12
- torch               1.7.1+cu110
- torchaudio          0.7.2
- torchvision         0.8.2+cu110
- tornado             6.1
- tqdm                4.64.1
- typing_extensions   4.1.1
- urllib3             1.26.20
- visdom              0.2.4
- websocket-client    1.3.1
- wheel               0.37.1
- zipp                3.6.0

## Dataset preparing

Organize datasets in the following structure.
```
path_to_your_dataset/
    VOC2012/
        Annotations/
        ImageSet/
        JPEGImages/
        SegmentationClassAug/
        proposal100/
        
    ADEChallengeData2016/
        annotations/
            training/
            validation/
        images/
            training/
            validation/
        proposal_adetrain/
        proposal_adeval/
```
You can get [proposal100](https://drive.google.com/file/d/1FxoyVa0I1IEwtW2ykGlNf-JkOYkK80E6/view?usp=sharing), [proposal_adetrain](https://drive.google.com/file/d/1kWfPNhoUnYz0uPuHJUALxiqvVqlCKrwW/view?usp=sharing), [proposal_adeval](https://drive.google.com/file/d/16xNMO4siqJXr5A03ywQDXU0F1Ld5OFtw/view?usp=sharing) here.

## Startup

We provide a training script `script_train.py` to facilitate the use of our proposed method. The script enables users to easily train CSISL with various settings, for example, the default config of CSISL is: 
```
 cd tools 
  python -u script_train.py 15-1 0,1,2,3,4,5 0 --batch_size 8 --val_batch_size 8 --freeze_low --lr 0.0001  --mem_size 100 --conloss_sparsity --conloss_compression --KDLoss --KDLoss_prelogit  --name swin_voc2012_best --unknown --dataset voc 
```
If you want to evaluate the model after training, add `--test_only`.

The trained .pth file is available in the ./checkpoints directory.



