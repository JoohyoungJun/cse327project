SpikingResFormer Vs. CNN
Name: Joohyoung Jun

1. SpikingResFormer - How to compile? 
    1-1. Local GPU:
            a. Make sure your pwd is in ../cse327project
            b. pip install -r requirements_sr.txt
            c. or, tpye in terminal 
                'pip install torch torchvision timm tensorboard spikingjelly thop cupy-cuda12x==12.2.0'
            d. run the code with 
                'python main.py -c configs/direct_training/fashionmnist.yaml --data-path ./data --output-dir ./outputs/fmnist'
    1-2. Google Colab: 
            a. Load and run 'SpikingResFormer.ipynb' in colab, 
                or do the following: 
            b. Upload and unzip cse327project to Colab
                b-1. or clone it from github with
                '!git clone https://github.com/JoohyoungJun/cse327project.git'
            c. Make sure you select runtime as Tesla T4 GPU
            d. Make sure your pwd is cse327project.
                'if not, type '%cd cse327project'
            e. install packages 
                '!pip install torch torchvision timm tensorboard spikingjelly thop'
                '!pip install cupy-cuda12x==12.2.0'
            f. run the code 
                '!python main.py -c configs/direct_training/fashionmnist.yaml --data-path ./data --output-dir ./outputs/fmnist'

2. CNN - How to compile?
    2-1. Local GPU:
        a. Make sure your pwd is in ../cse327project
        b. 'pip install -r requirements_cnn.txt'
        c. or, type in terminal
            'pip install torch torchvision tqdm'
        d. run the code with    
            'python cnn.py'
    2-2. Google Colab: 
        a. Load and run 'CNN.ipynb' in colab,
            or do the following: 
        b. Upload and unzip cse327project to colab
            b-1. or clone it from github with
                '!git clone https://github.com/JoohyoungJun/cse327project.git'
        c. Make sure you select runtime as Tesla T4 GPU.
        d. Make sure your pwd is cse327project.
            if not, type '%cd cse327project'
        e. Install packages: 
            '!pip install torch torchvision tqdm'
        f. Run the code: 
            '!python cnn.py'


