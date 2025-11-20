# This repo to implement the VAE from scratch

download the data set:
make sure you have `vae_from_scratch/video/dataset` dir

* 1. Goes to dataset folder 
```
    cd  ~/vae_from_scratch/video/dataset
```

* 2. Download the Dataset `recommended` path 
```
hf download ProgramerSalar/video_dataset video.zip --repo-type dataset --local-dir .

```

* 3. Unzip the `video dataset` folder 
```
    unzip video.zip
```

## Download the `OpenVid-1m` Dataset 

list of dataset sizes.

```
|Dataset part   | sizes        |
|---------------|--------------|
|    part0      |  31.6GB      |
|    part1      |  41.4GB      |
|    part2      |  43.3GB      |


```


* you should replace the part Like: OpenVid_part0.zip to part1, part2 ...

```
    hf download nkp37/OpenVid-1M OpenVid_part0.zip --repo-type dataset --local-dir .
```