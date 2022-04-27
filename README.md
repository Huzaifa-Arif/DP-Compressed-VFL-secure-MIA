# DP-Compressed-VFL-secure-MIA
The main results are obtained from Image Inversion file.Experiments assume that the model was trained on a specific configuration of quantization and variance of noise. Pretrained models not uploaded here due to space limitations.



## Training the model
Please refer to the file quant.py for training the VFL model on quantization


## To Run the experiments
To obtain the results in the paper follow these steps:
1) Pretrain a model using the configured quantization and variance of the noise
2) Go to the image inversion file and rename the checkpoint file of the pretrained model
3) Choose the folder where the pictures are saved for testing model inversion
4) Choose the type of quantization in image inversion (top-k or scalar quantization)
5) Set variance of noise in Image Inversion

