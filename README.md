# Speech Enhancement Auto-Encoder

## Results:
|                | PESQ          | SSNR     |
|---------------:|--------------:|---------:|
| Noisy          | 1.97          | 1.68     |
|  SEAE          | 2.2           | 7.6      |




## Dependencies:
* Tensorflow 1.14

## Data:
Please download the [VCTK dataset](https://drive.google.com/file/d/1NBIOCk1ouXqi_cY-XxH9_cDTftVYXYAR/view?usp=sharing)
## Usage:

### Train

```
./train.sh
```
**L1 loss**
<img src="loss/loss.png" width="650">
### Test

```
./inference.sh
```
