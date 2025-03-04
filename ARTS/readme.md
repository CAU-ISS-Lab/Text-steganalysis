#### Dependencies
- Python 3.6
- PyTorch 1.2.0
- numpy 1.19.5
- torchtext 0.6.0
- transformers 3.3.0
- termcolor 1.1.0
- tqdm 4.47.0

### Quickstart
```
cd ./arts/bin
sh arts.sh
```
### Code
```
`src/main.py` may be run with one of two modes: `train` and `test`.
`src/model/ad_cnn.py`: Task-invariant Generator
`src/model/modelD.py`: Discriminator
`src/model/pullproto.py`: Task-relevant Projector
```
