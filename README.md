# Patch-Mix Contrastive Learning (INTERSPEECH 2023)
[arXiv](https://arxiv.org/abs/2305.14032) | [BibTeX](#bibtex)


<p align="center">
<img width="300" src="https://github.com/raymin0223/patch-mix_contrastive_learning/assets/46586785/767a14f8-0c17-4f2c-9e44-ce41d59bf7fa">
</p>

**Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification**<br/>
[Sangmin Bae](https://www.raymin0223.com)\*,
[June-Woo Kim](https://github.com/kaen2891)\*,
[Won-Yang Cho](https://github.com/wonyangcho),
[Hyerim Baek](https://github.com/rimiyeyo),
[Soyoun Son](https://github.com/soyounson),
[Byungjo Lee](https://github.com/bzlee-bio),
[Changwan Ha](https://github.com/cwh1981),
[Kyongpil Tae](https://github.com/kyongpiltae),
[Sungnyun Kim](https://github.com/sungnyun)$^\dagger$,
[Se-Young Yun](https://fbsqkd.github.io)$^\dagger$ <br/>
\* equal contribution &nbsp;&nbsp; $^\dagger$ corresponding authors

- We demonstrate that the **pretrained model on large-scale visual and audio datasets** can be generalized to the respiratory sound classification task.
- We introduce a straightforward **Patch-Mix augmentation**, which randomly mixes patches between different samples, with Audio Spectrogram Transformer (AST).
- To overcome the label hierarchy in lung sound datasets, we propose an effective **Patch-Mix Contrastive Learning** to distinguish the mixed representations in the latent space.


## Requirements
Install the necessary packages with: 
```
$ pip install torch torchvision torchaudio
$ pip install -r requirements.txt
```


## Data Preparation
Download the ICBHI dataset files from [official_page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).     
```bash
$ wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```
All `*.wav` and `*.txt` should be saved in `data/icbhi_dataset/audio_test_data`.     

Note that ICBHI dataset consists of a total of 6,898 respiratory cycles, 
of which 1,864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.


## Training 
To simply train the model, run the shell files in `scripts/`.    
1. **`scripts/icbhi_ce.sh`**: Cross-Entropy loss with AST model.
2. **`scripts/icbhi_patchmix_ce.sh`**: Patch-Mix loss with AST model, where the label depends on the interpolation ratio.
3. **`scripts/icbhi_patchmix_cl.sh`**: Patch-Mix contrastive loss with AST model.

Important arguments for different data settings.
- `--dataset`: other lungsound datasets or heart sound can be implemented
- `--class_split`: "lungsound" or "diagnosis" classification
- `--n_cls`: set number of classes as 4 or 2 (normal / abnormal) for lungsound classification
- `--test_fold`: "official" denotes 60/40% train/test split, and "0"~"4" denote 80/20% split

Important arguments for models.
- `--model`: network architecture, see [models](models/)
- `--from_sl_official`: load ImageNet pretrained checkpoint
- `--audioset_pretrained`: load AudioSet pretrained checkpoint and only support AST and SSAST

Important arugment for evaluation.
- `--eval`: switch mode to evaluation without any training
- `--pretrained`: load pretrained checkpoint and require `pretrained_ckpt` argument.
- `--pretrained_ckpt`: path for the pretrained checkpoint

The pretrained model checkpoints will be saved at `save/[EXP_NAME]/best.pth`.     

## Result

Patch-Mix Contrastive Learning achieves the state-of-the-art performance of 62.37%, which is higher than previous Score by +4.08%.
<p align="center">
<img width="750" src="https://github.com/raymin0223/patch-mix_contrastive_learning/assets/50742281/2a1d8b4c-b46d-423b-adbe-1d43334e7b7d">
</p>


## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{bae2023patch,
  title={Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification},
  author={Bae, Sangmin and Kim, June-Woo and Cho, Won-Yang and Baek, Hyerim and Son, Soyoun and Lee, Byungjo and Ha, Changwan and Tae, Kyongpil and Kim, Sungnyun and Yun, Se-Young},
  journal={arXiv preprint arXiv:2305.14032},
  year={2023}
}
```

## Contact
- Sangmin Bae: bsmn0223@kaist.ac.kr
- June-Woo Kim: kaen2891@knu.ac.kr
