# Absolute Coordinates Make Motion Generation Easy (arXiv 2025)
![](./ACMDM.png)

<p align="center">
  <a href='https://arxiv.org/abs/2505.19377'>
    <img src='https://img.shields.io/badge/Arxiv-2505.19377-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/abs/2505.19377.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://neu-vi.github.io/ACMDM/'>
  <img src='https://img.shields.io/badge/Project-Page-orange?style=flat&logo=Google%20chrome&logoColor=orange'></a>
  <a href='https://github.com/neu-vi/ACMDM'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=neu-vi.ACMDM&left_color=gray&right_color=blue">
  </a>
</p>

<p align="center">
<strong>Absolute Coordinates Make Motion Generation Easy</strong></h1>
   <p align="center">
    <a href='https://cr8br0ze.github.io' target='_blank'>Zichong Meng</a>&emsp;
    <a href='https://show-han.github.io/' target='_blank'>Zeyu Han</a>&emsp;
    <a href='https://xiaogangpeng.github.io/' target='_blank'>Xiaogang Peng</a>&emsp;
    <a href='https://ymingxie.github.io/' target='_blank'>Yiming Xie</a>&emsp;
    <a href='https://jianghz.me/' target='_blank'>Huaizu Jiang</a>&emsp;
    <br>
    Northeastern University 
    <br>
    arXiv 2025
  </p>
</p>

### Official Simple & Minimalist PyTorch Implementation

## 📢 News
- Codes will be released in the next few weeks due to extensive supports and cleaning on Text-to-Motion, Controllable Generation, and direct text-to-SMPL-H vertices motion generation.

## 📜 TODO List
- [ ] Release the clean codes for implementation.
- [ ] Release the evaluation codes and the pretrained models.
- [ ] Release the simple and minimalist version of codes for implementation.


## 🍀 Acknowledgments
This code is standing on the shoulders of giants, we would like to thank the following contributors that our code is based on:.

Our original raw implementation is heavily based on [T2M](https://github.com/EricGuo5513/text-to-motion),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MMM](https://github.com/exitudio/MMM) 
and [MoMask](https://github.com/EricGuo5513/momask-codes).
The Diffusion part is primarily based on [DDPM](https://github.com/hojonathanho/diffusion),
[DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT),
[MAR](https://github.com/LTH14/mar/), [HOI-Diff](https://github.com/neu-vi/HOI-Diff),
[InterGen](https://github.com/tr3e/InterGen), [MDM](https://github.com/GuyTevet/motion-diffusion-model),
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion).

For open sourced version, we decide to restructure (and some rewrite) for a simple and minimalist version of PyTorch code implementation
that get rids of PyTorch Lighting implicit hooks, outer-space variable utilization and implicit argparse calls.
We hope our minimalist version implementation can lead to better code comprehension and contribution to the motion generation community. Thank you.

## 🤝 Citation
If you find this repository useful for your work, please consider citing it as follows:
```bibtex
@article{meng2025absolute,
    title={Absolute Coordinates Make Motion Generation Easy},
    author={Meng, Zichong and Han, Zeyu and Peng, Xiaogang and Xie, Yiming and Jiang, Huaizu},
    journal={arXiv preprint arXiv:2505.19377},
    year={2025}
  }
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neu-vi/ACMDM&type=Date)](https://www.star-history.com/#neu-vi/ACMDM&Date)
