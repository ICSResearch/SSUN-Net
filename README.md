# 🚀 SSUN-Net Training Scripts

> Welcome! If you’ve ever wondered how to pan-sharpen satellite imagery with style, you’re in the right place. Below is a neat little guide for running SSUN-Net on different satellites. Grab your coffee ☕, and let’s dive in.

---

## 📑 Table of Contents
1. [Official Paper](#official-paper)  
2. [Abstract](#abstract)  
3. [Training Scripts for Different Satellites](#training-scripts-for-different-satellites)  
4. [How to Run](#how-to-run)  
5. [Notes / Tips](#notes--tips)

---

## 📄 Official Paper
**[AAAI25 SSUN-Net Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32296)**  
*(Highly recommended reading if you want to impress your friends at conferences.)*

---

## 📝 Abstract
> Deep Unfolding Networks (DUNs), with their outstanding performance and partial interpretability, have revitalized the field of pan-sharpening. However, the current DUNs for pan-sharpening rely entirely on implicit deep priors, ignoring the intrinsic physical prior knowledge of multispectral image (MS) and panchromatic image (PAN) to guide the reconstruction process. Moreover, these methods often depend on single-scale prior features, failing to adequately capture multiscale information, resulting in spatial and spectral distortions in detail. In this paper, we introduce a spatial-spectral prior-aware framework for pan-sharpening, called SSPF, which formulates a constrained minimization problem integrating MS and PAN prior knowledge based on spatial and spectral domains. We further develop SSPF into a lightweight deep unfolding network, called SSUN-Net, which provides more efficient prior feature extraction and requires less computational cost. Additionally, we augment SSUN-Net's capabilities by integrating a customized multi-scale prior structure (MPS). MPS imposes constraints on the solution space at various scales through regularization, which markedly enhances the reconstruction of intricate details. Extensive experiments demonstrate the significant advantages of our proposed SSUN-Net over the current SOTA methods.  
*(Phew! That was a mouthful.)*

---

## 🛰 Training Scripts for Different Satellites
| Satellite    | Script |
|-------------|--------|
| WorldView-2 | `Pansharpening/TOOL_SSUN/train_wv2.py` |
| WorldView-3 | `Pansharpening/TOOL_SSUN/train_wv3.py` |
| Gaofen-2    | `Pansharpening/TOOL_SSUN/train_gf2.py` |

> Each script is tailored for a specific satellite. Think of it as choosing the right key for the right lock — or in our case, the right script for your precious satellite data. 🔑

---

## ⚡ How to Run
1. Make sure you have **Python 3.8** and **PyTorch 1.7.1** installed.  
2. Install any other dependencies listed in the `requirements.txt` (yes, it exists, and yes, you need it).  
3. Pick the satellite script you want to run (WorldView-2, WorldView-3, or Gaofen-2).  
4. Run it:  
```bash
python Pansharpening/TOOL_SSUN/train_wv2.py
