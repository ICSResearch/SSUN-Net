<div align="center">

# Training Script Overview  
*(Main scripts for SSUN-Net training)*

# **[AAAI25] SSUN-Net**  
### Spatial-Spectral Prior-Aware Unfolding Network for Pan-Sharpening  

**Authors:** Shijie Fang · Hongping Gan  
📄 **AAAI Paper:** https://ojs.aaai.org/index.php/AAAI/article/view/32296

</div>

---

## Abstract
> Deep Unfolding Networks (DUNs), with their outstanding performance and partial interpretability, have revitalized the field of pan-sharpening. However, the current DUNs for pan-sharpening rely entirely on implicit deep priors, ignoring the intrinsic physical prior knowledge of multispectral image (MS) and panchromatic image (PAN) to guide the reconstruction process. Moreover, these methods often depend on single-scale prior features, failing to adequately capture multiscale information, resulting in spatial and spectral distortions in detail. In this paper, we introduce a spatial-spectral prior-aware framework for pan-sharpening, called SSPF, which formulates a constrained minimization problem integrating MS and PAN prior knowledge based on spatial and spectral domains. We further develop SSPF into a lightweight deep unfolding network, called SSUN-Net, which provides more efficient prior feature extraction and requires less computational cost. Additionally, we augment SSUN-Net's capabilities by integrating a customized multi-scale prior structure (MPS). MPS imposes constraints on the solution space at various scales through regularization, which markedly enhances the reconstruction of intricate details. Extensive experiments demonstrate the significant advantages of our proposed SSUN-Net over the current SOTA methods.

---

## 🛰 Training Scripts for Different Satellites
<div align="center">
- **WorldView-2:** `Pansharpening/TOOL_SSUN/train_wv2.py`  
- **WorldView-3:** `Pansharpening/TOOL_SSUN/train_wv3.py`  
- **Gaofen-2:** `Pansharpening/TOOL_SSUN/train_gf2.py`  
</div>
