import torch

def SSIM(preds, target, data_range=255):
    from torchmetrics.functional.image import structural_similarity_index_measure
    return torch.sum(structural_similarity_index_measure(preds, target,data_range=data_range,reduction="none")).item()

def PSNR(pred, target,data_range=255):
    from torchmetrics.functional.image import peak_signal_noise_ratio
    return torch.sum(peak_signal_noise_ratio(pred, target,data_range=data_range,reduction="none")).item()

def ERGAS(preds, target,ratio=4):
    from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis
    return torch.sum(error_relative_global_dimensionless_synthesis(preds, target,ratio=ratio,reduction="none")).item()

def SCC(preds, target):
    from torchmetrics.functional.image import spatial_correlation_coefficient as scc
    return torch.sum(scc(preds, target, reduction="none")).item()

def SAM(preds, target):
    from torchmetrics.functional.image import spectral_angle_mapper
    return torch.sum(spectral_angle_mapper(preds, target,reduction="none")).item()

def D_lambda(preds,target):
    from torchmetrics.functional.image import spectral_distortion_index
    # from torchmetrics.image import SpectralDistortionIndex
    # preds = torch.rand([16, 3, 16, 16])
    # target = torch.rand([16, 3, 16, 16])
    # sdi = SpectralDistortionIndex()
    return torch.sum(spectral_distortion_index(preds, target,reduction="none",p=1)).item()

def D_s(preds,ms,pan):
    from torchmetrics.functional.image import spatial_distortion_index
    return torch.sum(spatial_distortion_index(preds, ms, pan,reduction="none",window_size=8)).item()

def D_s_D_lambda_qnr_tm(input_lr,input_pan,output):
    from torchmetrics.image import SpectralDistortionIndex
    from torchmetrics.image import SpatialDistortionIndex
    def qnr(D_lambda_idx, D_s_idx, alpha=1, beta=1):
        QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
        # print(img_fake)
        return QNR_idx

    input_lr, input_pan, output =input_lr,input_pan,output
    preds = input_lr
    target = output
    sdi = SpectralDistortionIndex(p=1).cuda()
    D_lambda=sdi(preds ,target)
    preds = output
    target = {
        'ms': input_lr,
        'pan': input_pan.repeat(1,4, 1, 1),
    }
    sdi = SpatialDistortionIndex(window_size=8).cuda()
    D_s=sdi(preds, target)
    q = qnr(D_lambda, D_s)
    return D_s,D_lambda,q

