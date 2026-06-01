def get_model(config, device):
    model_config = config['MODEL']
    architecture = model_config['architecture']

    if architecture == "UNET3Df":
        from models.UNet3D.unet3df import UNet3D_CSCL
        return UNet3D_CSCL(model_config).to(device)

    if architecture == "UNET3D":
        from models.UNet3D.unet3d import UNet3D
        return UNet3D(model_config).to(device)

    if architecture == "UNET2D-CLSTM":
        from models.CropTypeMapping.models import FCN_CRNN
        return FCN_CRNN(model_config).to(device)

    if architecture == "ConvBiRNN":
        from models.BiConvRNN.biconv_rnn import BiRNNSequentialEncoder
        return BiRNNSequentialEncoder(model_config, device).to(device)

    if architecture == "TSViTcls":
        from models.TSViT.TSViTcls import TSViTcls
        model_config['device'] = device
        return TSViTcls(model_config).to(device)

    if architecture == "TSViT":
        from models.TSViT.TSViTdense import TSViT
        return TSViT(model_config).to(device)

    if architecture == "TViT":
        from models.TSViT.TSViTdense import TViT
        return TViT(model_config).to(device)

    if architecture == "STViT":
        from models.TSViT.TSViTdense import STViT
        return STViT(model_config).to(device)

    raise NameError(
        "Model architecture %s not found, choose from: "
        "'UNET3D', 'UNET3Df', 'UNET2D-CLSTM', 'ConvBiRNN', "
        "'TSViT', 'TSViTcls', 'TViT', 'STViT'" % architecture
    )
