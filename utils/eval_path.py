import torchvision.models as models

imagenet_models = [
    models.resnet18, models.resnet34, models.resnet50, models.resnet101,
    models.wide_resnet50_2, models.wide_resnet101_2,
    models.vgg11, models.vgg11_bn, models.vgg16, models.vgg16_bn,
    models.mobilenet_v2, models.shufflenet_v2_x1_0,
    models.densenet121, models.densenet169, models.densenet201,
    models.vit_b_32, models.vit_b_16, models.vit_l_32, models.vit_l_16,
    models.swin_t, models.swin_s, models.swin_v2_t, models.swin_v2_s, 
]

cifar10_models = [
    'cifar10_resnet20', 'cifar10_resnet32', 'cifar10_resnet44', 'cifar10_resnet56',
    'cifar10_vgg11_bn', 'cifar10_vgg13_bn', 'cifar10_vgg16_bn', 'cifar10_vgg19_bn',
    'cifar10_mobilenetv2_x0_5', 'cifar10_mobilenetv2_x0_75', 'cifar10_mobilenetv2_x1_0', 'cifar10_mobilenetv2_x1_4',
    'cifar10_shufflenetv2_x0_5', 'cifar10_shufflenetv2_x1_0', 'cifar10_shufflenetv2_x1_5', 'cifar10_shufflenetv2_x2_0',
    'cifar10_repvgg_a0', 'cifar10_repvgg_a1', 'cifar10_repvgg_a2',
]

cifar100_models = [
    'cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44', 'cifar100_resnet56',
    'cifar100_vgg11_bn', 'cifar100_vgg13_bn', 'cifar100_vgg16_bn', 'cifar100_vgg19_bn',
    'cifar100_mobilenetv2_x0_5', 'cifar100_mobilenetv2_x0_75', 'cifar100_mobilenetv2_x1_0', 'cifar100_mobilenetv2_x1_4',
    'cifar100_shufflenetv2_x0_5', 'cifar100_shufflenetv2_x1_0', 'cifar100_shufflenetv2_x1_5', 'cifar100_shufflenetv2_x2_0',
    'cifar100_repvgg_a0', 'cifar100_repvgg_a1', 'cifar100_repvgg_a2',
]