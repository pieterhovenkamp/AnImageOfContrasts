#!/usr/bin/env python3

from tensorflow import keras

model_dict = {"mobilenet_v2_100_192":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/5",
                   "img_size": 192},
              "mobilenet_v2_100_160":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/5",
                   "img_size": 160},
              "mobilenet_v2_100_128":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/5",
                   "img_size": 128},
              "mobilenet_v2_100_96":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5",
                   "img_size": 96},
              "mobilenet_v2_075_192":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/feature_vector/5",
                   "img_size": 192},
              "mobilenet_v2_075_160":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/feature_vector/5",
                   "img_size": 160},
              "mobilenet_v2_075_128":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/feature_vector/5",
                   "img_size": 128},
              "mobilenet_v2_075_96":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/5",
                   "img_size": 96},
              "mobilenet_v2_050_192":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/5",
                   "img_size": 192},
              "mobilenet_v2_050_160":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/5",
                   "img_size": 160},
              "mobilenet_v2_050_128":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/5",
                   "img_size": 128},
              "mobilenet_v2_050_96":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/5",
                   "img_size": 96},
              "mobilenet_v3_large_100":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
                   "img_size": 224},
              "mobilenet_v3_small_100":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
                   "img_size": 224},
              "mobilenet_v3_large_075":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
                   "img_size": 224},
              "mobilenet_v3_small_075":
                  {"link": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
                   "img_size": 224},
              "inception_v3_imnet":
                  {"link": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
                   "img_size": 299},
              "inception_v3_inat":
                  {"link": "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5",
                   "img_size": 299},
              "resnet_152":
                  {"link": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5",
                   "img_size": 224},
              "resnet_101":
                  {"link": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5",
                   "img_size": 224},
              "resnet_050":
                  {"link": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
                   "img_size": 224},
              "efficientnet_v2_224":
                  {"link": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
                   "img_size": 224},
              "efficientnet_v2_240":
                  {"link": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
                   "img_size": 240},
              "efficientnet_v2_260":
                  {"link": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
                   "img_size": 260},
              "efficientnet_v2_300":
                  {"link": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
                   "img_size": 300},
              "inceptionresnet_v2":
                  {"link": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",
                   "img_size": 299},
              "deit_tiny_patch16_224_fe":
                  {"link": "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224_fe/1",
                   "img_size": 224},
              "deit_base_distilled_patch16_384":
                  {"link": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1",
                   "img_size": 384},
              "deit_base_distilled_patch16_224":
                  {"link": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1",
                   "img_size": 224},
              "deit_base_distilled_patch16_224_fe":
                  {"link": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224_fe/1",
                   "img_size": 224},
              "Xception":
                  {"link": keras.applications.Xception,
                   "img_size": 299,
                   "preprocessing_func": None},
              "VGG16":
                  {"link": keras.applications.VGG16,
                   "img_size": 224,
                   "preprocessing_func": None},
              "VGG19":
                  {"link": keras.applications.VGG19,
                   "img_size": 224,
                   "preprocessing_func": None},
              "ResNet50V2":
                  {"link": keras.applications.ResNet50V2,
                   "img_size": 224,
                   "preprocessing_func": None},
              "ResNet101V2":
                  {"link": keras.applications.ResNet101V2,
                   "img_size": 224,
                   "preprocessing_func": None},
              "ResNet152V2":
                  {"link": keras.applications.ResNet152V2,
                   "img_size": 224,
                   "preprocessing_func": None},
              "InceptionV3":
                  {"link": keras.applications.InceptionV3,
                   "img_size": 299,
                   "preprocessing_func": None},
              "InceptionResNetV2":
                  {"link": keras.applications.InceptionResNetV2,
                   "img_size": 299,
                   "preprocessing_func": None},
              "MobileNetV2":
                  {"link": keras.applications.MobileNetV2,
                   "img_size": 224,
                   "preprocessing_func": None},
              "DenseNet121":
                  {"link": keras.applications.DenseNet121,
                   "img_size": 224,
                   "preprocessing_func": None},
              "DenseNet169":
                  {"link": keras.applications.DenseNet169,
                   "img_size": 224,
                   "preprocessing_func": None},
              "DenseNet201":
                  {"link": keras.applications.DenseNet201,
                   "img_size": 224,
                   "preprocessing_func": None},
              "NASNetMobile":
                  {"link": keras.applications.NASNetMobile,
                   "img_size": 224,
                   "preprocessing_func": None},
              "NASNetLarge":
                  {"link": keras.applications.NASNetLarge,
                   "img_size": 331,
                   "preprocessing_func": None},
              "EfficientNetV2B0":
                  {"link": keras.applications.EfficientNetV2B0,
                   "img_size": 224,
                   "preprocessing_func": None},
              "EfficientNetV2B1":
                  {"link": keras.applications.EfficientNetV2B1,
                   "img_size": 240,
                   "preprocessing_func": None},
              "EfficientNetV2B2":
                  {"link": keras.applications.EfficientNetV2B2,
                   "img_size": 260,
                   "preprocessing_func": None},
              "EfficientNetV2B3":
                  {"link": keras.applications.EfficientNetV2B3,
                   "img_size": 300,
                   "preprocessing_func": None},
              "EfficientNetV2S":
                  {"link": keras.applications.EfficientNetV2S,
                   "img_size": 384,
                   "preprocessing_func": None},
              }

N_COLOR_CHANNELS = 3