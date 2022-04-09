
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import cv2



IMAGE_SIZE = 256
low_file_path="./user_case/low/"
high_file_path="./user_case/high/"
low_image_name=os.listdir(low_file_path)
high_image_name=os.listdir(high_file_path)
#%%
def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

def colorRestoration(img, alpha, beta):

    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):    

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img    

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):

    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)       

    return img_msrcr

#%%  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def selective_kernel_feature_fusion(
    multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3
):
    channels = list(multi_scale_feature_1.shape)[-1]
    combined_feature = layers.Add()(
        [multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3]
    )
    gap = layers.GlobalAveragePooling2D()(combined_feature)
    channel_wise_statistics = tf.reshape(gap, shape=(-1, 1, 1, channels))
    compact_feature_representation = layers.Conv2D(
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(channel_wise_statistics)
    feature_descriptor_1 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_2 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_3 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_1 = multi_scale_feature_1 * feature_descriptor_1
    feature_2 = multi_scale_feature_2 * feature_descriptor_2
    feature_3 = multi_scale_feature_3 * feature_descriptor_3
    aggregated_feature = layers.Add()([feature_1, feature_2, feature_3])
    return aggregated_feature

def spatial_attention_block(input_tensor):
    average_pooling = tf.reduce_max(input_tensor, axis=-1)
    average_pooling = tf.expand_dims(average_pooling, axis=-1)
    max_pooling = tf.reduce_mean(input_tensor, axis=-1)
    max_pooling = tf.expand_dims(max_pooling, axis=-1)
    concatenated = layers.Concatenate(axis=-1)([average_pooling, max_pooling])
    feature_map = layers.Conv2D(1, kernel_size=(1, 1))(concatenated)
    feature_map = tf.nn.sigmoid(feature_map)
    return input_tensor * feature_map


def channel_attention_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    average_pooling = layers.GlobalAveragePooling2D()(input_tensor)
    feature_descriptor = tf.reshape(average_pooling, shape=(-1, 1, 1, channels))
    feature_activations = layers.Conv2D(
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(feature_descriptor)
    feature_activations = layers.Conv2D(
        filters=channels, kernel_size=(1, 1), activation="sigmoid"
    )(feature_activations)
    return input_tensor * feature_activations


def dual_attention_unit_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    feature_map = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(input_tensor)
    feature_map = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(
        feature_map
    )
    channel_attention = channel_attention_block(feature_map)
    spatial_attention = spatial_attention_block(feature_map)
    concatenation = layers.Concatenate(axis=-1)([channel_attention, spatial_attention])
    concatenation = layers.Conv2D(channels, kernel_size=(1, 1))(concatenation)
    return layers.Add()([input_tensor, concatenation])


def down_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    main_branch = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(main_branch)
    main_branch = layers.MaxPooling2D()(main_branch)
    main_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.MaxPooling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])


def up_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    main_branch = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(main_branch)
    main_branch = layers.UpSampling2D()(main_branch)
    main_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.UpSampling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])


# MRB Block
def multi_scale_residual_block(input_tensor, channels):
    # features
    level1 = input_tensor
    level2 = down_sampling_module(input_tensor)
    level3 = down_sampling_module(level2)
    # DAU
    level1_dau = dual_attention_unit_block(level1)
    level2_dau = dual_attention_unit_block(level2)
    level3_dau = dual_attention_unit_block(level3)
    # SKFF
    level1_skff = selective_kernel_feature_fusion(
        level1_dau,
        up_sampling_module(level2_dau),
        up_sampling_module(up_sampling_module(level3_dau)),
    )
    level2_skff = selective_kernel_feature_fusion(
        down_sampling_module(level1_dau), level2_dau, up_sampling_module(level3_dau)
    )
    level3_skff = selective_kernel_feature_fusion(
        down_sampling_module(down_sampling_module(level1_dau)),
        down_sampling_module(level2_dau),
        level3_dau,
    )
    # DAU 2
    level1_dau_2 = dual_attention_unit_block(level1_skff)
    level2_dau_2 = up_sampling_module((dual_attention_unit_block(level2_skff)))
    level3_dau_2 = up_sampling_module(
        up_sampling_module(dual_attention_unit_block(level3_skff))
    )
    # SKFF 2
    skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)
    conv = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(skff_)
    return layers.Add()([input_tensor, conv])

"""### MIRNet Model"""

def recursive_residual_group(input_tensor, num_mrb, channels):
    conv1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_mrb):
        conv1 = multi_scale_residual_block(conv1, channels)
    conv2 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(conv1)
    return layers.Add()([conv2, input_tensor])


def mirnet_model(num_rrg, num_mrb, channels):
    input_tensor = keras.Input(shape=[None, None, 3])
    x1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_rrg):
        x1 = recursive_residual_group(x1, num_mrb, channels)
    conv = layers.Conv2D(3, kernel_size=(3, 3), padding="same")(x1)
    output_tensor = layers.Add()([input_tensor, conv])
    return keras.Model(input_tensor, output_tensor)

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

#MIRNET_model = load_model('MIRNET.h5',custom_objects={'charbonnier_loss': charbonnier_loss,'peak_signal_noise_ratio':peak_signal_noise_ratio})

#%%  
from keras.layers import Input, Lambda
import keras.backend as K
import keras.layers as KL

channel_axis = 1 if K.image_data_format() == "channels_first" else 3
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])
from keras.layers import Input, Lambda
def build_dce_net():
   input_img = keras.Input(shape=[None, None, 3])
   input_img = keras.Input(shape=[None, None, 3])
   conv11 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(input_img)
   conv1 =cbam_module(conv11)
   conv21 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv1)
   conv2 =cbam_module(conv21)
   conv31 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv2)
   conv3 =cbam_module(conv31)
   conv41 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv3)
   conv4 =cbam_module(conv41)
   int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
   conv51 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con1)
   conv5 =cbam_module(conv51)
   int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
   conv61 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con2)
   conv6 =cbam_module(conv61)
   int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
   x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(
        int_con3
    
    )
   return keras.Model(inputs=input_img, outputs=x_r)

class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")

    def get_enhanced_image(self, data, output):
        global out_tensor
        out_tensor=output.numpy()
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

zero_dce_model = ZeroDCE()
zero_dce_model.load_weights('ZERODEC.h5')  

#%% 
for i in range(len(low_image_name)):

    low_image_path=low_file_path+low_image_name[i]
    high_image_path=high_file_path+high_image_name[i]
    print(low_image_path,high_image_path)
    plt.figure(str(i),figsize=(50, 16), dpi=80)
    
    image = cv2.imread(low_image_path)    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
    image = keras.preprocessing.image.img_to_array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    # MIRNET_output_image=MIRNET_model(image)
    # MIRNET_output_image = tf.cast((MIRNET_output_image[0, :, :, :] * 255), dtype=np.uint8)
    # MIRNET_output_image = Image.fromarray(MIRNET_output_image.numpy())
    
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())

    ground_truth=image = cv2.imread(high_image_path)
    ground_truth=cv2.cvtColor(ground_truth,cv2.COLOR_BGR2RGB)
    
    ground_truth_low=image = cv2.imread(low_image_path)
    ground_truth_low=cv2.cvtColor(ground_truth_low,cv2.COLOR_BGR2RGB)
    
    (b,g,r) = cv2.split(ground_truth_low) #通道分解
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
    bH=clahe.apply(b)
    gH=clahe.apply(g)
    rH=clahe.apply(r)
    HE = cv2.merge((bH,gH,rH),)#通道合成
    
    size = 3
    # b_gray, g_gray, r_gray = cv2.split(ground_truth_low)
    # b_gray = SSR(b_gray, size)
    # g_gray = SSR(g_gray, size)
    # r_gray = SSR(r_gray, size)
    # SSR_result = cv2.merge([b_gray, g_gray, r_gray])
    SSR_result =img_msrcr = MSRCR(
        ground_truth_low,
        [15, 80, 200],
        5.0,
        25.0,
        125.0,
        46.0,
        0.01,
        0.99
    )
    
    plt.subplot(261)  
    plt.axis('off')
    plt.title('Ground Truth \n In Low Illumination',fontsize=36,fontweight='heavy') 
    plt.imshow(ground_truth_low)
    
    # plt.subplot(267)  
    # color = ('b','g','r')
    # legend=["1","2","3"]
    # for i,col in enumerate(color):
    #   histr = cv2.calcHist([ground_truth_low],[i],None,[256],[0,256])
    #   plt.plot(histr,color = col)
    # plt.xlim([0,256])
    # plt.legend(["B","G","R"],fontsize=32,loc =1)
    # plt.title('Histogram',fontsize=36,fontweight='heavy') 
    # plt.grid()  
    
    
    
    plt.subplot(262) 
    plt.axis('off')
    plt.title('Ground Truth \n In Normal Illumination',fontsize=36,fontweight='heavy') 
    plt.imshow(ground_truth)
    
    # plt.subplot(268)  
    # color = ('b','g','r')
    # legend=["1","2","3"]
    # for i,col in enumerate(color):
    #   histr = cv2.calcHist([ground_truth],[i],None,[256],[0,256])
    #   plt.plot(histr,color = col)
    # plt.xlim([0,256])
    # plt.legend(["B","G","R"],fontsize=32,loc =1)
    # plt.title('Histogram',fontsize=36,fontweight='heavy') 
    # plt.grid()  
    
    plt.subplot(263) 
    plt.axis('off')
    plt.title('CLAHE',fontsize=36,fontweight='heavy') 
    plt.imshow(HE)
    
    # plt.subplot(269)  
    # color = ('b','g','r')
    # legend=["1","2","3"]
    # for i,col in enumerate(color):
    #   histr = cv2.calcHist([HE],[i],None,[256],[0,256])
    #   plt.plot(histr,color = col)
    # plt.xlim([0,256])
    # plt.legend(["B","G","R"],fontsize=32,loc =1)
    # plt.title('Histogram',fontsize=36,fontweight='heavy') 
    # plt.grid()  
    
    plt.subplot(264) 
    plt.axis('off')
    plt.title('MSRCR',fontsize=36,fontweight='heavy') 
    plt.imshow(SSR_result)
    
    # plt.subplot(2,6,10)  
    # color = ('b','g','r')
    # legend=["1","2","3"]
    # for i,col in enumerate(color):
    #   histr = cv2.calcHist([SSR_result],[i],None,[256],[0,256])
    #   plt.plot(histr,color = col)
    # plt.xlim([0,256])
    # plt.legend(["B","G","R"],fontsize=32,loc =1)
    # plt.title('Histogram',fontsize=36,fontweight='heavy') 
    # plt.grid()  
    
    plt.subplot(265) 
    plt.axis('off')
    plt.title('MIRNET',fontsize=36,fontweight='heavy') 
    # plt.imshow(MIRNET_output_image)
    
    # plt.subplot(2,6,11)  
    # color = ('b','g','r')
    # legend=["1","2","3"]
    # for i,col in enumerate(color):
    #   histr = cv2.calcHist([np.array(MIRNET_output_image)],[i],None,[256],[0,256])
    #   plt.plot(histr,color = col)
    # plt.xlim([0,256])
    # plt.legend(["B","G","R"],fontsize=32,loc =1)
    # plt.title('Histogram',fontsize=36,fontweight='heavy') 
    # plt.grid()  
    
    plt.subplot(266)   
    plt.axis('off')
    plt.title('Our Method',fontsize=36,fontweight='heavy') 
    plt.imshow(output_image)
    
    # plt.subplot(2,6,12)  
    # color = ('b','g','r')
    # legend=["1","2","3"]
    # for i,col in enumerate(color):
    #   histr = cv2.calcHist([np.array(output_image)],[i],None,[256],[0,256])
    #   plt.plot(histr,color = col)
    # plt.xlim([0,256])
    # plt.legend(["B","G","R"],fontsize=32,loc =1)
    # plt.title('Histogram',fontsize=36,fontweight='heavy') 
    # plt.grid()  
    
    plt.tight_layout()
    plt.savefig(str(i)+".png")
    plt.show()
   
    plt.close('all')
    
#%% 


import time
import math
from skimage.metrics import structural_similarity
low_file_path="./lol_dataset/eval15/low/"
high_file_path="./lol_dataset/eval15/high/"
low_image_name=os.listdir(low_file_path)
high_image_name=os.listdir(high_file_path)
PSNR_dce=[]
PSNR_AHE=[]
PSNR_SSR=[]
PSNR_MIRNET=[]

TIME_dce=[]
TIME_AHE=[]
TIME_SSR=[]
TIME_MIRNET=[]

MAE_dce=[]
MAE_AHE=[]
MAE_SSR=[]
MAE_MIRNET=[]

SSIM_dce=[]
SSIM_AHE=[]
SSIM_SSR=[]
SSIM_MIRNET=[]

for i in range(len(low_image_name)):

    low_image_path=low_file_path+low_image_name[i]
    high_image_path=high_file_path+high_image_name[i]
   
    
    original = cv2.imread(low_image_path) 
    original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)  
    original = keras.preprocessing.image.img_to_array(original)
    original = original.astype("float32") / 255.0
    original = np.expand_dims(original, axis=0)
    
    reference= cv2.imread(high_image_path) 
    
    t1=time.time()
    contrast_dce = zero_dce_model(original)
    contrast_dce = tf.cast((contrast_dce[0, :, :, :] * 255), dtype=np.uint8)
    contrast_dce = Image.fromarray(contrast_dce.numpy())
    t2=time.time()
    TIME_dce.append(t2-t1)
    
    t1=time.time()
    (b,g,r) = cv2.split(cv2.imread(low_image_path)) #通道分解
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
    bH=clahe.apply(b)
    gH=clahe.apply(g)
    rH=clahe.apply(r)
    contrast_HE = cv2.merge((bH,gH,rH),)#通道合成 
    t2=time.time()
    TIME_AHE.append(t2-t1)
    
    t1=time.time()
    size = 3
    # b_gray, g_gray, r_gray = cv2.split(cv2.imread(low_image_path))
    # b_gray = SSR(b_gray, size)
    # g_gray = SSR(g_gray, size)
    # r_gray = SSR(r_gray, size)
    # contrast_SSR = cv2.merge([b_gray, g_gray, r_gray])
    contrast_SSR =img_msrcr = MSRCR(
        cv2.imread(low_image_path),
        [15, 80, 200],
        5.0,
        25.0,
        125.0,
        46.0,
        0.01,
        0.99
    )
    t2=time.time()
    TIME_SSR.append(t2-t1)
    
    t1=time.time()
    # MIRNET_output_image=MIRNET_model(original)
    # MIRNET_output_image = tf.cast((MIRNET_output_image[0, :, :, :] * 255), dtype=np.uint8)
    # contrast_MIRNET = Image.fromarray(MIRNET_output_image.numpy())
    t2=time.time()
    TIME_MIRNET.append(t2-t1)
    
    
    MAE_dce.append(np.mean( abs(contrast_dce - reference)))
    MAE_AHE.append(np.mean( abs(contrast_HE - reference)))
    MAE_SSR.append(np.mean( abs(contrast_SSR - reference)))
    # MAE_MIRNET.append(np.mean( abs(contrast_MIRNET - reference)))
    
    PSNR_dce.append((tf.image.psnr(reference,contrast_dce, max_val=255.0)).numpy())
    PSNR_AHE.append((tf.image.psnr(reference,contrast_HE, max_val=255.0)).numpy())
    PSNR_SSR.append((tf.image.psnr(reference,contrast_SSR, max_val=255.0)).numpy())
    # PSNR_MIRNET.append((tf.image.psnr(reference,contrast_MIRNET, max_val=255.0)).numpy())
    
   
    SSIM_dce.append(structural_similarity(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),cv2.cvtColor(np.asarray(contrast_dce), cv2.COLOR_BGR2GRAY), win_size=7, full=True)[0])
    SSIM_AHE.append(structural_similarity(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),cv2.cvtColor(np.asarray(contrast_HE), cv2.COLOR_BGR2GRAY), win_size=7, full=True)[0])
    SSIM_SSR.append(structural_similarity(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),cv2.cvtColor(np.asarray(contrast_SSR), cv2.COLOR_BGR2GRAY), win_size=7, full=True)[0])
    # SSIM_MIRNET.append(structural_similarity(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),cv2.cvtColor(np.asarray(contrast_MIRNET), cv2.COLOR_BGR2GRAY), win_size=7, full=True)[0])


print('psnr',np.mean(PSNR_dce),np.mean(PSNR_AHE),np.mean(PSNR_SSR),np.mean(PSNR_MIRNET))
print('MAE',np.mean(MAE_dce),np.mean(MAE_AHE),np.mean(MAE_SSR),np.mean(MAE_MIRNET))
print('SSIM',np.mean(SSIM_dce),np.mean(SSIM_AHE),np.mean(SSIM_SSR),np.mean(SSIM_MIRNET))
print('TIME',round(np.mean(TIME_dce)*1000,2),round(np.mean(TIME_AHE)*1000,2),round(np.mean(TIME_SSR)*1000,2),round(np.mean(TIME_MIRNET)*1000,2))
#%% 

result=list(
[['resut','DCE','AHE','SSR','MIRNET'],
['psnr',np.mean(PSNR_dce),np.mean(PSNR_AHE),np.mean(PSNR_SSR),np.mean(PSNR_MIRNET)],
['MAE',np.mean(MAE_dce),np.mean(MAE_AHE),np.mean(MAE_SSR),np.mean(MAE_MIRNET)],
['SSIM',np.mean(SSIM_dce),np.mean(SSIM_AHE),np.mean(SSIM_SSR),np.mean(SSIM_MIRNET)],
['TIME',round(np.mean(TIME_dce)*1000,2),round(np.mean(TIME_AHE)*1000,2),round(np.mean(TIME_SSR)*1000,2),round(np.mean(TIME_MIRNET)*1000,2)]])

import csv
with open("Result.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerows(result)




 
