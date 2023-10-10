import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import tifffile


def haar_img():
    img_u8 = cv2.imread("len_std.jpg")
    img_f32 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img_f32, 'haar')
    cA, (cH, cV, cD) = coeffs

    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    return img


if __name__ == '__main__':
    # img = haar_img()
    # plt.imshow(img, 'gray')
    # plt.title('img')
    # plt.show()

    path = "/ssd/0/qjy/Dataset/COVID19_CROP/study_0666.tif"
    data = tifffile.imread(path)
    print(data.shape)
    data_slice = data[15,:,:,:]
    data_slice = np.squeeze(data_slice)
    print(data_slice.shape)

    coeffs = pywt.dwt2(data_slice, 'haar')
    cA, (cH, cV, cD) = coeffs
    print(cA.shape)
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    print(img.shape)

    plt.imshow(cA)
    plt.savefig('wavelet/cA.jpg')

    plt.imshow(cH)
    plt.savefig('wavelet/cH.jpg')

    plt.imshow(cV)
    plt.savefig('wavelet/cV.jpg')

    plt.imshow(cD)
    plt.savefig('wavelet/cD.jpg')

    plt.imshow(img)
    plt.savefig('wavelet/img.jpg')



    # https://blog.csdn.net/qq_42817826/article/details/129759368
    # Dt = loadmat(filepath)
    # clean_img = Dt['clean']
    # clean_img = np.sqrt(clean_img)
    # plt.imshow(clean_img*255, cmap='gray', vmin=0, vmax=255)
    # plt.savefig('./1.jpg')
    
    # plt.show()
    # noisy_img = Dt['noisy']
    # noisy_img = np.sqrt(noisy_img)
    # plt.imshow(noisy_img*255, cmap='gray', vmin=0, vmax=255)
    # plt.savefig('./2.jpg')