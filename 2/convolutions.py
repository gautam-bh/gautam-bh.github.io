import numpy as np

def pad(img, shape, mode="same"):
    kh,kw = shape
    h,w = img.shape
    if mode == "full":
        top=bottom=kh-1
        left=right=kw-1
    elif mode=="same":
        top=(kh-1)//2
        bottom = kh-1-top
        left=(kw-1)//2
        right=kw-1-left
    padded = np.zeros((h+top+bottom,w+left+right))
    padded[top:top+h,left:left+w]=img
    return padded
     

def convolution4(img, kernel, mode = "same"):
    #kernel is 2d array
    K = kernel[::-1, ::-1]
    p = pad(img, kernel.shape, mode)
    H, W = img.shape
    kH, kW = kernel.shape
    H -= kH - 1
    W -= kW - 1
    result = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            sum = 0
            for h in range(kH):
                for w in range(kW):
                    sum += p[i + h, j + w] * K[h, w]
            result[i][j] = sum
    return result

def convolution(img, kernel, mode="same"):
    #kernel is 2d array
        K = kernel[::-1, ::-1]
        p = pad(img, kernel.shape, mode)
        H, W = p.shape
        kH, kW = kernel.shape
        H -= kH - 1
        W -= kW - 1
        result = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                sum = np.sum(p[i:i+kH, j:j+kW] * K)
                result[i][j] = sum
        return result