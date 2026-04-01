import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

poucoContraste  = cv2.imread("imagem/PoucoContraste.png")

"================== Questão 1 ======================="

def AumentarBrilhoContraste(image, beta, k):

    image = image.astype(np.float32)

    img = image + beta*255

    media = np.mean(img, axis=(0,1))

    img = k * (img - media) + media

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

brilho_02 = AumentarBrilhoContraste(poucoContraste, 0.2, 1)
brilho_04 = AumentarBrilhoContraste(poucoContraste, 0.4, 1)
brilho_06 = AumentarBrilhoContraste(poucoContraste, 0.6, 1)
brilho_08 = AumentarBrilhoContraste(poucoContraste, 0.8, 1)
brilho_10 = AumentarBrilhoContraste(poucoContraste, 1, 1)
brilho_n02 = AumentarBrilhoContraste(poucoContraste, -0.2, 1)
brilho_n04 = AumentarBrilhoContraste(poucoContraste, -0.4, 1)
brilho_n06 = AumentarBrilhoContraste(poucoContraste, -0.6, 1)
brilho_n08 = AumentarBrilhoContraste(poucoContraste, -0.8, 1)
brilho_n10 = AumentarBrilhoContraste(poucoContraste, -1, 1)
contraste_04 = AumentarBrilhoContraste(poucoContraste, 0 ,0.4)
contraste_08 = AumentarBrilhoContraste(poucoContraste, 0 ,0.8)
contraste_12 = AumentarBrilhoContraste(poucoContraste, 0 ,1.2)
contraste_16 = AumentarBrilhoContraste(poucoContraste, 0 ,1.6)
contraste_20 = AumentarBrilhoContraste(poucoContraste, 0 ,2)

exit()
def plotGrid(imagens, nomes, shape, filename="teste.png"):

    assert len(imagens) == len(nomes), "Listas devem ter mesmo tamanho"
    
    rows, cols = shape
    assert len(imagens) <= rows * cols, "Mais imagens que espaços no grid"

    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), dpi=150)
    fig.patch.set_facecolor('white')

    if rows == 1:
        axs = np.array([axs])
    if cols == 1:
        axs = axs.reshape(-1, 1)

    for i in range(rows * cols):
        r = i // cols
        c = i % cols

        if i < len(imagens):
            img = np.clip(imagens[i], 0, 255).astype(np.uint8)
            nome = nomes[i]

            # tratar RGB/BGR
            if len(img.shape) == 3:
                img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_show = img

            axs[r, c].imshow(img_show, cmap='gray' if len(img.shape) == 2 else None)
            axs[r, c].set_title(nome, fontsize=11)
        else:
            axs[r, c].axis("off")

        axs[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# plotGrid(
#     [poucoContraste,brilho_n02,brilho_n04,brilho_n06
#      ,brilho_n08,brilho_n10],
#     ["original", "-0.2", "-0.4", "-0.6", "-0.8", "-1.0"],
#     (2,3), 
#     "grid_brilho1.png")

# plotGrid(
#     [poucoContraste, brilho_02, brilho_04,
#      brilho_06, brilho_08, brilho_10],
#     ["original", "+0.2", "+0.4", "+0.6", "+0.8", "+1.0"],
#     (2,3),
#     "grid_brilho2.png"
# )

# plotGrid(
#     [poucoContraste, contraste_04, contraste_08,
#      contraste_12, contraste_16, contraste_20],
#     ["original", "0.4", "0.8", "1.2", "1.6", "2.0"],
#     (2,3),
#     "grid_contraste.png"
# )

"================== Questão 2 ======================="

def plotrgb(imagens, nomes, filename="teste_rgb.png"):
    assert len(imagens) == len(nomes), "Listas devem ter mesmo tamanho"

    n = len(imagens)

    fig, axs = plt.subplots(n, 4, figsize=(16, 4*n))

    if n == 1:
        axs = [axs]

    for i in range(n):
        img = imagens[i]
        nome = nomes[i]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axs[i][0].imshow(img_rgb)
        axs[i][0].set_title(nome)
        axs[i][0].axis("off")

        axs[i][1].hist(img_rgb[:, :, 0].ravel(), bins=256, color='r')
        axs[i][1].set_title("R")

        axs[i][2].hist(img_rgb[:, :, 1].ravel(), bins=256, color='g')
        axs[i][2].set_title("G")

        axs[i][3].hist(img_rgb[:, :, 2].ravel(), bins=256, color='b')
        axs[i][3].set_title("B")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plotgray(imagens, nomes, filename="teste.png"):
    n = len(imagens)

    fig, axs = plt.subplots(n, 2, figsize=(10, 4*n))

    # garantir que axs funcione mesmo com 1 imagem
    if n == 1:
        axs = [axs]

    for i in range(n):
        img = imagens[i]
        nome = nomes[i]

        # converter BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # imagem
        axs[i][0].imshow(img_rgb)
        axs[i][0].set_title(nome)
        axs[i][0].axis("off")

        # histograma
        axs[i][1].hist(img.ravel(), bins=256)
        axs[i][1].set_title(f"Histograma {nome}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plotgray([poucoContraste],["original"], "Histograma 1")

plotgray([brilho_02,brilho_04,brilho_06],["+ 0.2","+ 0.4","+ 0.8"], "Histograma 2")
plotgray([brilho_n02,brilho_n04,brilho_n06],["- 0.2","- 0.4","- 0.8"], "Histograma 3")

plotgray([contraste_04,contraste_08,contraste_16],["0.4","0.8","1.6"], "Histograma 4")

bonkers_b02 = AumentarBrilhoContraste(bonkers, 0.2, 1)
bonkers_b04 = AumentarBrilhoContraste(bonkers, 0.4, 1)
bonkers_b08 = AumentarBrilhoContraste(bonkers, 0.8, 1)
bonkers_c04 = AumentarBrilhoContraste(bonkers, 0, 0.4)
bonkers_c08 = AumentarBrilhoContraste(bonkers, 0, 0.8)
bonkers_c16 = AumentarBrilhoContraste(bonkers, 0, 1.6)

plotrgb([bonkers],["original"], "Histograma 5")

plotrgb([bonkers_b02,bonkers_b04,bonkers_b08],["+ 0.2","+ 0.4","+ 0.8"], "Histograma 6")
plotrgb([bonkers_c04,bonkers_c08,bonkers_c16],["0.4","0.8","1.6"], "Histograma 7")

bonkers_bcv2 = cv2.convertScaleAbs(bonkers, alpha=1, beta=0.4*255)
bonkers_ccv2 = cv2.convertScaleAbs(bonkers, alpha=0.8, beta=0)

plotrgb([bonkers_b04,bonkers_bcv2],["função original","opencv"], "Histograma 8")
plotrgb([bonkers_c08,bonkers_ccv2],["função original","opencv"], "Histograma 9")

"================== Questão 3 ======================="

def Cores1(image):

    image = image.astype(np.float32)

    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    w_dot_w = np.dot(w, w)

    Y = np.tensordot(image, w, axes=([2],[0]))
    c = np.mean(Y)

    v_parallel = (Y[..., None] / w_dot_w) * w
    v_perp = image - v_parallel

    v_parallel_c = (c / w_dot_w) * w
    img_const = v_perp + v_parallel_c

    img_const = np.clip(img_const, 0, 255)

    return img_const.astype(np.uint8)

def Cores2(image):

    image = image.astype(np.float32)

    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    Y = np.tensordot(image, w, axes=([2],[0]))
    Y_mean = np.mean(Y)

    alpha = Y_mean / (Y + 1e-8)

    img_new = image * alpha[..., None]

    img_new = np.clip(img_new, 0, 255)

    return img_new.astype(np.uint8)

# bonkers_q31 = Cores1(bonkers)
# bonkers_q32 = Cores2(bonkers)

# megamam_q31 = Cores1(megamam)
# megamam_q32 = Cores2(megamam)

# plotGrid([bonkers,bonkers_q31,bonkers_q32], ["Original", "Projeção", "Escalonamento"],(1,3), "cores1")
# plotGrid([megamam,megamam_q31,megamam_q32], ["Original", "Projeção", "Escalonamento"],(1,3), "cores2")

"================== Questão 4 ======================="

def convolution(image, filter):

    filter = np.flip(filter)

    image = image.astype(np.float32)
    altura, largura, _ = image.shape

    kH, kW = filter.shape
    pad_h = kH // 2
    pad_w = kW // 2

    extendida = np.pad(image[:, :, 0], ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    colors = np.zeros((altura, largura))

    for i in range(altura):
        for j in range(largura):
            region = extendida[i:i+kH, j:j+kW]
            colors[i, j] = np.sum(region * filter)

    newimage = np.zeros_like(image)
    newimage[:, :, 0] = colors
    newimage[:, :, 1] = colors
    newimage[:, :, 2] = colors

    newimage = np.clip(newimage, 0, 255)

    return newimage.astype(np.uint8)

blur = np.array([[1/9,1/9,1/9],
                  [1/9,1/9,1/9],
                  [1/9,1/9,1/9]])

gaussian = np.array([[0.023,  0.034,  0.038,  0.034,  0.023],
 [0.034 , 0.049 , 0.056 , 0.049 , 0.034],
 [0.038 , 0.056 , 0.063 , 0.056 , 0.038],
 [0.034 , 0.049  ,0.056 , 0.049 , 0.034],
 [0.023 , 0.034 , 0.038 , 0.034 , 0.023]])

Dx = np.array([[-1, 0, 1]])

Dy = np.array([
    [-1],
    [ 0],
    [ 1]
])

Sx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

Sy = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

kernel = np.array([
    [-2,-1,0],
    [-1, 1,1],
    [ 0, 1,2]
])

# poucoContraste_blur = convolution(poucoContraste, blur)

# poucoContraste_dx = convolution(poucoContraste, Dx)

# poucoContraste_dy = convolution(poucoContraste, Dy)

# poucoContraste_Sx = convolution(poucoContraste, Sx)

# poucoContraste_Sy = convolution(poucoContraste, Sy)

# poucoContraste_gaussian = convolution(poucoContraste, gaussian)

# chess_blur = convolution(chess, blur)

# chess_dx = convolution(chess, Dx)

# chess_dy = convolution(chess, Dy)

# chess_Sx = convolution(chess, Sx)

# chess_Sy = convolution(chess, Sy)

# chess_gaussian = convolution(chess, gaussian)

# plotGrid([poucoContraste_blur, poucoContraste_dx, poucoContraste_dy, poucoContraste_Sx, poucoContraste_Sy, poucoContraste_gaussian],
#          ["Blur", "Dx", "Dy", "Gx", "Gy", "gaussian"], (2,3), "Filtros_1")

# plotGrid([chess_blur, chess_dx, chess_dy, chess_Sx, chess_Sy, chess_gaussian],
#          ["Blur", "Dx", "Dy", "Gx", "Gy", "gaussian"], (2,3), "Filtros_2")


"================== Questão 5 ======================="

def reduceimage(image):
    newimage = image.copy()
    newimage = np.delete(newimage, np.arange(1, newimage.shape[0], 2), axis=0)
    newimage = np.delete(newimage, np.arange(1, newimage.shape[1], 2), axis=1)
    return newimage

# cutpoucoContraste = reduceimage(poucoContraste)
# cutpoucoContraste2 = reduceimage(cutpoucoContraste)
# cutpoucoContraste3 = reduceimage(cutpoucoContraste2)

# cut_megamam = reduceimage(megamam)
# cut_megamam2 = reduceimage(cut_megamam)
# cut_megamam3 = reduceimage(cut_megamam2)

# print(megamam.shape)
# print(cut_megamam.shape)
# print(cut_megamam2.shape)
# print(cut_megamam3.shape)

# cut_bonkers = reduceimage(bonkers)
# cut_bonkers2 = reduceimage(cut_bonkers)
# cut_bonkers3 = reduceimage(cut_bonkers2)

# plotGrid([poucoContraste, cutpoucoContraste, cutpoucoContraste2, cutpoucoContraste3], ["original", "/2", "/4", "/8"],(1,4), "Questão_5_1")
# plotGrid([megamam, cut_megamam, cut_megamam2, cut_megamam3],["original", "/2", "/4", "/8"], (1,4),"Questão_5_2")
# plotGrid([bonkers, cut_bonkers, cut_bonkers2, cut_bonkers3], ["original", "/2", "/4", "/8"],(1,4), "Questão_5_3")