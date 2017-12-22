import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import imread

def imshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def sobel_filter(img):
    x_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]).T
    y_kernel = x_kernel.T
    G_x = convolve2d(img, x_kernel, mode='same')
    G_y = convolve2d(img, y_kernel, mode='same')
    return G_x, G_y  

def is_edge(x, y, I, angle_round):
    """
    Rudimentary function for |edge_thin| to
    determine whether or not the given point |x|, |y| is an edge point.
    Parameters:
    x, y - point to check
    I - 2d array of gradient intensity
    angle_round - 2d array of rounded angle values.
    See edge_thin for details.
    Returns:
    A boolean value indicating whether the given point is an edge point or not.
    """
    prod = 1
    if angle_round[y][x] == 0:
        return (I[y - 1][x] <= prod * I[y][x]) and \
               (I[y + 1][x] <= prod * I[y][x])
    if angle_round[y][x] == 1:
        return (I[y + 1][x + 1] <= prod * I[y][x]) and \
               (I[y - 1][x - 1] <= prod * I[y][x])
    elif angle_round[y][x] == 2:
        return (I[y][x + 1] <= prod * I[y][x]) and \
               (I[y][x - 1] <= prod * I[y][x])
    elif angle_round[y][x] == 3:
        return (I[y - 1][x + 1] <= prod * I[y][x]) and \
               (I[y + 1][x - 1] <= prod * I[y][x])  
    return False

def edge_thin(G_x, G_y):
    """
    Thins edges without interpolation.

    Parameters: 
    G_x, G_y - 2d arrays of gradients in 
    x, y direction.

    Returns: 2d array of boolean values without borders
    indicating whether the given point in an edge point or not.
    """
    I = (G_x ** 2 + G_y ** 2) ** 0.5
    angle = np.arctan(G_y / G_x)

    # |angle_round| will be an array of directions:
    # 0: horizontal direction
    # 1: lower right direction
    # 2: upwards direction
    # 3: upper right direction
    angle_round = np.round(angle * 4 / np.pi) % 4
    result = np.zeros_like(G_x, dtype=np.bool)

    # Process only non-border cases

    top       = (I[ :-1,  :  ] >= I[1:  ,  :  ])[1:  , 1:-1]
    bottom    = (I[ :-1,  :  ] <= I[1:  ,  :  ])[ :-1, 1:-1]

    right     = (I[ :  ,  :-1] >= I[ :  , 1:  ])[1:-1, 1:  ]
    left      = (I[ :  ,  :-1] <= I[ :  , 1:  ])[1:-1,  :-1]

    top_left  = (I[ :-1, 1:  ] >= I[1:  ,  :-1])[1:  ,  :-1]
    bot_right = (I[ :-1, 1:  ] <= I[1:  ,  :-1])[ :-1, 1:  ]

    top_right = (I[ :-1,  :-1] >= I[1:  , 1:  ])[1:  , 1:  ]
    bot_left  = (I[ :-1,  :-1] <= I[1:  , 1:  ])[ :-1,  :-1]

    result[1:-1, 1:-1][
        np.logical_and(
            np.logical_and(top, bottom), 
            angle_round[1:-1, 1:-1] == 0
        )
    ] = True

    result[1:-1, 1:-1][
        np.logical_and(
            np.logical_and(bot_left, top_right), 
            angle_round[1:-1, 1:-1] == 1
        )
    ] = True

    result[1:-1, 1:-1][
        np.logical_and(
            np.logical_and(left, right), 
            angle_round[1:-1, 1:-1] == 2
        )
    ] = True

    result[1:-1, 1:-1][
        np.logical_and(
            np.logical_and(top_left, bot_right), 
            angle_round[1:-1, 1:-1] == 3
        )
    ] = True

    return result


def edge_detect(img, sigma=1.4, window=7, gradient=sobel_filter):
    # Step 1: smoothen the image
    t = (((window - 1)/2)-0.5)/sigma  # Some magic for filter kernel size
    smoothened = gaussian_filter(img, sigma=sigma, truncate=t)
    plt.imshow(smoothened, cmap='gray')
    plt.title('Smoothened image')
    plt.show()

    # Step 2: compute image gradients
    G_x, G_y = gradient(smoothened)
    I = (G_x ** 2 + G_y ** 2) ** 0.5

    # Step 3: thin edges
    thin_map = edge_thin(G_x, G_y)

    # Step 4: Apply an intensity filter
    # to weed out noise edges.
    thin_map[I < np.mean(I)] = False

    # Step 5 (TODO): strong/weak edge filtering

    plt.imshow(thin_map, cmap='gray')
    plt.title('Edge thinning map')
    plt.show()


def main():
    filename = sys.argv[1]
    img = imread(filename, flatten=True)
    print(img.shape)
    edge_detect(img, sigma=100, gradient=sobel_filter)


if __name__ == "__main__":
    main()