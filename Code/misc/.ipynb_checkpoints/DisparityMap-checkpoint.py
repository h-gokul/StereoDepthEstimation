import numpy as np
from tqdm import tqdm

def SSD(imL_window, imR_window):
    """
    Args:
        imL_window : image patch from left image
        imR_window : image patch from right image
    Returns:
        float: Sum of square difference between individual pixels
    """
    if imL_window.shape != imR_window.shape:
        return -1

    return np.sum(np.square(imL_window - imR_window))

    


def DisparityMap(imL_rectified, imR_rectified, warpedlinesL, warpedlinesR, win_size = 7, searchRange = 56 ):
    """
    Args:
        imL_window : image patch from left image
        imR_window : image patch from right image
    Returns:
        float: Sum of square difference between individual pixels
    """
    imL_rectified = imL_rectified.astype(np.int32)
    imR_rectified = imR_rectified.astype(np.int32)
    if imL_rectified.shape != imR_rectified.shape:
        raise "Left-Right image shape mismatch!"
    height, width = imL_rectified.shape
    disparity_map = np.zeros((height, width))
#     print(disparity_map.shape)

    # Go over each epipolar Line 
#     for i in tqdm(range(len(warpedlinesL))):
#         y = int(np.mean([warpedlinesL[i][:,1], warpedlinesR[i][:,1]]))
    # Go over each pixel
    for y in tqdm(range(win_size, height-win_size)):
        for x in range(win_size, width - win_size):
            imL_window = imL_rectified[y:y + win_size, x:x + win_size]
            x_ = BlockMatching(y, x, imL_window, imR_rectified, win_size, searchRange)
            
            disparity_map[y, x] = np.abs(x_ - x)
    return disparity_map

def BlockMatching(y, x, imL_window, imR_rectified, block_size=7, searchRange = 56):
    wR = imR_rectified.shape[1]
    
    # set up searchRange for given block
    x_start = max(0, x - searchRange)
    x_end = min(wR, x + searchRange)
#     x_start = x    
#     x_end = wR
    
    # initialise variables
    startFlag = True
    min_ssd, min_index = None, None

    for x in range(x_start, x_end):
        imR_window = imR_rectified[y: y+block_size,x: x+block_size]
        
        ssd = SSD(imL_window, imR_window)

        if startFlag:
            min_ssd = ssd

            min_x = x
            startFlag = False
        else:
            if ssd < min_ssd:
                min_ssd = ssd
                min_x = x
    return min_x


####################################################################
# def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
#     """
#     Args:
#         pixel_vals_1 (numpy.ndarray): pixel block from left image
#         pixel_vals_2 (numpy.ndarray): pixel block from right image
#     Returns:
#         float: Sum of absolute difference between individual pixels
#     """
#     if pixel_vals_1.shape != pixel_vals_2.shape:
#         return -1

#     return np.sum(abs(pixel_vals_1 - pixel_vals_2))


# def compare_blocks(y, x, block_left, right_array, block_size=5, SEARCH_BLOCK_SIZE = 56):
#     """
#     Compare left block of pixels with multiple blocks from the right
#     image using SEARCH_BLOCK_SIZE to constrain the search in the right
#     image.
#     Args:
#         y (int): row index of the left block
#         x (int): column index of the left block
#         block_left (numpy.ndarray): containing pixel values within the 
#                     block selected from the left image
#         right_array (numpy.ndarray]): containing pixel values for the 
#                      entrire right image
#         block_size (int, optional): Block of pixels width and height. 
#                                     Defaults to 5.
#     Returns:
#         tuple: (y, x) row and column index of the best matching block 
#                 in the right image
#     """
#     # Get search range for the right image
#     x_min = max(0, x - SEARCH_BLOCK_SIZE)
#     x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
#     #print(f'search bounding box: ({y, x_min}, ({y, x_max}))')
#     first = True
#     min_sad = None
#     min_index = None
#     for x in range(x_min, x_max):
#         block_right = right_array[y: y+block_size,
#                                   x: x+block_size]
#         sad = sum_of_abs_diff(block_left, block_right)
#         #print(f'sad: {sad}, {y, x}')
#         if first:
#             min_sad = sad
#             min_index = (y, x)
#             first = False
#         else:
#             if sad < min_sad:
#                 min_sad = sad
#                 min_index = (y, x)

#     return min_index


# def get_disparity_map(left_array, right_array, BLOCK_SIZE = 7, SEARCH_BLOCK_SIZE = 56):

#     left_array = left_array.astype(int)
#     right_array = right_array.astype(int)
#     if left_array.shape != right_array.shape:
#         raise "Left-Right image shape mismatch!"
#     h, w = left_array.shape
#     # left_im = cv2.imread("data/left.png", 0)
#     disparity_map = np.zeros((h, w))
#     # Go over each pixel position
#     for y in tqdm(range(BLOCK_SIZE, h-BLOCK_SIZE)):
#         for x in range(BLOCK_SIZE, w-BLOCK_SIZE):
#             block_left = left_array[y:y + BLOCK_SIZE,
#                                     x:x + BLOCK_SIZE]
#             min_index = compare_blocks(y, x, block_left,
#                                        right_array,
#                                        block_size=BLOCK_SIZE, SEARCH_BLOCK_SIZE = SEARCH_BLOCK_SIZE  )
#             disparity_map[y, x] = abs(min_index[1] - x)

#     return disparity_map
    