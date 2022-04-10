import os.path
from os import path
import cv2
import numpy as np
from tqdm import tqdm 
from scipy.sparse.linalg import svds

# directory where videos are stored
dataDir = "."

# Data files should be .avi's and have the following form:
# '<dataDir><dataFilePrefix><startingFileNum>.avi'

# Values users can modify:
filename = '22-04-01-1403_scope_1.avi' # '22-04-01-1403_scope_'
# filename = 'AlreadyDone/21-11-05-1450_scope_1.avi' # '22-04-01-1403_scope_'
max_frames = 25000
# -------------------------------

# TODO: Grab frames per file from metadate

data_file = os.path.join(dataDir,filename)
assert(os.path.exists(data_file))

# # Read in a single frame from the file to get the dimensions 
# cap = cv2.VideoCapture(data_file)
# ret, _frame = cap.read()
# if (ret is False):
#     raise(IndexError('Read past end of file looking for frame'))
# frame = _frame[:,:,1]

# # Frame dimensions
# dx, dy = frame.shape


def generate_fft_mask(dx, dy, shape='pie_slice'):
    assert(dx==dy) # TODO: Mask code assumes square
    mid = dy//2

    if shape == 'hamming':
        # Mask 1: Hamming window stripe around zero in the vertical axis, 
        #   with vertical DC components let through, also using a Hamming window
        hamming_stripe_width = 3 # single sided width
        dc_width = 30
        col_mask = np.zeros((dx,dy), np.float32)
        col_mask[:,(mid-hamming_stripe_width):(mid+hamming_stripe_width+1)] += np.hamming(2*hamming_stripe_width+1) # ridge

        row_mask = np.ones((dx,dy), np.float32)
        row_mask[:,(mid-dc_width):(mid+dc_width+1)] -= np.hamming(2*dc_width+1) # let DC through

        mask = 1 - (col_mask * row_mask.T)

    elif shape == 'sharp':
        # Mask 2: Same as hamming window, but sharp edges
        stripe_width = 3 # single sided width
        dc_width = 30
        col_mask = np.zeros((dx,dy), np.float32)
        col_mask[:,(mid-stripe_width):(mid+stripe_width+1)] += 1 # ridge

        row_mask = np.ones((dx,dy), np.float32)
        row_mask[:,(mid-dc_width):(mid+dc_width+1)] -= 1 # let DC through

        mask = 1 - (col_mask * row_mask.T)

    elif shape == 'pie_slice':
        # Mask 3: Pie slice filter similar to https://link.springer.com/article/10.1007/s00367-012-0293-z
        angle_width = 5 # degrees
        dc_pass = 15
        # row_mask[:,(mid-dc_width):(mid+dc_width+1)] -= np.hamming(2*dc_width+1) # let DC through
        mask = np.ones((dx,dy), np.float32)
        for yi in range(dx):
            for xi in range(dy):
                if np.sqrt((xi-mid)**2 + (yi-mid)**2) > dc_pass:
                    angle = np.arctan2((xi - mid),(yi - mid))
                    awidth = np.deg2rad(angle_width/2)
                    if (-awidth < angle < awidth) or (angle > np.pi - awidth) or (angle < -np.pi + awidth):
                        mask[yi,xi] = 0

    else:
        raise(ValueError('Unknown mask shape "{}".'.format(shape)))


    opencv_mask = np.zeros((dx, dy, 2))
    opencv_mask[:,:,0] = mask
    opencv_mask[:,:,1] = mask
    return opencv_mask



# Load full video into memory
frame_data = []

print('Loading file into memory.')
cap = cv2.VideoCapture(data_file)

mask = None

for frameNum in tqdm(range(0, max_frames, 1), total = max_frames, 
                     desc ="Loading file {}".format(data_file)):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    ret, frame = cap.read()
    if (ret is False):
        break
    frame = frame[:,:,1]
    dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
    dft_shift = np.fft.fftshift(dft)

    if mask is None:
        mask = generate_fft_mask(*frame.shape)

    masked_dft = dft_shift * mask
    f_ishift = np.fft.ifftshift(masked_dft)
    img_back = cv2.idft(f_ishift)

    # img_back = img_back[:,:,0] # We started with a real signal, so we want a real signal
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    frame_data.append(img_back)

T = len(frame_data)
dx, dy = frame_data[0].shape

print('Convert list to numpy array for svd.')

frame_data_m = np.array(frame_data)
del frame_data # let's try to clear up some memory!!!

print('Running SVD.')
# Find the decomposition of the first singular value of the movie.
# This should correspond to the flicker that we see!
u2, s2, v2 = svds(frame_data_m.reshape(T,dx*dy) - np.mean(frame_data_m,axis=0).reshape(1,dx*dy), k=1)

# Apply FFT spatial filtering and lowpass filtering to data and potentially save as a new video

# Values users can modify:
# # Select one below -
# mode = "display"
mode = 'save'

# Select one below -
compressionCodec = "FFV1"
# compressionCodec = "GREY"
# --------------------

print('Writing back to disk.')

codec = cv2.VideoWriter_fourcc(compressionCodec[0],compressionCodec[1],compressionCodec[2],compressionCodec[3])

if (mode == "save" and not path.exists(dataDir + "Denoised")):
    os.mkdir(dataDir + "Denoised")

if (mode == "save"):
    write_file_name = os.path.join(dataDir, 'Denoised', 
         os.path.splitext(os.path.basename(data_file))[0] + '_denoised.avi')
    writeFile = cv2.VideoWriter(write_file_name,codec, 60, (dy,dx), isColor=False) 

for frameNum in tqdm(range(0, T, 1), total = T, desc ="Processing file".format(data_file)):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)        
        svd_noise_model = np.reshape(u2[frameNum]*s2*v2, (dx, dy))
        img_back = frame_data_m[frameNum,:,:] - svd_noise_model
        # img_back = frame_data_m[frameNum,:,:]
        img_back[img_back >255] = 255
        img_back[img_back <0] = 0
        img_back = np.uint8(img_back)

        if (mode == "save"):
            writeFile.write(img_back)


if (mode == "save"):
    writeFile.release()


