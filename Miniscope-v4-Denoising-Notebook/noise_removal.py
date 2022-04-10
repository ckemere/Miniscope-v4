import os.path
from os import path
import cv2
import numpy as np
from tqdm import tqdm 
from scipy.signal import butter, lfilter, freqz, filtfilt

# directory where videos are stored
dataDir = "."

# Data files should be .avi's and have the following form:
# '<dataDir><dataFilePrefix><startingFileNum>.avi'

# Values users can modify:
# filename = '22-04-01-1403_scope_1.avi' # '22-04-01-1403_scope_'
filename = 'AlreadyDone/21-11-05-1450_scope_1.avi' # '22-04-01-1403_scope_'
max_frames = 25000
# -------------------------------

# TODO: Grab frames per file from metadate

data_file = os.path.join(dataDir,filename)
assert(os.path.exists(data_file))


# Read in a single frame from the file to get the dimensions 
cap = cv2.VideoCapture(data_file)
ret, _frame = cap.read()
if (ret is False):
    raise(IndexError('Read past end of file looking for frame'))
frame = _frame[:,:,1]

# Frame dimensions
_r, _c = frame.shape
mid = _c//2

# Mask 1: Hamming window stripe around zero in the vertical axis, 
#   with vertical DC components let through, also using a Hamming window
hamming_stripe_width = 3 # single sided width
dc_width = 30
col_mask = np.zeros((_r,_c), np.float32)
col_mask[:,(mid-hamming_stripe_width):(mid+hamming_stripe_width+1)] += np.hamming(2*hamming_stripe_width+1) # ridge

row_mask = np.ones((_r,_c), np.float32)
row_mask[:,(mid-dc_width):(mid+dc_width+1)] -= np.hamming(2*dc_width+1) # let DC through

hamming_mask = 1 - (col_mask * row_mask.T)

# Mask 2: Same as hamming window, but sharp edges
hamming_stripe_width = 3 # single sided width
dc_width = 30
col_mask = np.zeros((_r,_c), np.float32)
col_mask[:,(mid-hamming_stripe_width):(mid+hamming_stripe_width+1)] += 1 # ridge

row_mask = np.ones((_r,_c), np.float32)
row_mask[:,(mid-dc_width):(mid+dc_width+1)] -= 1 # let DC through

sharp_mask = 1 - (col_mask * row_mask.T)

# Mask 3: Pie slice filter similar to https://link.springer.com/article/10.1007/s00367-012-0293-z
angle_width = 5 # degrees
dc_pass = 15
# row_mask[:,(mid-dc_width):(mid+dc_width+1)] -= np.hamming(2*dc_width+1) # let DC through
pie_slice_mask = np.ones((_r,_c), np.float32)
for yi in range(_r):
    for xi in range(_c):
        if np.sqrt((xi-mid)**2 + (yi-mid)**2) > dc_pass:
            angle = np.arctan2((xi - mid),(yi - mid))
            awidth = np.deg2rad(angle_width/2)
            if (-awidth < angle < awidth) or (angle > np.pi - awidth) or (angle < -np.pi + awidth):
                pie_slice_mask[yi,xi] = 0

# Load full video into memory
frame_data = []

cap = cv2.VideoCapture(data_file)

mask = np.zeros((pie_slice_mask.shape[0], pie_slice_mask.shape[1], 2))
mask[:,:,0] = pie_slice_mask
mask[:,:,1] = pie_slice_mask

for frameNum in tqdm(range(0, max_frames, 1), total = max_frames, 
                     desc ="Loading file {}".format(data_file)):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    ret, frame = cap.read()
    if (ret is False):
        break
    frame = frame[:,:,1]
    dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
    dft_shift = np.fft.fftshift(dft)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    frame_data.append(img_back)

frame_data_m = np.array(frame_data)

T = len(frame_data)
dx, dy = frame_data[0].shape

# Find the decomposition of the first singular value of the movie.
# This should correspond to the flicker that we see!
u2, s2, v2 = svds(frame_data_m.reshape(T,dx*dy) - np.mean(frame_data,axis=0).reshape(1,dx*dy), k=1)


# Apply FFT spatial filtering and lowpass filtering to data and potentially save as a new video

# Values users can modify:
# Select one below -
# mode = "display"
mode = 'save'


# Select one below -
compressionCodec = "FFV1"
# compressionCodec = "GREY"
# --------------------

running = True


codec = cv2.VideoWriter_fourcc(compressionCodec[0],compressionCodec[1],compressionCodec[2],compressionCodec[3])

if (mode is "save" and not path.exists(dataDir + "Denoised")):
    os.mkdir(dataDir + "Denoised")


if (mode is "save"):
    write_file_name = os.path.join(dataDir, 'Denoiseed', 
         os.path.splitext(os.path.basename(data_file))[0] + '_denoised.avi')
    writeFile = cv2.VideoWriter(write_file_name,codec, 60, (_c,_r), isColor=False) 

for frameNum in tqdm(range(0, len(frame_data), 1), total = len(frame_data), 
                           desc ="Processing file".format(data_file)):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)        
        svd_noise_model = np.reshape(u2[frameNum]*s2*v2, (dx, dy))
        img_back = frame_data[frameNum] - svd_noise_model
        img_back[img_back >255] = 255
        img_back = np.uint8(img_back)

        if (mode is "save"):
            writeFile.write(img_back)

        # if (mode is "display"):
        #     im_diff = (128 + (frame - img_back)*2)
        #     im_v = cv2.hconcat([frame, img_back])
        #     im_v = cv2.hconcat([im_v, im_diff])

        #     im_v = cv2.hconcat([frame, img_back, im_diff])
        #     cv2.imshow("Cleaned video", im_v/255)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         running = False
        #         cap.release()
        #         break


if (mode is "save"):
    writeFile.release()

cv2.destroyAllWindows()

