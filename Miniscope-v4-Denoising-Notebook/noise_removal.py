import os.path
import cv2
import numpy as np
from tqdm import tqdm 
from scipy.sparse.linalg import svds

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




def remove_noise(input_filename, remove_stripes=True, remove_flicker=True,
                 output_directory='Denoised', output_suffix='_denoised',
                 compression_codec = "FFV1",
                 max_frames=25000):

    if not (compression_codec in ['FFV1', 'GREY']):
        raise(ValueError('Unsupported compression_codec {}'.format(compression_codec)))

    if (not remove_stripes) and (not remove_flicker):
        raise(ValueError('No noise removal chosen! (neither remove_stripes and/or remove_flicker)'))

    assert(os.path.exists(input_filename))

    frame_data = [] # This will hold the whole video
    # print('Loading file {} into memory.'.format(input_filename))
    cap = cv2.VideoCapture(input_filename)

    mask = None
    for frameNum in tqdm(range(0, max_frames, 1), total = max_frames, 
                        desc ="Loading file {}".format(data_file)):
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        ret, frame = cap.read()
        if (ret is False):
            break
        frame = frame[:,:,1]

        if remove_stripes:
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
        else:
            frame_data.append(frame)

    T = len(frame_data)
    dx, dy = frame_data[0].shape

    # print('Convert list to numpy array for svd.')
    frame_data_m = np.array(frame_data)
    del frame_data # let's try to clear up some memory!!!

    # Find the decomposition of the first singular value of the movie.
    # This should correspond to the flicker that we see!
    # print('Running SVD.')
    if remove_flicker:
        u2, s2, v2 = svds(frame_data_m.reshape(T,dx*dy) - np.mean(frame_data_m,axis=0).reshape(1,dx*dy), k=1)

    # Apply FFT spatial filtering and lowpass filtering to data and potentially save as a new video

    # print('Writing back to disk.')
    codec = cv2.VideoWriter_fourcc(compression_codec[0],compression_codec[1],
                                   compression_codec[2],compression_codec[3])

    input_dir = os.path.dirname(input_filename)
    if output_directory is not None:
        if not os.path.isabs(output_directory): # output directory not specified as a full path
            output_directory = os.path.join(input_dir, output_directory)
        if not os.path.exists(output_directory):
            os.mkdir(os.path.join(output_directory))

    write_file_name = os.path.join(output_directory,  
        os.path.splitext(os.path.basename(data_file))[0] + '{}.avi'.format(output_suffix))

    writeFile = cv2.VideoWriter(write_file_name,codec, 60, (dy,dx), isColor=False) 

    for frameNum in tqdm(range(0, T, 1), total = T, desc ="Processing file".format(data_file)):
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)   
        if remove_flicker:     
            svd_noise_model = np.reshape(u2[frameNum]*s2*v2, (dx, dy))
            img_back = frame_data_m[frameNum,:,:] - svd_noise_model
        else:
            img_back = frame_data_m[frameNum,:,:]
        img_back[img_back >255] = 255
        img_back[img_back <0] = 0
        img_back = np.uint8(img_back)

        writeFile.write(img_back)


    writeFile.release()


