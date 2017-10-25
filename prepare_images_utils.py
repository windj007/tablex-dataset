import numpy, glob, os, random, h5py
from PIL import Image
from joblib import Parallel, delayed


def binarize_tensor(arr, threshold=0.7):
    return numpy.where(arr >= threshold, 1, 0)


def to_grayscale(arr):
    return arr.mean(axis=-1)


def identity(x):
    return x


def rgba_to_rgb(image, color=(255, 255, 255)):
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def la_to_l(image, color=255):
    image.load()  # needed for split()
    background = Image.new('L', image.size, color)
    background.paste(image, mask=image.split()[-1])  # 3 is the alpha channel
    return background


def load_image_opaque(fname, mode='RGB'):
    img = Image.open(fname)
    if not mode is None and img.mode != mode:
        if img.mode == 'LA':
            img = la_to_l(img)
        elif img.mode == 'RGBA':
            img = rgba_to_rgb(img)
        img = img.convert(mode)
    return img


def load_image_to_array(fname, proc=identity, mode=None):
    return proc(numpy.array(load_image_opaque(fname, mode=mode)))


footer  = numpy.array([255,   0,   0,   0], dtype='uint8')
header  = numpy.array([  0, 255,   0,   0], dtype='uint8')
body    = numpy.array([  0,   0, 255,   0], dtype='uint8')
nothing = numpy.array([  0,   0,   0, 255], dtype='uint8')

def color_to_gold_output(pixel):
    if all(pixel > 200): # body
        return body
    elif pixel[1] > 200: # header
        return header
    elif pixel[0] > 200: # footer
        return footer
    else:
        return nothing


footer_and_header2  = numpy.array([255,   0,   0], dtype='uint8')
body2               = numpy.array([  0, 255,   0], dtype='uint8')
nothing2            = numpy.array([  0,   0, 255], dtype='uint8')

def color_to_gold_output_join_fh(pixel):
    if all(pixel > 200): # body
        return body2
    elif pixel[1] > 200:
        return footer_and_header2
    elif pixel[0] > 200:
        return footer_and_header2
    else:
        return nothing2


def prepare_multioutput(image):
    return numpy.apply_along_axis(color_to_gold_output, -1, image)


def prepare_multioutput2(image):
    result = numpy.zeros(image.shape[:-1] + (3,), dtype='uint8')

    body_mask = image.min(-1) > 200 # all > 200
    body_x, body_y = numpy.where(body_mask)
    result[body_x, body_y, 1] = 255

    fh_mask = (image[:, :, :-1].max(-1) > 200) & ~body_mask
    fh_x, fh_y = numpy.where(fh_mask) # at least one > 200
    result[fh_x, fh_y, 0] = 255

    result[:, :, 2] = 255 - result.sum(-1) # rest are nothing
    return result
    # return numpy.apply_along_axis(color_to_gold_output_join_fh, -1, image)


def prepare_multioutput3(image):
    result = numpy.array(image)
    result[:, :, -1] = 255 - result[:, :, :-1].sum(-1)
    return result


def read_images_to_tensor(filenames, n_jobs=-1, proc=identity, mode='RGB'):
    data = list(Parallel(n_jobs=n_jobs)(delayed(load_image_to_array)(fname, proc=proc, mode=mode)
                                        for fname in filenames))
    return numpy.stack(data).astype('float32') / 255.0


def load_dataset(dirname, n_jobs=-1, take_n=None, in_mode='L', out_mode='RGB', in_proc_func=identity, out_proc_func=prepare_multioutput3):
    prefixes = [fname[:-7] for fname
                in glob.glob(os.path.join(dirname, '*_in.png'))
                #in glob.glob('./data/5_ready/train/12147373-0005_*in.png')
               ]
    random.shuffle(prefixes)
    if not take_n is None:
        take_n = min(len(prefixes), take_n)
    else:
        take_n = len(prefixes)
    prefixes = prefixes[:take_n]
    in_files = [prefix + '_in.png' for prefix in prefixes]
    out_files = [prefix + '_out.png' for prefix in prefixes]

    in_data = read_images_to_tensor(in_files, n_jobs=n_jobs, proc=in_proc_func, mode=in_mode)
    out_data = read_images_to_tensor(out_files, n_jobs=n_jobs, proc=out_proc_func, mode=out_mode)
    return (numpy.expand_dims(in_data, -1),
            out_data)


def convert_directory_to_hdf5(dirname, out_file, chunk_size=10000, n_jobs=-1, in_mode='L', out_mode='RGB', in_proc_func=identity, out_proc_func=prepare_multioutput2, shuffle=True):
    prefixes = [fname[:-7] for fname
                in glob.glob(os.path.join(dirname, '*_in.png'))
                #in glob.glob('./data/5_ready/train/12147373-0005_*in.png')
               ]
    if shuffle:
        random.shuffle(prefixes)
    else:
        prefixes.sort()
    samples_n = len(prefixes)

    in_files = [prefix + '_in.png' for prefix in prefixes]
    out_files = [prefix + '_out.png' for prefix in prefixes]

    in_shape = load_image_to_array(in_files[0],
                                   proc=in_proc_func,
                                   mode=in_mode).shape + (1,)
    out_shape = load_image_to_array(out_files[0],
                                    proc=out_proc_func,
                                    mode=out_mode).shape

    with h5py.File(out_file, 'w') as f:
        in_data = f.create_dataset('in_data', (samples_n,) + in_shape, chunks=True)
        out_data = f.create_dataset('out_data', (samples_n,) + out_shape, chunks=True)

        for chunk_start in range(0, samples_n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, samples_n)
            in_data[chunk_start:chunk_end] = numpy.expand_dims(
                read_images_to_tensor(in_files[chunk_start:chunk_end],
                                      n_jobs=n_jobs,
                                      proc=in_proc_func,
                                      mode=in_mode),
                -1)
            out_data[chunk_start:chunk_end] = read_images_to_tensor(out_files[chunk_start:chunk_end],
                                                                    n_jobs=n_jobs,
                                                                    proc=out_proc_func,
                                                                    mode=out_mode)
        


def arr_to_img(arr):
    return Image.fromarray((arr * 255).astype('uint8'))


def arr3_to_img(arr):
    arr = numpy.rollaxis(arr, 0, 2).reshape((arr.shape[1], arr.shape[0] * arr.shape[2])+arr.shape[3:])
    print(arr.shape)
    return arr_to_img(arr)
