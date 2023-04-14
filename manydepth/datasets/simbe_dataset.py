import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2


from .mono_dataset import MonoDataset


class SimbeDataset(MonoDataset):
    """Superclass for Simbe dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SimbeDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        # cam intrinsics
        self.K = np.array([[0.74877225, 0.0, 0.49140713, 0.0],
                           [0.0, 0.99643171, 0.4797712, 0.0], 
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        self.full_res_shape = (640, 480)

    def get_image_path(self, folder, frame_index):
        f_str = f"{str(frame_index)}{self.img_ext}"
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 2:
            frame_index = int(line[1])
        else:
            frame_index = 0


        return folder, frame_index

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, do_flip):
        f_str = f"{str(frame_index).zfill(8)}_d.png"
        image_path = os.path.join(self.data_path, folder, f_str)
        depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        depth_gt = depth.astype(np.float32)/1000.0

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

