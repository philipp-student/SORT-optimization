from frame_source import FrameSource, ServerFrameSource, DirectoryFrameSource
from carla_camera_frame import CARLACameraFrame

img_path = r'D:\Philipp Student\HRW\Repositories\fas-2-pietryga-student\img'

src = DirectoryFrameSource(img_path)
