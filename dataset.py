import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import math
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

class FrameExtractor:
    """Utility class for extracting frames from videos."""
    def __init__(self, num_frames: int = 15):
        self.num_frames = num_frames

    def extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            numpy array of shape (num_frames, height, width, channels)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Calculate frame indices to extract
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                raise ValueError(f"Failed to read frame {idx} from {video_path}")
        
        cap.release()
        return np.array(frames)

class SuperResolution:
    """Real-ESRGAN super-resolution model wrapper."""
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Real-ESRGAN model
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        
        if model_path is None:
            # Download pretrained model if not provided
            model_path = load_file_from_url(
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model_dir='weights'
            )
        
        self.model.load_state_dict(torch.load(model_path)['params'])
        self.model.eval()
        self.model.to(self.device)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply super-resolution to a single frame.
        
        Args:
            frame: numpy array of shape (height, width, channels)
            
        Returns:
            Enhanced frame of shape (height*4, width*4, channels)
        """
        # Convert to tensor
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame = frame.to(self.device)
        
        with torch.no_grad():
            output = self.model(frame)
        
        # Convert back to numpy
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        return output

class DeepFakeDataset(Dataset):
    """Dataset class for DeepFake detection."""
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_frames: int = 15,
        use_sr: bool = False,
        transform = None
    ):
        """
        Args:
            root_dir: Path to dataset root directory
            split: 'train' or 'test'
            num_frames: Number of frames to extract per video
            use_sr: Whether to apply super-resolution
            transform: Optional transforms to apply to frames
        """
        self.root_dir = os.path.join(root_dir, split)
        self.num_frames = num_frames
        self.use_sr = use_sr
        self.transform = transform
        
        self.frame_extractor = FrameExtractor(num_frames)
        self.sr_model = SuperResolution() if use_sr else None
        
        # Get all video paths and labels
        self.video_paths = []
        self.labels = []
        
        # Process real videos
        real_dir = os.path.join(self.root_dir, 'real')
        for video_name in os.listdir(real_dir):
            if video_name.endswith(('.mp4', '.avi', '.mov')):
                self.video_paths.append(os.path.join(real_dir, video_name))
                self.labels.append(0)
        
        # Process fake videos
        fake_dir = os.path.join(self.root_dir, 'fake')
        for video_name in os.listdir(fake_dir):
            if video_name.endswith(('.mp4', '.avi', '.mov')):
                self.video_paths.append(os.path.join(fake_dir, video_name))
                self.labels.append(1)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self.frame_extractor.extract_frames(video_path)
        
        # Apply super-resolution if enabled
        if self.use_sr:
            frames = np.array([self.sr_model.enhance(frame) for frame in frames])
        
        # Apply transforms if any
        if self.transform:
            frames = np.array([self.transform(frame) for frame in frames])
        
        # Convert to tensor
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)  # (num_frames, channels, height, width)
        
        return frames, label 