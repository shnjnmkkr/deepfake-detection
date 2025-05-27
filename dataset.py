import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from PIL import Image

# Optional super-resolution imports
USE_SR = False  # Global flag to enable/disable super-resolution
if USE_SR:
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        SR_AVAILABLE = True
    except ImportError:
        print("Warning: Super-resolution modules not available. Install with: pip install basicsr realesrgan")
        SR_AVAILABLE = False
else:
    SR_AVAILABLE = False

class FrameExtractor:
    """Utility class for extracting frames from videos."""
    def __init__(self, num_frames: int = 5):
        self.num_frames = num_frames

    def extract_frames(self, video_path: str) -> np.ndarray:
        """Extract frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
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
        if not SR_AVAILABLE:
            raise ImportError("Super-resolution modules not available")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        
        if model_path is None:
            model_path = load_file_from_url(
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model_dir='weights'
            )
        
        self.model.load_state_dict(torch.load(model_path)['params'])
        self.model.eval()
        self.model.to(self.device)
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply super-resolution to a single frame."""
        output, _ = self.upsampler.enhance(frame)
        return output

class DeepFakeDataset(Dataset):
    """Dataset class for DeepFake detection."""
    def __init__(self, root_dir, split='train', transform=None, num_frames=8, use_sr=False):
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.use_sr = use_sr and SR_AVAILABLE
        
        if self.use_sr and not SR_AVAILABLE:
            print("Warning: Super-resolution requested but not available. Continuing without SR.")
        
        # Initialize SR model if needed
        self.sr_model = SuperResolution() if self.use_sr else None
        
        # Set paths
        self.real_dir = os.path.join(self.root_dir, split, 'real')
        self.fake_dir = os.path.join(self.root_dir, split, 'fake')
        
        # Check directories
        if not os.path.exists(self.real_dir):
            raise FileNotFoundError(f"Real directory not found: {self.real_dir}")
        if not os.path.exists(self.fake_dir):
            raise FileNotFoundError(f"Fake directory not found: {self.fake_dir}")
        
        # Get video files
        self.real_videos = [f for f in os.listdir(self.real_dir) if f.endswith('.mp4')]
        self.fake_videos = [f for f in os.listdir(self.fake_dir) if f.endswith('.mp4')]
        
        # Create video list with labels
        self.videos = []
        for video in self.real_videos:
            self.videos.append((os.path.join(self.real_dir, video), 0))
        for video in self.fake_videos:
            self.videos.append((os.path.join(self.fake_dir, video), 1))
            
        print(f"Found {len(self.videos)} videos in {split} split")
        print(f"Real videos: {len(self.real_videos)}, Fake videos: {len(self.fake_videos)}")

    def __len__(self):
        return len(self.videos)

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = np.array(frame)
                    frame = torch.from_numpy(frame).float() / 255.0
                    frame = frame.permute(2, 0, 1)
                
                frames.append(frame)
            else:
                frames.append(torch.zeros(3, 160, 160))
        
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        
        try:
            frames = self.extract_frames(video_path)
            frames = torch.stack(frames)
            frames = frames.permute(0, 2, 3, 1)
            
            return frames, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return torch.zeros(self.num_frames, 160, 160, 3), torch.tensor(label, dtype=torch.float32)