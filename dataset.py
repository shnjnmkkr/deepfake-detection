import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import math
from PIL import Image

class FrameExtractor:
    """Utility class for extracting frames from videos."""
    def __init__(self, num_frames: int = 5):
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
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_frames = 15  # Fixed number of frames
        
        # Set paths
        self.real_dir = os.path.join(root_dir, split, 'real')
        self.fake_dir = os.path.join(root_dir, split, 'fake')
        
        # Get all video files
        self.real_videos = [f for f in os.listdir(self.real_dir) if f.endswith('.mp4')]
        self.fake_videos = [f for f in os.listdir(self.fake_dir) if f.endswith('.mp4')]
        
        # Create video list with labels
        self.videos = []
        for video in self.real_videos:
            self.videos.append((os.path.join(self.real_dir, video), 0))  # 0 for real
        for video in self.fake_videos:
            self.videos.append((os.path.join(self.fake_dir, video), 1))  # 1 for fake
            
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
        
        # Calculate frame indices to extract
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
                    # Convert PIL Image to numpy array, then to tensor
                    frame = np.array(frame)
                    frame = torch.from_numpy(frame).float() / 255.0
                    frame = frame.permute(2, 0, 1)  # Convert to (C, H, W)
                
                frames.append(frame)
            else:
                # Use a zero tensor as fallback
                frames.append(torch.zeros(3, 160, 160))
        
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        
        try:
            # Extract frames from video
            frames = self.extract_frames(video_path)
            
            # Stack frames and ensure correct shape
            frames = torch.stack(frames)  # Shape: (15, C, H, W)
            frames = frames.permute(0, 2, 3, 1)  # Convert to (15, H, W, C)
            
            return frames, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros(self.num_frames, 160, 160, 3), torch.tensor(label, dtype=torch.float32)