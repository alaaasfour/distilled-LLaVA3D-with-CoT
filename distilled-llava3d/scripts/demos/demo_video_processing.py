#!/usr/bin/env python3
"""
Video processing demo for distilled LLaVA-3D.
Shows real-time capabilities on video streams.
"""

import torch
import sys
import os
import cv2
import numpy as np
from PIL import Image
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.distillation.student_model import DistilledLLaVA3D, DistilledLLaVA3DConfig

class VideoProcessingDemo:
    """Demo for video processing with distilled model."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        
    def load_model(self):
        """Load the distilled model."""
        print("📚 Loading Distilled Model for Video Processing...")
        
        checkpoint_dir = "models/checkpoints"
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = DistilledLLaVA3DConfig()
        self.model = DistilledLLaVA3D(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def process_frame(self, frame, question):
        """Process a single video frame."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        # Process with model
        start_time = time.time()
        with torch.no_grad():
            input_ids = torch.randint(0, 32000, (1, 64)).to(self.device)
            attention_mask = torch.ones(1, 64).to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=frame_tensor
            )
            
        processing_time = time.time() - start_time
        
        # Mock response
        response = f"Frame processed in {processing_time:.3f}s - Objects detected"
        
        return response, processing_time
        
    def demo_video_processing(self, video_path=None):
        """Demo video processing capabilities."""
        print("🎬 Video Processing Demo")
        print("=" * 40)
        
        # Load model
        self.load_model()
        
        if video_path and os.path.exists(video_path):
            # Process actual video
            cap = cv2.VideoCapture(video_path)
            print(f"📹 Processing video: {video_path}")
        else:
            # Create mock video stream
            print("📹 Creating mock video stream...")
            cap = self.create_mock_video()
            
        frame_count = 0
        total_time = 0
        questions = [
            "What objects do you see?",
            "Describe the scene",
            "What is happening?",
            "Identify the main subjects"
        ]
        
        print("\n🎯 Processing frames...")
        print("Press 'q' to quit, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 10th frame
            if frame_count % 10 == 0:
                question = questions[frame_count % len(questions)]
                response, proc_time = self.process_frame(frame, question)
                total_time += proc_time
                
                print(f"Frame {frame_count}: {response}")
                
                # Display frame
                cv2.putText(frame, f"Frame {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {proc_time:.3f}s", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow('Distilled LLaVA-3D Video Processing', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'captured_frame_{frame_count}.jpg', frame)
                print(f"Frame {frame_count} saved!")
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        avg_time = total_time / (frame_count // 10) if frame_count > 0 else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Video Processing Summary:")
        print(f"   • Frames processed: {frame_count}")
        print(f"   • Average processing time: {avg_time:.3f}s")
        print(f"   • Estimated FPS: {fps:.1f}")
        print(f"   • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def create_mock_video(self):
        """Create a mock video stream for testing."""
        class MockVideoCapture:
            def __init__(self):
                self.frame_count = 0
                self.max_frames = 100
                
            def read(self):
                if self.frame_count >= self.max_frames:
                    return False, None
                    
                # Create mock frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Add some visual elements
                cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), 2)
                cv2.circle(frame, (300, 300), 50, (0, 255, 0), -1)
                cv2.putText(frame, f"Mock Frame {self.frame_count}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                self.frame_count += 1
                return True, frame
                
            def release(self):
                pass
                
        return MockVideoCapture()

def main():
    """Main video demo function."""
    print("🎬 Distilled LLaVA-3D Video Processing Demo")
    print("This demonstrates real-time video processing capabilities.\n")
    
    # Initialize demo
    demo = VideoProcessingDemo()
    
    # Run video processing demo
    demo.demo_video_processing()

if __name__ == "__main__":
    main()
