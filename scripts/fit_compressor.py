import sys
import os
sys.path.append(os.getcwd())

import h5py
from src.data.compression import WaveformCompressor

def main():
    train_path = 'training_data/waveforms_train.h5'
    print(f"Loading first 15,000 samples from {train_path} to fit SVD parameters reliably...")
    with h5py.File(train_path, 'r') as f:
        real = f['hp_real'][:15000]
        imag = f['hp_imag'][:15000]
        
    print("Fitting new compressor using Real and Imaginary domains...")
    comp = WaveformCompressor(n_basis=100)
    comp.fit(real, imag)
    
    os.makedirs('checkpoints', exist_ok=True)
    comp.save('checkpoints/compressor.h5')
    print("Successfully replaced checkpoints/compressor.h5 with Re/Im compressor!")

if __name__ == '__main__':
    main()
