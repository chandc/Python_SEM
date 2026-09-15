with open('/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/walkthrough.md', 'r') as f:
    content = f.read()

content = content.replace('**Inverse FFT:** We transpose back and apply `np.fft.irfft` to instantly recover the 3D physical domain.\n\n### 3D Performance Results', '### 3D Performance Results')

with open('/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/walkthrough.md', 'w') as f:
    f.write(content)
