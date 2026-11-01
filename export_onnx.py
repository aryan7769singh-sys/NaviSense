"""Export the trained VelNet to ONNX so the phone can run it."""
import pickle, numpy as np, torch
from pathlib import Path
from proto.velnet import VelNet, CHANNELS

d = pickle.load(open('artifacts/full_run.pkl','rb'))
b = d['bundle']
model = b.model.eval()
W = b.window

out = Path('app/src/main/assets'); out.mkdir(parents=True, exist_ok=True)
dummy = torch.randn(1, CHANNELS, W)

torch.onnx.export(
    model, dummy, out/'velnet.onnx',
    input_names=['imu'], output_names=['speed','log_var'],
    dynamic_axes={'imu':{0:'batch'},'speed':{0:'batch'},'log_var':{0:'batch'}},
    opset_version=17,
)
# Normalisation must travel with the model or inference is silently wrong.
np.savez(out/'velnet_norm.npz', mean=b.mean, std=b.std,
         window=np.array([W]), channels=np.array([CHANNELS]))

print(f"window={W} channels={CHANNELS}")
print(f"mean={np.round(b.mean,3)}")
print(f"std ={np.round(b.std,3)}")

import onnxruntime as ort
sess = ort.InferenceSession(str(out/'velnet.onnx'))
x = ((np.random.randn(1,W,CHANNELS).astype(np.float32)-b.mean)/b.std).transpose(0,2,1).astype(np.float32)
o = sess.run(None, {'imu': x})
with torch.no_grad():
    t = model(torch.tensor(x))
print(f"\nONNX vs PyTorch: speed {o[0][0]:.6f} vs {float(t[0][0]):.6f}  "
      f"diff {abs(o[0][0]-float(t[0][0])):.2e}")
print(f"onnx size: {(out/'velnet.onnx').stat().st_size/1024:.0f} KB")
