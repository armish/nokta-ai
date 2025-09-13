import torch

if torch.backends.mps.is_available():
    print("MPS backend is available.")
    mps_device = torch.device("mps")
    print(f"Using MPS device: {mps_device}")
    # You can now create tensors or move models to the MPS device
    x = torch.ones(5, device=mps_device)
    print(f"Tensor created on MPS device: {x}")
else:
    print("MPS backend is not available.")
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
