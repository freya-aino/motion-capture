import time
import torch as T
import torch.nn as nn



def model_speedtest(
        model: nn.Module, 
        input_shape: tuple, 
        ntests: int = 1000,
        device: str = "cuda"):
    
    print(f"speedtest engaged for {model.__class__}")
    print(f"\t# parameters: {sum([p.numel() for p in model.parameters()]) / 1000000} M")
    
    model = model.eval()
    model = model.to(device)
    
    if type(input_shape[0]) == int:
        test_input = T.rand(input_shape).to(device)
    else:
        test_input = [T.rand(shape).to(device) for shape in input_shape]
    
    model(test_input)
    
    with T.inference_mode():
        dt = time.perf_counter()
        for _ in range(ntests):
            model(test_input)
            T.cuda.synchronize()
        print(f"\tfps: {ntests / (time.perf_counter() - dt)}")
