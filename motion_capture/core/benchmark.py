import time
import torch as T
import torch.nn as nn



def model_speedtest(
        model: nn.Module, 
        input_shape: tuple, 
        ntests: int = 1000,
        device: str = "cuda"):
    
    print(f"speedtest engaged for {model.__class__}")
    print(f"\t# parameters: {sum([T.prod(T.tensor(p.shape)) for p in model.parameters()]) / 1000000} M")
    
    model = model.eval()

    test_input = T.rand(input_shape).to(device)

    with T.inference_mode():
    
        dt = time.perf_counter()
        for i in range(ntests):
            model(test_input)
            T.cuda.synchronize()
        print(f"\tfps: {1000 / (time.time() - dt)}")
