import os
import time
import torch
import pandas as pd

import config


# ==========================================================
# Parameter Analysis
# ==========================================================

def count_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable_params = (
        total_params - trainable_params
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }


# ==========================================================
# FLOPs Analysis
# ==========================================================

# def calculate_flops(model, input_size=(1, 3, 224, 224)):

#     try:

#         from thop import profile

#         dummy = torch.randn(input_size)

#         flops, params = profile(
#             model,
#             inputs=(dummy,),
#             verbose=False
#         )

#         return {
#             "flops": flops,
#             "params": params
#         }

#     except Exception as e:

#         print("THOP not installed.")
#         print(e)

#         return {
#             "flops": 0,
#             "params": 0
#         }

def calculate_flops(model, input_size=(1, 3, 224, 224)):

    try:

        from thop import profile
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location("active_network_module", config.NETWORK_FILE)
        _active_network = _ilu.module_from_spec(spec)
        spec.loader.exec_module(_active_network)

        # Build a fresh CPU copy of the architecture so THOP profiling
        # never has to deal with a CUDA model / CPU dummy mismatch.
        cpu_model = _active_network.CustomModel(num_classes=config.NUM_CLASSES)
        cpu_model.load_state_dict(model.state_dict())
        cpu_model = cpu_model.to("cpu")
        cpu_model.eval()

        dummy = torch.randn(input_size)

        flops, params = profile(
            cpu_model,
            inputs=(dummy,),
            verbose=False
        )

        return {
            "flops": flops,
            "params": params
        }

    except Exception as e:

        print("FLOPs calculation failed.")
        print(e)

        return {
            "flops": 0,
            "params": 0
        }


# ==========================================================
# Inference Time Analysis
# ==========================================================

def measure_inference_time(
    model,
    device,
    input_size=(1, 3, 224, 224),
    runs=100
):

    model.eval()

    dummy = torch.randn(
        input_size
    ).to(device)

    with torch.no_grad():

        for _ in range(20):
            _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        for _ in range(runs):
            _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    total_time = end - start

    avg_time = total_time / runs

    return avg_time


# ==========================================================
# FPS Analysis
# ==========================================================

def calculate_fps(avg_inference_time):

    if avg_inference_time == 0:
        return 0

    return 1.0 / avg_inference_time


# ==========================================================
# Performance Ratios
# ==========================================================

def performance_parameter_ratio(
    accuracy,
    total_parameters
):

    return accuracy / total_parameters


def performance_flops_ratio(
    accuracy,
    flops
):

    if flops == 0:
        return 0

    return accuracy / flops


# ==========================================================
# Save Full Complexity Report
# ==========================================================

def save_complexity_report(
    model,
    accuracy,
    device
):

    param_info = count_parameters(model)

    flops_info = calculate_flops(model)

    avg_time = measure_inference_time(
        model,
        device
    )

    fps = calculate_fps(avg_time)

    pp_ratio = performance_parameter_ratio(
        accuracy,
        param_info["total_parameters"]
    )

    pf_ratio = performance_flops_ratio(
        accuracy,
        flops_info["flops"]
    )

    result = pd.DataFrame([{
        "total_parameters":
            param_info["total_parameters"],

        "trainable_parameters":
            param_info["trainable_parameters"],

        "non_trainable_parameters":
            param_info["non_trainable_parameters"],

        "flops":
            flops_info["flops"],

        "avg_inference_time_sec":
            avg_time,

        "fps":
            fps,

        "performance_parameter_ratio":
            pp_ratio,

        "performance_flops_ratio":
            pf_ratio
    }])

    csv_path = os.path.join(
        config.TEST_RESULT_DIR,
        "complexity_report.csv"
    )

    result.to_csv(
        csv_path,
        index=False
    )

    txt_path = os.path.join(
        config.TEST_RESULT_DIR,
        "complexity_report.txt"
    )

    with open(txt_path, "w") as f:

        f.write(
            "MODEL COMPLEXITY ANALYSIS\n"
        )

        f.write("=" * 60 + "\n\n")

        for col in result.columns:

            f.write(
                f"{col}: "
                f"{result.iloc[0][col]}\n"
            )

    print(
        f"Complexity report saved to {csv_path}"
    )

    return result