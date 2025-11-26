from pathlib import Path
from typing import Any
import platform

from backend.device import is_openvino_device, is_openvino_available
from backend.tiny_autoencoder import get_tiny_autoencoder_repo_id
from constants import DEVICE, LCM_DEFAULT_MODEL_OPENVINO
from paths import get_base_folder_name

# Only import OpenVINO modules if available (not on Mac ARM)
_openvino_imports_available = False

if is_openvino_available():
    try:
        from optimum.intel.openvino import OVDiffusionPipeline
        from optimum.intel.openvino.modeling_diffusion import (
            OVModelVae,
            OVModelVaeDecoder,
            OVModelVaeEncoder,
        )
        _openvino_imports_available = True
    except ImportError as e:
        print(f"OpenVINO optimum imports not available: {e}")
        _openvino_imports_available = False

if _openvino_imports_available and is_openvino_device():
    from huggingface_hub import snapshot_download
    from optimum.intel.openvino.modeling_diffusion import (
        OVBaseModel,
        OVStableDiffusionImg2ImgPipeline,
        OVStableDiffusionPipeline,
        OVStableDiffusionXLImg2ImgPipeline,
        OVStableDiffusionXLPipeline,
    )


def ov_load_tiny_autoencoder(
    pipeline: Any,
    use_local_model: bool = False,
):
    if not _openvino_imports_available:
        raise RuntimeError("OpenVINO is not available on this platform (Mac ARM)")
    taesd_dir = snapshot_download(
        repo_id=get_tiny_autoencoder_repo_id(pipeline.__class__.__name__),
        local_files_only=use_local_model,
    )
    vae_decoder = OVModelVaeDecoder(
        model=OVBaseModel.load_model(f"{taesd_dir}/vae_decoder/openvino_model.xml"),
        parent_pipeline=pipeline,
        model_name="vae_decoder",
    )
    vae_encoder = OVModelVaeEncoder(
        model=OVBaseModel.load_model(f"{taesd_dir}/vae_encoder/openvino_model.xml"),
        parent_pipeline=pipeline,
        model_name="vae_encoder",
    )
    pipeline.vae = OVModelVae(
        decoder=vae_decoder,
        encoder=vae_encoder,
    )
    pipeline.vae.config.scaling_factor = 1.0


def get_ov_text_to_image_pipeline(
    model_id: str = LCM_DEFAULT_MODEL_OPENVINO,
    use_local_model: bool = False,
) -> Any:
    if not _openvino_imports_available:
        raise RuntimeError("OpenVINO is not available on this platform (Mac ARM)")
    if "xl" in get_base_folder_name(model_id).lower():
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )
    else:
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )

    return pipeline


def get_ov_image_to_image_pipeline(
    model_id: str = LCM_DEFAULT_MODEL_OPENVINO,
    use_local_model: bool = False,
) -> Any:
    if not _openvino_imports_available:
        raise RuntimeError("OpenVINO is not available on this platform (Mac ARM)")
    if "xl" in get_base_folder_name(model_id).lower():
        pipeline = OVStableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )
    else:
        pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )
    return pipeline


def get_ov_diffusion_pipeline(
    model_id: str,
    use_local_model: bool = False,
) -> Any:
    if not _openvino_imports_available:
        raise RuntimeError("OpenVINO is not available on this platform (Mac ARM)")
    pipeline = OVDiffusionPipeline.from_pretrained(
        model_id,
        local_files_only=use_local_model,
        ov_config={"CACHE_DIR": ""},
        device=DEVICE.upper(),
    )
    return pipeline
