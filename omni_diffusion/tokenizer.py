from .constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    IMG_TAG_TOKEN,
    PATCH_CONTEXT_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VID_CONTEXT_TOKEN,
    VID_END_TOKEN,
    VID_START_TOKEN,
    VID_TAG_TOKEN,
)


def update_tokenizer(tokenizer, audio_tokenizer_type=None):
    """
    Adds special tokens related to multimodal tasks (Image, Video, Patch, Bounding Box, etc.)
    to the tokenizer vocabulary.
    Also handles specific tokenizer updates for audio models like SenseVoice/GLM4Voice.
    """
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)

    if audio_tokenizer_type == "sensevoice_glm4voice":
        from .tokenizer_sensevoice_glm4voice import (
            update_tokenizer_for_sensevoice_glm4voice,
            SenseVoiceGLM4VoiceTokenizer,
        )
        return update_tokenizer_for_sensevoice_glm4voice(tokenizer)

    raise NotImplementedError


def get_audio_tokenizer(model_name_or_path, audio_tokenizer_type, flow_path=None, rank=None):
    """
    Factory function to initialize and return the appropriate audio tokenizer instance
    based on the provided type (e.g., 'sensevoice_glm4voice').
    """
    if audio_tokenizer_type is None:
        return None

    if audio_tokenizer_type == "sensevoice_glm4voice":
        from .tokenizer_sensevoice_glm4voice import (
            update_tokenizer_for_sensevoice_glm4voice,
            SenseVoiceGLM4VoiceTokenizer,
        )
        return SenseVoiceGLM4VoiceTokenizer(model_name_or_path, flow_path=flow_path, rank=rank)

    raise NotImplementedError
