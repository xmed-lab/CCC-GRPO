from abc import ABC, abstractmethod


class VLMBaseModule(ABC):
    def post_model_init(self, model, processing_class):
        pass

    def is_embeds_input(self):
        return False

    @abstractmethod
    def get_vlm_key(self):
        pass

    @abstractmethod
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        pass

    @abstractmethod
    def get_processing_class(self):
        pass

    @abstractmethod
    def get_vision_modules_keywords(self):
        pass

    @abstractmethod
    def get_custom_multimodal_keywords(self):
        pass

    @abstractmethod
    def get_non_generate_params(self):
        pass

    @abstractmethod
    def get_custom_processing_keywords(self):
        pass

    @abstractmethod
    def prepare_prompt(self, processing_class, inputs):
        pass

    @abstractmethod
    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text,
        images,
        return_tensors,
        padding,
        padding_side,
        add_special_tokens,
    ):
        pass
