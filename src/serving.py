import os
import zipfile
import json
from typing import Any, Dict, Union
import numpy as np
import transformers

import mlrun.artifacts
from mlrun.serving.v2_serving import V2ModelServer

import torch
import evaluate

SUBJECT_MARK = "Subject: "
CONTENT_MARK = "\nContent: "
PROMPT_FORMAT = SUBJECT_MARK + "{}" + CONTENT_MARK


def preprocess(request: dict) -> dict:
    # Read bytes:
    if isinstance(request, bytes):
        request = json.loads(request)

    # Get the prompt:
    prompt = request.pop("prompt")
    
    # Format the prompt as subject:
    prompt = PROMPT_FORMAT.format(str(prompt))
    
    # Update the request and return:
    request = {"inputs": [{"prompt": [prompt], **request}]}
    return request


class LLMModelServer(V2ModelServer):
    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_class: str = None,
        tokenizer_class: str = None,
        # Load from MLRun args:
        model_path: str = None,
        # Load from hub args:
        model_name: str = None,
        tokenizer_name: str = None,
        # Deepspeed args:
        use_deepspeed: bool = False,
        n_gpus: int = 1,
        is_fp16: bool = True,
        # Inference args:
        **class_args,
    ):
        # Initialize the base server:
        super(LLMModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            **class_args,
        )

        # Save class names:
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class

        # Save hub loading parameters:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        # Save deepspeed parameters:
        self.use_deepspeed = use_deepspeed
        self.n_gpus = n_gpus
        self.is_fp16 = is_fp16

        # Prepare variables for future use:
        self.model = None
        self.tokenizer = None
        self._model_class = None
        self._tokenizer_class = None

    def load(self):
        # Get classes:
        self._model_class = getattr(transformers, self.model_class)
        self._tokenizer_class = getattr(transformers, self.tokenizer_class)

        # Load the model and tokinzer:
        if self.model_path:
            self._load_from_mlrun()
        else:
            self._load_from_hub()

        # Use deepspeed if needed:
        if self.use_deepspeed:
            import deepspeed

            self.model = deepspeed.init_inference(
                model=self.model,
                mp_size=self.n_gpus,
                dtype=torch.float16 if self.is_fp16 else torch.float32,
                replace_method="auto",
                replace_with_kernel_inject=True,
            )

    def _load_from_mlrun(self):
        # Get the model artifact and file:
        (
            model_file,
            model_artifact,
            extra_data,
        ) = mlrun.artifacts.get_model(self.model_path)

        # Read the name:
        model_name = model_artifact.spec.db_key

        # Extract logged model files:
        model_directory = os.path.join(os.path.dirname(model_file), model_name)
        with zipfile.ZipFile(model_file, "r") as zip_file:
            zip_file.extractall(model_directory)

        # Loading the saved pretrained tokenizer and model:
        self.tokenizer = self._tokenizer_class.from_pretrained(model_directory)
        self.model = self._model_class.from_pretrained(model_directory)

    def _load_from_hub(self):
        # Loading the pretrained tokenizer and model:
        self.tokenizer = self._tokenizer_class.from_pretrained(
            self.tokenizer_name,
            model_max_length=512,
        )
        self.model = self._model_class.from_pretrained(self.model_name)

    def predict(self, request: Dict[str, Any]) -> dict:
        # Get the inputs:
        kwargs = request["inputs"][0]
        prompt = kwargs.pop("prompt")[0]

        # Tokenize:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.use_deepspeed:
            input_ids = input_ids.cuda()

        # Create the attention mask and pad token id:
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = self.tokenizer.eos_token_id

        # Infer through the model:
        output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            **kwargs
        )

        # Detokenize:
        prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return {"prediction": prediction, "prompt": prompt}

    def explain(self, request: Dict) -> str:
        return f"LLM model server named {self.name}"


def postprocess(inputs: dict) -> dict:
    # Read the prediction:
    prediction = inputs["outputs"]["prediction"]

    # Look for a 'Content: ' mark to know the model found the subject, otherwise, it is probably garbage:
    content_index = prediction.find(CONTENT_MARK)
    if content_index == -1:
        output = f"I'm not sure about it but I'll do my best: {prediction}"
    else:
        output = prediction[content_index + len(CONTENT_MARK) :]

    return {"inputs": [{"prediction": output.strip(), "prompt": inputs["outputs"]["prompt"]}]}


class ToxicityClassifierModelServer(V2ModelServer):
    def __init__(self, context, name: str, threshold: float = 0.7, **class_args):
        # Initialize the base server:
        super(ToxicityClassifierModelServer, self).__init__(
            context=context,
            name=name,
            model_path=None,
            **class_args,
        )

        # Store the threshold of toxicity:
        self.threshold = threshold

    def load(self):
        self.model = evaluate.load("toxicity", module_type="measurement")

    def predict(self, inputs: Dict) -> str:
        # Read the user's input and model output:
        prediction = inputs["inputs"][0]["prediction"]
        prompt = inputs["inputs"][0]["prompt"]

        # Infer through the evaluator model:
        result = self.model.compute(predictions=[prediction, prompt])["toxicity"]
        if any(np.array(result) > self.threshold):
            return "This bot do not respond to toxicity."

        return prediction

    def explain(self, request: Dict) -> str:
        return f"Text toxicity classifier server named {self.name}"
