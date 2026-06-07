import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
import time

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import InferInput, InferRequestedOutput
    from tritonclient.utils import InferenceServerException
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton client not available. Inference will be limited.")


class TritonInferenceClient:
    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "srgan",
        model_version: str = "1",
        input_name: str = "input",
        output_name: str = "output",
        timeout: int = 300,
        verbose: bool = False
    ):
        if not TRITON_AVAILABLE:
            raise ImportError("tritonclient is required for Triton inference")
        
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name
        self.timeout = timeout
        self.verbose = verbose
        
        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._client = grpcclient.InferenceServerClient(
                url=self.server_url,
                verbose=self.verbose
            )
            
            if not self._client.is_server_live():
                raise ConnectionError(f"Triton server at {self.server_url} is not live")
            
            if not self._client.is_model_ready(self.model_name, self.model_version):
                raise RuntimeError(f"Model {self.model_name} v{self.model_version} is not ready")
            
            logger.info(f"Connected to Triton server at {self.server_url}")
            logger.info(f"Model {self.model_name} v{self.model_version} is ready")
            
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise

    @property
    def client(self) -> grpcclient.InferenceServerClient:
        if self._client is None:
            self._connect()
        return self._client

    def _preprocess(self, images: np.ndarray) -> np.ndarray:
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 127.5 - 1.0
        
        images = np.transpose(images, (0, 3, 1, 2))
        return images.astype(np.float32)

    def _postprocess(self, outputs: np.ndarray) -> np.ndarray:
        outputs = np.transpose(outputs, (0, 2, 3, 1))
        outputs = (outputs + 1.0) * 127.5
        outputs = np.clip(outputs, 0, 255)
        return outputs.astype(np.uint8)

    def infer(self, images: np.ndarray) -> np.ndarray:
        batch_size = images.shape[0] if images.ndim == 4 else 1
        
        processed_input = self._preprocess(images)
        
        infer_input = InferInput(self.input_name, processed_input.shape, "FP32")
        infer_input.set_data_from_numpy(processed_input)
        
        infer_output = InferRequestedOutput(self.output_name)
        
        try:
            start_time = time.time()
            results = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[infer_input],
                outputs=[infer_output],
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            
            output_data = results.as_numpy(self.output_name)
            sr_images = self._postprocess(output_data)
            
            logger.debug(f"Inference completed in {elapsed*1000:.2f}ms for batch size {batch_size}")
            
            return sr_images
            
        except InferenceServerException as e:
            logger.error(f"Triton inference error: {e}")
            raise

    def infer_async(
        self,
        images: np.ndarray,
        callback=None,
        user_data: Dict = None
    ) -> None:
        processed_input = self._preprocess(images)
        
        infer_input = InferInput(self.input_name, processed_input.shape, "FP32")
        infer_input.set_data_from_numpy(processed_input)
        
        infer_output = InferRequestedOutput(self.output_name)
        
        def wrapped_callback(result, error):
            if error is not None:
                logger.error(f"Async inference error: {error}")
                if callback:
                    callback(None, error, user_data)
                return
            
            output_data = result.as_numpy(self.output_name)
            sr_images = self._postprocess(output_data)
            if callback:
                callback(sr_images, None, user_data)
        
        self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[infer_output],
            callback=wrapped_callback,
            timeout=self.timeout
        )

    def get_model_metadata(self) -> Dict:
        try:
            metadata = self.client.get_model_metadata(
                self.model_name,
                self.model_version
            )
            return {
                "name": metadata.name,
                "versions": list(metadata.versions),
                "platform": metadata.platform,
                "inputs": [
                    {"name": inp.name, "shape": list(inp.shape), "dtype": inp.datatype}
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {"name": out.name, "shape": list(out.shape), "dtype": out.datatype}
                    for out in metadata.outputs
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise

    def is_alive(self) -> bool:
        try:
            return self._client is not None and self._client.is_server_live()
        except:
            return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Closed Triton client connection")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
